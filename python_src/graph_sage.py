
"""
Implement training of the GraphSage algorithm for
assigning logits to outgoing links
- First, we create a "base" embedding for every node
in the network based on the network topology and
link bandwidths.
- Then we have a linear layer that is a function
of the base embedding of the node, the utilization
ratio of the outgoing link, and information in the
routing table entry.
"""

import tensorflow as tf
import numpy as np
import sys, os
import json
import pdb
from simulate_bgp import (get_average_delay_sage,
                          initialize_subset,
                          graph_diameter,
                          initialize_routing_tables)


MAX_BANDWIDTH = 2600000
MAX_LENGTH = 20037500


def initialize_graph(num_nodes):
    np.random.seed(np.random.randint(10000, 20000))
    graph_subset = initialize_subset(num_nodes)
    diameter = graph_diameter(graph_subset)
    initialize_routing_tables(graph_subset, diameter)
    return graph_subset


class WeightModel(tf.keras.Model):
    def __init__(self, K, k_width, agg_width):
        """
        It is nice that we can have a relatively small number of parametersx
        """
        super(WeightModel, self).__init__()
        self.K = K
        self.k_width = k_width
        self.agg_width = agg_width
        self.node_matrices = []
        for _ in range(K):
            self.node_matrices.append(tf.keras.layers.Dense(k_width, activation="relu"))
        self.aggregators = []
        for _ in range(K):
            self.aggregators.append(tf.keras.layers.Dense(agg_width, activation="relu"))
        self.output_layer = tf.keras.layers.Dense(1)
        self(self.__get_init_subset__())
        

    def __compute_base_embeddings__(self, nodes):
        """
        This function uses the GraphSage algorithm to form an embedding for
        each node. This embedding is created by Sampling and Aggregating. 
        """
        for node_id, node in nodes.items():
            node.initial_embedding = tf.constant([len(node.neighbors) / len(nodes)], dtype=tf.float32)
        for k in range(self.K):
            for node_id, node in nodes.items():
                node_embedding = node.initial_embedding if k == 0 else node.embedding
                neighbor_embeddings = []
                for i, neighbor in enumerate(node.neighbors):
                    neighbor_embedding = neighbor.initial_embedding if k == 0 else neighbor.embedding
                    neighbor_embeddings.append(
                        tf.concat([neighbor_embedding,
                                   tf.constant([node.links[i].bandwidth / MAX_BANDWIDTH,
                                                node.links[i].length / MAX_LENGTH,
                                                node.links[i].utilization / node.links[i].bandwidth],
                                               dtype=tf.float32)], axis=0)[None, :])
                projected_neighbors = self.aggregators[k](tf.concat(neighbor_embeddings, axis=0))
                aggregated_neighbors = tf.reduce_max(projected_neighbors, axis=0)
                node.embedding = self.node_matrices[k](tf.concat([aggregated_neighbors,
                                                                  node_embedding], axis=0)[None, :])[0]


    def __compute_logits__(self, nodes):
        """
        This function uses the embeddings at each node to compute a logit that
        is used for the weights. These logits are then used in the softmax operation
        to direct traffic to a nodes neighbors
        """
        for node_id, node in nodes.items():
            for k, entry in node.routing_table.items():
                entry_embedding = tf.constant([entry["d"] / MAX_LENGTH,
                                               entry["bad"] / MAX_LENGTH,
                                               entry["b"] / MAX_BANDWIDTH,
                                               entry["bah"] / len(nodes),
                                               entry["dab"] / MAX_BANDWIDTH,
                                               entry["h"] / len(nodes)], tf.float32)
                total_embedding = tf.concat([node.embedding, entry_embedding], axis=0)
                entry["weight"] = self.output_layer(total_embedding[None, :])[0][0]

                
    def __call__(self, nodes):
        """
        Call the GraphSage model on the given nodes with the given aggregators, layers,
        and output_layer. (The additional parameters may be different than the ones defined
        in the object)
        """
        self.__compute_base_embeddings__(nodes)
        self.__compute_logits__(nodes)


    def train_step_robust(self, num_nodes, n_variations, std=0.1, alpha=0.01):
        """
        Same as train_step but a new topology is created at every variation
        """
        weights = []
        orig_weights = []
        perterbations = []
        for weight_matrix in self.trainable_weights:
            orig_weights.append(weight_matrix)
            weight_variations = tf.concat([tf.identity(weight_matrix)[None, :] for _ in range(n_variations)],
                                          axis=0)
            perterbation = tf.random.normal(weight_variations.shape)
            perterbations.append(perterbation)
            weight_variations += perterbation * std
            weights.append(weight_variations)

        variation_results = np.zeros(n_variations, dtype=np.float32)
        for i in range(n_variations):
            np.random.seed(np.random.randint(0, 1000))
            p_communicate = np.random.random()
            mean_traffic = np.random.random() * 10000
            nodes = initialize_graph(num_nodes)
            print("\rVariation", i, "of", n_variations, end="") 
            self.set_weights([w[i] for w in weights])
            reward = get_average_delay_sage(self, nodes, p_communicate, mean_traffic, mean_traffic / 4)
            variation_results[i] = -1 * reward # We try to maximize the negation
        print()
        A = (variation_results - np.mean(variation_results)) / np.std(variation_results)
        result_weights = []
        for i in range(len(orig_weights)):
            nice_A = tf.reshape(A, [tf.size(A)] + [1] * (len(perterbations[i].shape) - 1))
            weighted_perterbations = tf.reduce_sum(perterbations[i] * nice_A, axis=0)
            result_weights.append(orig_weights[i] + alpha/(n_variations*std) * weighted_perterbations)
        #result_weights = tf.nn.softmax(-1 * np.array(variation_results, dtype=np.float32))
        #final_weights = [np.average(w, axis=0, weights=result_weights) for w in weights]
        self.set_weights(result_weights)
        return np.mean(variation_results)

        


    def train_step(self, nodes, n_variations, std=0.1, alpha=0.0001):
        """
        Train the network by adding gaussian noise to its parameters, simulating BGP,
        and taking an average of the parameters weighted by their performance.
        """
        weights = []
        orig_weights = []
        perterbations = []
        for weight_matrix in self.trainable_weights:
            orig_weights.append(weight_matrix)
            weight_variations = tf.concat([tf.identity(weight_matrix)[None, :] for _ in range(n_variations)],
                                          axis=0)
            perterbation = tf.random.normal(weight_variations.shape)
            perterbations.append(perterbation)
            weight_variations += perterbation * std
            weights.append(weight_variations)

        variation_results = np.zeros(n_variations, dtype=np.float32)
        np.random.seed(np.random.randint(0, 1000))
        p_communicate = np.random.random()
        mean_traffic = np.random.random() * 10000
        for i in range(n_variations):
            print("\rVariation", i, "of", n_variations, end="") 
            self.set_weights([w[i] for w in weights])
            reward = get_average_delay_sage(self, nodes, p_communicate, mean_traffic, mean_traffic / 4)
            variation_results[i] = -1 * reward # We try to maximize the negation
        print()
        A = (variation_results - np.mean(variation_results)) / np.std(variation_results)
        result_weights = []
        for i in range(len(orig_weights)):
            nice_A = tf.reshape(A, [tf.size(A)] + [1] * (len(perterbations[i].shape) - 1))
            weighted_perterbations = tf.reduce_sum(perterbations[i] * nice_A, axis=0)
            result_weights.append(orig_weights[i] + alpha/(n_variations*std) * weighted_perterbations)
        #result_weights = tf.nn.softmax(-1 * np.array(variation_results, dtype=np.float32))
        #final_weights = [np.average(w, axis=0, weights=result_weights) for w in weights]
        self.set_weights(result_weights)
        return np.mean(variation_results)

        
    def get_config(self):
        config = {
            "K": self.K,
            "k_width": self.k_width,
            "agg_width": self.agg_width
        }
        return config


    def save(self, path):
        """
        Save this model to the given path
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "config"), "w") as fout:
            fout.write(json.dumps(self.get_config(), indent=2))
        weights = self.get_weights()
        weights_args = {}
        for i, weight_arr in enumerate(weights):
            weights_args[str(i)] = weight_arr
        np.savez_compressed(os.path.join(path, "weights"),
                            **weights_args)


    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "config"), "r") as fin:
            config = json.load(fin)
        model = cls.from_config(config)
        weights = np.load(os.path.join(path, "weights.npz"))
        i = 0
        weights_arr = []
        while str(i) in weights:
            weights_arr.append(weights[str(i)])
            i += 1
        model.set_weights(weights_arr)
        return model
        

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    
    def __get_init_subset__(self):
        graph_subset = initialize_subset(10)
        diameter = graph_diameter(graph_subset)
        initialize_routing_tables(graph_subset, diameter)
        return graph_subset
            
            
            
        
