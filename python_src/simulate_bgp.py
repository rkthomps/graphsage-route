
"""
This module will simulate BGP over a subset of the topology
created by create_peering_topology.py
"""
import json
import sys, os
import pdb
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt

from create_peering_topology import (TOPOLOGY_DATA_LOC,
                                     LINK_DATA_LOC,
                                     Node, Link,
                                     set_link_lengths,
                                     calc_haversine)


RESULTS_DIR = os.path.join("results")


class Found(Exception): pass
def select_subset(all_nodes, num_nodes):
    """
    Select a subset of num_nodes nodes
    to use for the simulated topology. Randomly
    select a node and perform a breadth-first search

    One mistake I made at first: I forgot to get rid of neighbors
    links to nodes not in the network.
    """
    try:        
        while True:
            root_node_key = np.random.choice(list(all_nodes.keys()))
            next_nodes = [all_nodes[root_node_key]]
            selected = {}
            while len(next_nodes) > 0:
                cur_nodes = next_nodes
                next_nodes = []
                for node in cur_nodes:
                    selected[node.node_id] = node
                    if len(selected) == num_nodes:
                        raise Found
                    for neighbor in node.neighbors:
                        if neighbor.node_id not in selected:
                            next_nodes.append(neighbor)
    except Found:
        for node_id, node in selected.items():
            new_neighbors = []
            new_links = []
            for i, neighbor in enumerate(node.neighbors):
                if neighbor.node_id in selected:
                    new_neighbors.append(neighbor)
                    new_links.append(node.links[i])
            node.neighbors = new_neighbors
            node.links = new_links
        return selected

    

def initialize_subset(num_nodes=None):
    """
    Read the topology dumped to a temporary file
    select num_nodes of the nodes to use in the simulation.
    If num_nodes is None, use all of the nodes
    """
    nodes = {}
    with open(TOPOLOGY_DATA_LOC) as fin:
        as_configs = json.load(fin)

    for as_config in as_configs:
        key = as_config["node_id"]
        nodes[key] = Node.from_config(as_config)

    links = {}
    with open(LINK_DATA_LOC) as fin:
        link_configs = json.load(fin)

    for link_config in link_configs:
        links[link_config["link_id"]] = Link.from_config(link_config)

    for node_id, node in nodes.items():
        node.fill_neighbors(nodes)

    set_link_lengths(nodes, links)

    if num_nodes is not None and num_nodes < len(nodes):
        nodes = select_subset(nodes, num_nodes)
    return nodes


def graph_diameter(node_subset):
    """
    Naively computes the diameter of the graph by running
    breadth-first search on every node. 
    """
    max_min_distance = -1
    for i, (cur_node_id, cur_node) in enumerate(node_subset.items()):
        print("\rGetting diameter for " + str(i) + "th node.", end = "")
        next_frontier = [cur_node]
        explored = set([cur_node_id])
        hops = 0
        while len(next_frontier) > 0:
            frontier = next_frontier
            next_frontier = []
            for node in frontier:
                for neighbor in node.neighbors:
                    if neighbor.node_id not in explored:
                        explored.add(neighbor.node_id)
                        next_frontier.append(neighbor)
            hops += 1
        if hops > max_min_distance:
            max_min_distance = hops
    print("\r")
    return hops


def initialize_routing_tables(node_subset, diameter=15):
    """
    node_subset: dictionary of (node_id -> node ...) entries.
    diameter: maximum # hops between any two nodes. We need to
    broadcast this many times. 
    """
    print("Broadcasting...", end="")
    for i in range(diameter):
        print("\rBroadcasting... Round", i, "of", diameter, end="")
        for j, (node_id, node) in enumerate(node_subset.items()):
            node.broadcast_to_neighbors()
    print()


def reset_all_links(node_subset):
    """
    Set utilization of all links to zero
    """
    for node_id, node in node_subset.items():
        node.reset_links()



def get_average_delay_sage(weight_model, node_subset, p_communicate, mean_traffic,
                           sd_traffic, ut_iterations=3, loop_penalty=2):
    """
    Get average delay where policies are aware of the policies are aways
    of the utilization of their outgoing links. Allow ut_iterations
    or stabilization of link utilizations. weight_model is the graphsage model used
    to set the weights

    1. call the weight model to set the initial weights.
    2. for ut_iterations, call the weights model again to account for link utilization
    3. calculate and report the distance inflation factor
    4. the loop penalty determines how bad a route is penalized for not completing
    """
    np.random.seed(1)
    weight_model(node_subset) # sets the weights for all routing tables
    reset_all_links(node_subset)
    node_pairs = [] # list of communicating pairs
    sum_straight_line_dist = 0
    for i, (id1, node1) in enumerate(node_subset.items()):
        for id2, node2 in node_subset.items():
            if id1 == id2:
                continue
            if np.random.random() > p_communicate:
                continue
            traffic = max(np.random.normal(mean_traffic, sd_traffic), 0)
            sum_straight_line_dist += calc_haversine(node1.lat, node1.lon,
                                                     node2.lat, node2.lon)
            node_pairs.append((node1, node2, traffic))
            
    for i in range(ut_iterations):
        nested_terminal_paths = [] # list of list of paths for each pair
        for node1, node2, traffic in node_pairs:
            terminals = node1.find_paths("max-link-logits", node2.node_id, traffic, len(node_subset))
            nested_terminal_paths.append(terminals)
            
    if sum_straight_line_dist == 0:
        sum_straight_line_dist += 1

    sum_expected_distance = 0
    for terminal_paths in nested_terminal_paths:
        expected_distance = 0
        total_prob = 0
        for prob, links, node in terminal_paths:
            outer_sum = 0
            for i, link in enumerate(links):
                inner_sum = 0
                p_link_success = link.bandwidth / max(link.utilization,
                                                      link.bandwidth)
                for j in range(i):
                    inner_sum += links[j].length
                outer_sum += (((1 / p_link_success) - 1) * inner_sum + link.length)
            expected_distance += (outer_sum * prob)
            total_prob += prob
        if total_prob == 0:
            sum_expected_distance += 10000 * sum_straight_line_dist
        else:
            sum_expected_distance += (expected_distance / (total_prob ** loop_penalty))
    return sum_expected_distance / sum_straight_line_dist


def get_average_delay(node_subset, p_communicate, mean_traffic,
                      sd_traffic, method):
    """
    Every node has a p_communicate probability of communicating
    with every other node. If the node is communicating, the amount
    of traffic is determined by a normal distribution with the given
    mean and std. There can't be negative traffic. R_obj is a random
    object with a seed.

    Method is the policy used to decide which node to send traffic to
    can be one of

    Distance increase factor: How much worse were you then straight line distance?
    - "hops": decide based on the number of hops
    """
    np.random.seed(1)
    reset_all_links(node_subset)
    nested_terminal_paths = [] # list of list of paths for each pair
    sum_straight_line_dist = 0
    for i, (id1, node1) in enumerate(node_subset.items()):
        for id2, node2 in node_subset.items():
            if id1 == id2:
                continue
            if np.random.random() > p_communicate:
                continue
            traffic = max(np.random.normal(mean_traffic, sd_traffic), 0)
            sum_straight_line_dist += calc_haversine(node1.lat, node1.lon,
                                                     node2.lat, node2.lon)
            terminals = node1.find_paths(method, id2, traffic, len(node_subset))
            nested_terminal_paths.append(terminals)

    sum_expected_distance = 0
    for terminal_paths in nested_terminal_paths:
        expected_distance = 0
        total_prob = 0
        for prob, links, node in terminal_paths:
            outer_sum = 0
            for i, link in enumerate(links):
                inner_sum = 0
                p_link_success = link.bandwidth / max(link.utilization,
                                                      link.bandwidth)
                for j in range(i):
                    inner_sum += links[j].length
                outer_sum += (((1 / p_link_success) - 1) * inner_sum + link.length)
            expected_distance += (outer_sum * prob)
            total_prob += prob
        if total_prob == 0:
            sum_expected_distance += 10000 * sum_straight_line_dist
        else:
            sum_expected_distance += (expected_distance / (total_prob ** 2))
        sum_expected_distance += expected_distance
    return sum_expected_distance / sum_straight_line_dist


def write_results(results, args, methods, connects, means):
    """
    Save the given results and arguments used to create them 
    """
    timestamp = datetime.datetime.now()
    result_dir = os.path.join(RESULTS_DIR, timestamp.strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, "args.txt"), "w") as fout:
        fout.write("\n".join(args))
    np.savez_compressed(os.path.join(result_dir, "results"), results=results)

    better_name = {
        "hops": "Route by Hops",
        "distance": "Route by Distance",
        "bandwidth": "Route by Average Bandwidth",
        "distance-bandwidth": "Route by Distance/Bandwidth",
        "sage": "Route by GraphSage Weights"
    }

    fig, ax = plt.subplots(nrows=1, ncols=len(methods), squeeze=False, sharex=True, sharey=True)
    fig.suptitle("Relitive Distance Travelled by Routing Policy")
    vmin = np.quantile(results, 0.25)
    vmax = np.quantile(results, 0.75)
    for i, method in enumerate(methods):
        cbar = ax[0, i].imshow(results[i], origin="lower", vmin=vmin, vmax=vmax)
        ax[0, i].set_title(better_name[method])
        ax[0, i].set_ylabel("Mean traffic from an AS (Mbps)")
        ax[0, i].set_xlabel("Network connectivity")
        ax[0, i].set_xticks(np.arange(results[i].shape[1]))
        ax[0, i].set_yticks(np.arange(results[i].shape[0]))
        ax[0, i].set_xticklabels(connects)
        ax[0, i].set_yticklabels(means)
        
    fig.colorbar(cbar, ax=fig.get_axes(), orientation="horizontal")
    fig.set_size_inches(14, 4)
    fig.savefig(os.path.join(result_dir, "tileplot.png"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BGP simulated performance.")
    parser.add_argument("n_trials", type=int, help="Number of different seeds to run.")
    parser.add_argument("n_nodes", type=int, help="Number of nodes to use in topology.")
    parser.add_argument("methods", type=str, help="Policies to use during simulation. Dimit with ;")
    parser.add_argument("connectivities", type=str, help="Connectivities to try during simulation. Delimit with ;")
    parser.add_argument("means", type=str, help="Means to try during simulation delimit with ;")
    parser.add_argument("-m", "--model_loc", type=str, help="Location of graphsage model")
    args = parser.parse_args(sys.argv[1:])

    methods = [m.strip() for m in args.methods.split(";")]
    connectivities = [float(c.strip()) for c in args.connectivities.split(";")]
    means = [float(m.strip()) for m in args.means.split(";")]
    if args.model_loc:
        from graph_sage import WeightModel
        sage_model = WeightModel.load(args.model_loc)
        methods.append("sage")

    results = np.zeros((len(methods), len(means), len(connectivities)))
    for trial in range(args.n_trials):
        np.random.seed(trial)
        graph_subset = initialize_subset(args.n_nodes)
        diameter = graph_diameter(graph_subset)
        initialize_routing_tables(graph_subset)
        for i, method in enumerate(methods):
            for j, mean in enumerate(means):
                for k, p_connect in enumerate(connectivities):
                    print("\r", method, j, "out of", len(means), ";", k,
                          "out of", len(connectivities), end="")
                    if method == "sage":
                        sum_delay = get_average_delay_sage(sage_model, graph_subset, p_connect, mean,
                                                           mean / 4)
                    else:
                        sum_delay = get_average_delay(graph_subset, p_connect, mean, mean / 4, method)
                    
                    results[i][j][k] += sum_delay
        print()
    results /= args.n_trials
    write_results(results, sys.argv[1:], methods, connectivities, means)
    
    


    






