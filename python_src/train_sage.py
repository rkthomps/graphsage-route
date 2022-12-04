
"""
The purpose of this module is to train the final GraphSage models

Following the work in the graphsage paper, we will use k = 2.
We will use embedding sizes of 32 
"""

import sys, os
import numpy as np
import datetime
import argparse

from graph_sage import WeightModel
from simulate_bgp import (initialize_subset,
                          graph_diameter,
                          initialize_routing_tables)

MODEL_LOC = os.path.join("models")

def initialize_graph(num_nodes):
    np.random.seed(np.random.randint(10000, 20000))
    graph_subset = initialize_subset(num_nodes)
    diameter = graph_diameter(graph_subset)
    initialize_routing_tables(graph_subset, diameter)
    return graph_subset


def save_model(model):
    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(os.path.join(MODEL_LOC, date_str))


def train_sage():
    model = WeightModel(K=2, k_width=32, agg_width=32)
    train_steps = 20
    training_nodes = 20
    variations_per_epoch = 8
    average_results = []
    for epoch in range(train_steps):
        print("Beginning epoch:", epoch)
        nodes = initialize_graph(training_nodes)
        avg_result = model.train_step(nodes, variations_per_epoch)
        average_results.append(avg_result)
        print("Epoch", epoch, "average result", avg_result)
    save_model(model)


def train_sage_robust():
    model = WeightModel(K=2, k_width=32, agg_width=32)
    train_steps = 20
    training_nodes = 20
    variations_per_epoch = 32
    average_results = []
    for epoch in range(train_steps):
        print("Beginning epoch:", epoch)
        avg_result = model.train_step_robust(training_nodes, variations_per_epoch)
        average_results.append(avg_result)
        print("Epoch", epoch, "average result", avg_result)
    save_model(model)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robust", help="whether to use a new topolology at each epoch",
                        action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.robust:
        train_sage_robust()
    else:
        train_sage()



