
"""
Quick module for plotting the results 
"""

import pdb
import sys, os
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
import numpy as np


rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_txt(input_file, smoothing):
    with open(input_file, "r") as fin:
        contents = fin.read()
    lines = contents.split("\n")
    lines = [l for l in lines if l.startswith("Epoch")]
    results = np.array([float(l.split()[4]) for l in lines])
    window = np.array([1] * smoothing + [0] * (len(results) - smoothing))
    window_matrix = np.array([np.roll(window, i) for i in range(len(results)-smoothing)])
    smooth_results = window_matrix @ results

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(smooth_results)) + 1, smooth_results)
    ax.set_title("Average Reward Per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Reward")
    #fig.set_size_inches(1, 1)
    fig.savefig(os.path.join(os.path.dirname(input_file),
                             os.path.basename(input_file) + "plot.png"))
    



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Path to stdout of training")
    parser.add_argument("smoothing", type=int, help="Smoothing factor to use when plotting")
    args = parser.parse_args(sys.argv[1:])
    plot_txt(args.input_file, args.smoothing)

    




