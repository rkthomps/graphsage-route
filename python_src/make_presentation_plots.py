



import matplotlib.pyplot as plt
from matplotlib import rc
import sys, os
import numpy as np
import pdb
import argparse

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def make_plot(data_dir, num_show):

    results = np.load(os.path.join(data_dir, "results.npz"))["results"]

    with open(os.path.join(data_dir, "args.txt")) as fin:
        args = fin.read().split("\n")
    start_index = 0
    if args[0] == "-m":
        start_index += 2
    num_trials = int(args[start_index])
    num_nodes =  int(args[start_index + 1])
    methods = [a.strip() for a in args[start_index + 2].split(";")]
    connects = [float(c) for c in args[start_index + 3].split(";")]
    means = [float(m) for m in args[start_index + 4].split(";")]
    if start_index > 0:
        methods.append("sage")
    methods = methods[:num_show]


    better_name = {
        "hops": "Route by Hops",
        "distance": "Route by Distance",
        "bandwidth": "Route by Average Bandwidth",
        "distance-bandwidth": "Route by Distance/Bandwidth",
        "sage": "Route by GraphSage Weights"
    }

    fig, ax = plt.subplots(nrows=1, ncols=len(methods), squeeze=False, sharex=True, sharey=True)
    fig.suptitle("Relitive Distance Travelled by Routing Policy")
    vmin = np.quantile(results[:num_show], 0.25)
    vmax = np.quantile(results[:num_show], 0.75)
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
    fig.set_size_inches(3.8 * num_show, 4)
    fig.savefig(os.path.join(data_dir, str(num_show) + "tileplot.png"))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Location of the data directory.")
    parser.add_argument("num_show", type=int, help="Number of plots to show")
    args = parser.parse_args(sys.argv[1:])
    make_plot(args.data_dir, args.num_show)



