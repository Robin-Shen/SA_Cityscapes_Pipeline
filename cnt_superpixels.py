import os
import sys
import networkx as nx
import argparse

from PATH import *
from utils import *

if __name__ == "__main__":

    path = DATA_PATH

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', type=str, default=path+"/graphs/")
    args = parser.parse_args()

    # biuld data loader
    cnt = 0
    nums = 0
    data_generator = data_loader.load_cityscapes(path, "scribbles")
    for filename, image, sseg, inst, scribbles in data_generator:
        if os.path.isfile(args.graph_dir + "/" + filename + ".gpickle"):
            graph = nx.read_gpickle(args.graph_dir + "/" + filename + ".gpickle")
            nums += len(graph)
            cnt += 1
        else:
            print("Skipping image {} because it does not have graph...".format(filename))

print("Average number of superpixels of {} graphs is {}".format(cnt, int(nums/cnt)))
