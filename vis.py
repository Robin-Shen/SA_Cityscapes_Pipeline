#!/usr/bin/env python
# coding: utf-8

import os
import sys
import cv2
from PIL import Image
import networkx as nx
import numpy as np
from skimage import segmentation as sg
import argparse

from PATH import *
from utils import *

def vis_superpixels(image, groups):
    """
    get image with superpixel
    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = sg.mark_boundaries(img, groups, color=(1,1,1))
    img = (img * 255).astype(np.uint8)

    return img

def vis_scribbles(image, mask, annotated):
    """
    get image with scribbles
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated = np.expand_dims(annotated == 255, axis=2)
    img = image * (1 - annotated) + mask * annotated

    return img.astype(np.uint8)


if __name__ == "__main__":

    path = DATA_PATH

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default="scribbles")
    parser.add_argument('--graph_dir', type=str, default=path+"/graphs/")
    args = parser.parse_args()

    # guarantee valid args
    assert args.type in ["scribbles", "arti_scribbles", "superpixels"], "Type should be scribbles arti_scribbles or superpixels"

    # output folder
    output = "./experiments_eccv/vis/" + args.type

    # create folder
    if not os.path.isdir("./experiments_eccv/vis/"):
        os.mkdir("./experiments_eccv/vis")
    if not os.path.isdir("./experiments_eccv/vis/" + args.type):
        os.mkdir("./experiments_eccv/vis/" + args.type)

    # biuld data loader
    cnt = 0
    data_generator = data_loader.load_cityscapes(path, args.type)
    for filename, image, sseg, inst, scribbles in data_generator:
        cnt += 1

        # superpixel
        if args.type == "superpixels":
            print("{}: Visualizing superpixels for image {}...".format(cnt, filename))
            # get superpixels
            graph = nx.read_gpickle(args.graph_dir + "/" + filename + ".gpickle")
            superpixels = graph.get_superpixels_map()
            # visualize
            img = vis_superpixels(image, superpixels)

        if args.type == "arti_scribbles":
            if scribbles is not None:
                print("{}: Visualizing artificial scribbles for image {}...".format(cnt, filename))
                # reformat scribbles
                scribbles = data_loader.scribble_convert(scribbles)
                # get mask
                mask, annotated = to_image.get_mask(scribbles)
                # visualize
                img = vis_scribbles(image, mask, annotated)
            else:
                cnt -= 1
                print("{}: Skipping image {} because it does not have annotation...".format(cnt, filename))
                continue

        if args.type == "scribbles":
            if scribbles is not None:
                print("{}: Visualizing scribbles for image {}...".format(cnt, filename))
                # reformat scribbles
                # BGR to RGB
                scribbles = cv2.cvtColor(scribbles, cv2.COLOR_BGR2RGB)
                scribbles[:,:,1] = np.where(scribbles[:,:,1]==255, 128, scribbles[:,:,1])
                # get mask
                mask, annotated = to_image.get_mask(scribbles, erode=True)
                # visualize
                img = vis_scribbles(image, mask, annotated)
            else:
                cnt -= 1
                print("{}: Skipping image {} because it does not have annotation...".format(cnt, filename))
                continue

        # show image
        #cv2.imshow("vis", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # write image
        cv2.imwrite(output + "/" + filename + ".png", img)
