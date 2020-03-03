#!/usr/bin/env python
# coding: utf-8

import os
import sys

import argparse
import numpy as np
from PIL import Image

from PATH import *
from utils import *

# ignore warnings
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_fdr', type=str, default="prob_heur")
    args = parser.parse_args()

    intersections = np.zeros((21))
    unions = np.zeros((21))

    path = DATA_PATH
    prob_path = PROB_PATH
    data_generator = data_loader.load_cityscapes(path, "scribbles")

    cnt = 0
    ssegs = []
    preds = []

    for filename, image, sseg, inst, scribbles in data_generator:

        if scribbles is None:
            continue

        # get prediction
        prob = np.load(prob_path + filename + "_leftImg8bit.npy")[0].astype("float")
        pred = np.argmax(prob, axis=0)

        # store for score
        preds += list(pred%21)
        ssegs += list(sseg)

        # visualize
        # mask_show(image, mask, inst_pred, name="image")
        # cv2.destroyAllWindows()

        # terminate with iteration limit
        cnt += 1
        if cnt > 1:
             break

    # calculate MIoU
    print("Score for origin scribbles:")
    print(metrics.scores(ssegs, preds, 19))
