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
    data_generator = data_loader.load_cityscapes(path, "")

    cnt = 0
    ssegs = []
    preds = []

    for filename, image, sseg, inst, scribbles in data_generator:
        cnt += 1
        # get annotation
        anno_path =  "./experiments_eccv/" + args.exp_fdr +  "/" + filename + "_gtFine_labelIds.png"
        if not os.path.isfile(anno_path):
            #print("Annotation does not exists, skip {}".format(filename))
            cnt -= 1
            continue

        anno = np.array(Image.open(anno_path))
        pred = np.zeros_like(anno, dtype=np.int8)
        for trainid in np.unique(anno):
            id = data.id2train[trainid]
            pred += id * (anno == trainid)

        # store for score
        preds += list(pred%21)
        ssegs += list(sseg)

        # visualize
        # mask_show(image, mask, inst_pred, name="image")
        # cv2.destroyAllWindows()

        # terminate with iteration limit

        #if cnt > 1:
        #     break

    # calculate MIoU
    print("Score for {} scribbles:".format(cnt))
    print(metrics.scores(ssegs, preds, 19))
