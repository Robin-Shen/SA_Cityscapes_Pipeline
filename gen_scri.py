#!/usr/bin/env python
# coding: utf-8

import os
import glob

import numpy as np
from PIL import Image
import cv2

from PATH import *
from utils import data

# get gts
gt_path = DATA_PATH + "/gtFine/*a*/*/*_instanceIds.png"
files = glob.glob(gt_path)
files.sort()

for file in files:

    # get gt
    gt = np.array(Image.open(file))

    # get scribb;es
    name = file.split("\\")[-1]
    scri_path = DATA_PATH + "/arti_scribbles/" + name.replace("gtFine_instanceIds", "scri")
    if not os.path.isfile(scri_path):
        continue
    scri = np.array(Image.open(scri_path))
    print(np.unique(scri))

    # get ignore region
    ignore = np.zeros_like(scri)
    print(data.ignoreId)
    for id in data.ignoreId:
        if id != 3:
            ignore += (gt == id)
    ignore = (ignore * 255).astype(np.uint8)
    ignore = cv2.erode(ignore, None, iterations=4) - cv2.erode(ignore, None, iterations=6)
    ignore = ignore.astype(np.uint16)
    print(np.unique(ignore))

    # add ignore
    #scri = (scri + ignore).astype(np.uint16)

    print(scri.dtype)
    print(np.unique(scri))

    from matplotlib import pyplot as plt
    plt.imshow(scri == 0)
    plt.show()
    break
