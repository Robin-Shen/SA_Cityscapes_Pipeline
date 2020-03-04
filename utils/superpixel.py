#!/usr/bin/env python
# coding: utf-8

import numpy as np
from skimage import segmentation as sg
from skimage import exposure
import cv2

def get(image):
    """
    apply some superpixel algorithm to segment image
    it returns an array of labels
    """
    h, w = image.shape[:2]
    if min(h, w) > 10:
        # histogram equalization
        print("Generating superpixels with adaptive equalization histogram...")
        image = exposure.equalize_adapthist(image, clip_limit=0.01)
    else:
        print("Generating superpixels...")

    # get superpixel labels
    # superpixels = sg.slic(image, compactness=4, n_segments=h*w//25, sigma=0.5, enforce_connectivity=True)
    # superpixels = sg.quickshift(image, ratio=0.5)
    superpixels = sg.felzenszwalb(image, scale=0.01)
    # retval= cv2.ximgproc.createSuperpixelSEEDS(w, h, 3, w*h//10, num_levels=20, prior=2, histogram_bins=10)
    # retval.iterate(image.astype(np.uint8), 50)
    # superpixels = retval.getLabels()
    return superpixels.astype(int)

def split(superpixels, annotation):
    """
    split superpixels with different label
    """
    superpixels = sg.join_segmentations(superpixels, annotation[:,:,0] // 255 * annotation[:,:,1])
    superpixels = sg.join_segmentations(superpixels, annotation[:,:,0] // 255 * annotation[:,:,2])
    superpixels = enforce_connectivity(superpixels)
    superpixels, _, _ = sg.relabel_sequential(superpixels)
    return superpixels + 1

def enforce_connectivity(segments):
    """
    mark unconnected segement as different parts
    """
    multiplier = np.max(segments)
    for i in np.unique(segments):
        cnt, comps = cv2.connectedComponents((segments == i).astype(np.uint8))
        if cnt > 2:
            segments += comps * multiplier
    return segments
