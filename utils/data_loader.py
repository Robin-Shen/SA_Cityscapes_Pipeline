#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
from PIL import Image
import torchvision.datasets
from utils.data import labels, class_info


class Dataset(torchvision.datasets.Cityscapes):
    """
    custom dataset that includes image name
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(Dataset, self).__getitem__(index)
        # the image file path
        path = self.images[index]
        # the image name
        name = path.split("\\")[-1].split("/")[-1][:-16]
        # make a new tuple that includes original and the path
        tuple_with_name = (original_tuple + (name,))
        return tuple_with_name


def load_cityscapes(path, fdr):
    """
    load Cityscapes val set
    """
    dataset = Dataset(path, split='val', mode="fine", target_type=["semantic", "instance"])

    from PATH import SCRI_PATH as spath

    for image, (sseg, inst), name in dataset:
        image = np.array(image)
        sseg = gt_covert(sseg)
        inst = np.array(inst)
        if os.path.exists(spath + "/" + fdr + "/" + name + "_scri.png"):
            scribbles = np.array(Image.open(spath + "/" + fdr + "/" + name + "_scri.png"))
        else:
            scribbles = None
        # scribbles = scribble_convert(scribbles)
        yield name, image, sseg, inst, scribbles


def gt_covert(gt):
    """
    convert id of groud truth to train id
    """
    gt = np.array(gt)
    new_gt = np.zeros_like(gt)
    for label in labels:
        if label.id != 0 and label.trainId == 255:
            new_gt = new_gt + (gt == label.id) * 128
        elif label.id != -1:
            new_gt = new_gt + (gt == label.id) * label.trainId
        else:
            new_gt = new_gt + (gt == label.id) * 255
    return new_gt


def scribble_convert(scribbles):
    """
    convert scribble as train id
    """
    new_scribbles = np.zeros((*scribbles.shape, 3), dtype=np.uint8)
    # class id
    labels = gt_covert(scribbles * (scribbles < 1000) + (scribbles // 1000) * (scribbles >= 1000))
    new_scribbles[:, :, 1] = labels
    # annotated
    annotated = (labels != 255) * 255
    new_scribbles[:, :, 0] = annotated
    # instance id
    instance = (scribbles % 1000) * (scribbles >= 1000)
    new_scribbles[:, :, 2] = instance
    return new_scribbles
