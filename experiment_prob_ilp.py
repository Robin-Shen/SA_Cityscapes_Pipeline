#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from collections import namedtuple
import cv2
from PIL import Image
import networkx as nx
import numpy as np
from skimage import segmentation as sg
import argparse

from PATH import *
from utils import *
import solver_prob as solver

import torch
import torch.nn as nn
from torch.nn import functional as F
import models
from matplotlib import pyplot as plt


# ignore warnings
import warnings
warnings.filterwarnings("ignore")


class FeatureOut(nn.Module):
    def __init__(self, model, extracted_layer):
        super(FeatureOut, self).__init__()
        self.features = nn.Sequential(
            *list(model.module.base.children())[0],
            *list(model.module.base.children())[1]
        )[:extracted_layer]
    def forward(self, x):
        x = self.features(x)
        return x

def set_model(checkpoint):
    """
    set the model
    """
    model = models.DeepLabV2_ResNet101_MSC(21)
    state_dict = torch.load(checkpoint)
    print("    Init:", checkpoint)
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = torch.nn.DataParallel(model)
    model.cuda(0)
    model.eval()
    return model


def mask_show(image, mask, groups, name="image"):
    """
    show image with mask
    """
    img = cv2.addWeighted(image, 0.4, mask, 0.6, 0)
    img = sg.mark_boundaries(img, groups, color=(1,1,1))
    cv2.imshow(name, img)
    cv2.waitKey(0)


def mask_to_label(mask):
    """
    convert mask image into label martix
    """
    # get the image size
    h, w, _ = mask.shape

    # build a color to label map
    color_to_idx = {}
    for label in class_info:
        color_to_idx[class_info[label].color] = class_info[label].id

    # generate label matrix
    label = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            b, g, r = mask[y, x]
            color = (r, g, b)
            label[y, x] = color_to_idx[color]

    return label


def label_to_mask(labels):
    """
    convert label martix into mask image
    """
    # get the image size
    h, w = labels.shape

    # build a color to label map
    idx_to_color = {}
    for label in class_info:
        idx_to_color[class_info[label].id] = class_info[label].color

    # generate label matrix
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            id = labels[y, x]
            r, g, b = idx_to_color[id]
            mask[y, x] = np.array([b, g, r])

    return mask


def transform(image):

    # BGR to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize
    img = cv2.resize(img, (513, 513), interpolation=cv2.INTER_LINEAR).astype(int)
    mean = np.array([104, 117, 123], dtype=int)
    img -= mean
    #std = np.array([0.229, 0.224, 0.225])
    #img /= std
    # to tensor
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    img = img.view(1, 3, 513, 513)
    img.cuda(0)

    return img


def get_map(image, model, softmax=False):
    """
    get feature map and probability map from cnn
    """
    # transform image
    img = transform(image).cuda(0)

    # run inference
    with torch.no_grad():
        feat_map = model(img)

    feat_map = F.interpolate(feat_map, size=(image.shape[0], image.shape[1]), mode='bilinear')
    if softmax:
        feat_map = F.softmax(feat_map, dim=1)
    feat_map = feat_map.data.cpu().numpy()[0]

    return feat_map

def show_feat(feat_map):
    """
    visualize each feature map
    """
    for i in range(feat_map.shape[0]):
        plt.imshow(feat_map[i])
        plt.show()


if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=float, default=3)
    parser.add_argument('--timelimit', type=int, default=20)
    args = parser.parse_args()



    intersections = np.zeros((21))
    unions = np.zeros((21))

    path = DATA_PATH
    data_generator = data_loader.load_cityscapes(path, "scribbles")
    prob_path = PROB_PATH

    # create folder
    if not os.path.isdir("./experiments_eccv"):
        os.mkdir("./experiments_eccv")

    # experiment folder for ilp,  + "_mipemphasis = hiddenfeas" + "_modi/"
    saved_folder = "./experiments_eccv/prob_ilp_param=" + str(args.param) + "_t=" + str(args.timelimit) + "/"
    print(saved_folder)

    if not os.path.isdir(saved_folder):
        os.mkdir(saved_folder)

    cnt = 0
    ssegs = []
    preds = []

    no_solution_list = []
    opt_list = []

    solution_cnt = 0

    algo_time = 0
    number_nodes = 0

    tick = time.time()
    opt_cnt = 0
    number_opt_nodes = 0
    infea = []

    for filename, image, sseg, inst, scribbles in data_generator:
        cnt += 1
        height, width = image.shape[:2]

        # skip existed gt
        if os.path.isfile(saved_folder + filename + "_gtFine_labelIds.png"):
            print("Annotation exists, skip {}".format(filename))
            cnt -= 1
            continue

        if scribbles is not None:
            print("\n##########{}: Generating ground truth approach for image {}...".format(cnt, filename))
            # BGR to RGB
            scribbles = cv2.cvtColor(scribbles, cv2.COLOR_BGR2RGB)
            scribbles[:,:,1] = np.where(scribbles[:,:,1]==255, 128, scribbles[:,:,1])
            # remove instance id
            scribbles[:,:,2] = 0
        else:
            # skip image which does not have annotation
            print("Skipping image {} because it does not have annotation...".format(filename))
            cnt -= 1
            continue

        

        # generate superpixels
        # superpixels = superpixel.get(image)
        if not os.path.isfile(path + "/graphs/" + filename + ".gpickle"):
            print("Skipping image {} because it does not have graph...".format(filename))
            cnt -= 1
            continue
        graph = nx.read_gpickle(path + "/graphs/" + filename + ".gpickle")
        superpixels = graph.get_superpixels_map()
        # split by annotation
        superpixels = superpixel.split(superpixels, scribbles)

        # build graph
        graph = to_graph.to_superpixel_graph(image, scribbles, superpixels)

        # get prob map
        prob = np.load(prob_path + filename + "_leftImg8bit.npy")[0].astype("float")
        #show_feat(prob)
        graph.load_feat_map(prob, attr="feat")

        # lambd=0.1
        # psi=0.0
        # phi=0.001
        # heuristic_graph = solver.heuristic.solve(graph.copy(), lambd, psi, phi, attr="feat")
        # convert into mask
        # mask, pred = to_image.graph_to_image(heuristic_graph, height, width, scribbles)
        # show the mask
        # mask_show(image, mask, pred, name="heur")
        # cv2.destroyAllWindows()

        # load heuristic result directly
        pred_id = np.array(Image.open("/m/Camera_team/01_team_members/ruobing/experiments_eccv/prob_heur/" + filename + "_gtFine_labelIds.png"))
        pred = np.zeros_like(pred_id, dtype=np.int8)
        for trainid in np.unique(pred_id):
            id = data.id2train[trainid]
            pred += id * (pred_id == trainid)

        # mark ignore
        ignore = (pred_id == 0)
        scribbles[:,:,0] = scribbles[:,:,0] * (1 - ignore) + 255 * ignore
        scribbles[:,:,1] = scribbles[:,:,1] * (1 - ignore) + 128 * ignore

        ilp_graph = graph.copy()
        # drop instance id
        for i in ilp_graph.nodes:
            label = ilp_graph.nodes[i]["label"]
            if label:
                ilp_graph.nodes[i]["label"] = label.split("_")[0]
        # update heigh and weight
        ilp_graph.height, ilp_graph.width = height, width
        # merge nodes with same label
        ilp_graph.add_pseudo_edge()
        # get superpixels map
        superpixels = ilp_graph.get_superpixels_map() * (pred != 255)
        # integer linear programming
        ilp = solver.ilp.build_model(ilp_graph, args.param)
        solver.ilp.warm_start(ilp, pred%21, superpixels)

        # print # of nodes per image and accumulate them for computing the average
        print("#### Number of nodes of ILP: {} \n" .format(ilp_graph.number_of_nodes()))
        number_nodes += ilp_graph.number_of_nodes()

        # set time limit
        timelimit = args.timelimit
        ilp.parameters.timelimit.set(timelimit)

        # change mip emphasis ---- not done
        #ilp.parameters.emphasis.mip.set(4)

        tick1 = time.time()


        # solve
        ilp.solve()

        algo_time += time.time() - tick1
        #print("###Status of ilp = {} ###".format(ilp.solution.get_status()))
        #print("###Status of ilp = {} ###".format(ilp.solution.get_status()))
        #if ilp.solution.get_status() == 101 or 107 or 109 or 127 or 105 or 111 or 113:
        if ilp.solution.get_status() != 103 and ilp.solution.get_status() != 108:
            print("##Status of ilp = {} ##".format(ilp.solution.get_status()))
            mask, pred = to_image.ilp_to_image(ilp_graph, ilp, height, width, scribbles)
            cv2.imwrite(saved_folder + filename + "_gtFine_color.png", mask)
            solution_cnt += 1
            if ilp.solution.get_status() == 101:
                opt_cnt += 1
                opt_list.append(filename)
                number_opt_nodes += ilp_graph.number_of_nodes()
        else:
            print("###############Status of ilp = {} ###############".format(ilp.solution.get_status()))
            no_solution_list.append(filename)
            #mask, pred = to_image.ilp_to_image(ilp_graph, ilp, height, width, scribbles)



        # show the mask
        # mask_show(image, mask, pred, name="ilp")
        # cv2.destroyAllWindows()

        # get formatted sseg and inst
        sseg_pred, inst_pred = to_image.format(pred)
        # save annotation
        Image.fromarray(sseg_pred).save(saved_folder  + filename + "_gtFine_labelIds.png")
        # Image.fromarray(inst_pred).save("./experiments_eccv/prob_ilp_arti/" + filename + "_gtFine_instanceIds.png")
        #cv2.imwrite(saved_folder + filename + "_gtFine_color.png", mask)

        #Image.fromarray(sseg_pred).save("./experiments_eccv/prob_ilp/"  + filename + "_gtFine_labelIds.png")
        # Image.fromarray(inst_pred).save("./experiments_eccv/prob_ilp/" + filename + "_gtFine_instanceIds.png")

        # store for score
        preds += list(pred%21)
        ssegs += list(sseg)
        #print("ILPs that have no solution: {}".format(no_solution_list))
        #print(no_solution_list)
        # visualize
        #mask_show(image, mask, inst_pred, name="image")
        #cv2.destroyAllWindows()

        # terminate with iteration limit
        #if cnt > 1:
        #     break

    # show paramters
    #print(lambd, psi, phi)

    print("Real used average time: {}".format((time.time() - tick) / cnt))
    print("Average algo time: {}".format(algo_time / cnt))
    print("Average number of nodes of ILP: {} \n".format(number_nodes / cnt))
    print("Average number of optimal ILP nodes: {} \n".format(number_opt_nodesnodes / opt_cnt))
    print("Number of feasible ILPs: {} \n".format(solution_cnt))
    print("Number of optimal ILPs: {} \n".format(opt_cnt))

    print(saved_folder)

    # calculate MIoU
    print("Score for {} scribbles:".format(cnt)) 
    print(metrics.scores(ssegs, preds, 19))

    print("ILPs that have no solution: {}".format(no_solution_list))
    print("ILPs that are optimal: {}".format(opt_list))
