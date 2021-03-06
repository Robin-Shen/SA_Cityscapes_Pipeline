#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import glob

from PATH import *
from utils import *

if __name__ == "__main__":

    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default="./heur_prob_del.txt")
    parser.add_argument('--delete_fdr', type=str, default="/m/Camera_team/01_team_members/ruobing/experiments_eccv/prob_ilp_param=3.0_t=20")
    args = parser.parse_args()

    with open (args.list, "r") as f:
        lst = eval(f.readlines()[0])

    for name in lst:
        search = args.delete_fdr + "/" + name + "*"
        files = glob.glob(search)
        for file in files:
            print("Removing {}...".format(file))
            os.remove(file)
