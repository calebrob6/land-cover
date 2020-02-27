#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Le Hou <lehou0312@gmail.com>
"""Script for running computing accuracy given prediction and lc labels
"""
# Stdlib imports
import sys
import os
import datetime
import argparse

# Library imports
import numpy as np
import pandas as pd

import shapely
import shapely.geometry
import rasterio
import rasterio.mask
from shapely.geometry import mapping
from collections import defaultdict

# Setup
from helpers import get_logger
from utils import handle_labels

logger = get_logger(__name__)

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------


def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument(
        "--output",
        action="store",
        dest="output",
        type=str,
        required=True,
        help="Output directory to store predictions",
    )
    parser.add_argument(
        "--input",
        action="store",
        dest="input_fn",
        type=str,
        required=True,
        help="Path to filename that lists tiles",
    )
    parser.add_argument(
        "--classes", type=int, default=5, help="Number of target classes",
    )

    return parser.parse_args(arg_list)


def bounds_intersection(bound0, bound1):
    left0, bottom0, right0, top0 = bound0
    left1, bottom1, right1, top1 = bound1
    left, bottom, right, top = (
        max([left0, left1]),
        max([bottom0, bottom1]),
        min([right0, right1]),
        min([top0, top1]),
    )
    return (left, bottom, right, top)


def get_confusion_matrix(lc_vec, pred_vec, classes):
    cnf = np.zeros((classes - 1, classes - 1), dtype=int)
    for j in range(0, classes - 1):
        for k in range(0, classes - 1):
            cnf[j, k] = ((lc_vec == j + 1) & (pred_vec == k + 1)).sum()
    return cnf


def compute_accuracy(
    pred_dir: str,
    input_fn: str,
    classes: int = 5,
    hr_label_key: str = "data/cheaseapeake_to_hr_labels.txt",
    lr_label_key: str = "data/nlcd_to_lr_labels.txt",
):
    data_dir = os.path.dirname(input_fn)

    logger.info(
        "Starting %s at %s"
        % ("Accuracy computing script", str(datetime.datetime.now()))
    )

    try:
        df = pd.read_csv(input_fn)
        fns = df[["naip-new_fn", "lc_fn", "nlcd_fn"]].values
    except Exception as e:
        logger.error("Could not load the input file")
        logger.error(e)
        return

    cm = np.zeros((classes - 1, classes - 1,), dtype=np.float32)
    cm_dev = np.zeros((classes - 1, classes - 1,), dtype=np.float32)
    acc_sum = 1e-6
    acc_num = 1e-6
    for i in range(len(fns)):
        naip_fn = os.path.join(data_dir, fns[i][0])
        lc_fn = os.path.join(data_dir, fns[i][1])
        nlcd_fn = os.path.join(data_dir, fns[i][2])

        pred_fn = os.path.join(pred_dir, os.path.basename(naip_fn)[:-4] + "_class.tif")
        pred_f = rasterio.open(pred_fn, "r")
        pred = pred_f.read()
        pred_f.close()

        lc_f = rasterio.open(lc_fn, "r")
        lc = lc_f.read()
        lc_f.close()

        nlcd_f = rasterio.open(nlcd_fn, "r")
        nlcd = nlcd_f.read()
        nlcd_f.close()

        nlcd = handle_labels(nlcd.squeeze().astype(int), lr_label_key)

        lc = handle_labels(np.squeeze(lc).astype(int), hr_label_key)

        pred = handle_labels(np.squeeze(pred).astype(int), hr_label_key)

        roi = (lc > 0) & (pred > 0)
        roi_dev = (lc > 0) & (pred > 0) & (nlcd >= 21) & (nlcd <= 24)

        if np.sum(roi) > 0:
            if np.sum(roi_dev) > 0:
                cm_dev += get_confusion_matrix(
                    lc[roi_dev > 0].flatten(), pred[roi_dev > 0].flatten(), classes
                )
            cm += get_confusion_matrix(
                lc[roi > 0].flatten(), pred[roi > 0].flatten(), classes
            )
            accuracy = np.sum(lc[roi > 0] == pred[roi > 0]) / np.sum(roi)
            acc_sum += np.sum(lc[roi > 0] == pred[roi > 0])
            acc_num += np.sum(roi)
        else:
            accuracy = -1

        logger.info(
            "Accuracy %f %s\t%d/%d %f"
            % (accuracy, lc_fn, i + 1, len(fns), acc_sum / acc_num)
        )
    return acc_sum / acc_num, cm, cm_dev


def main():
    program_name = "Accuracy computing script"
    args = do_args(sys.argv[1:], program_name)
    acc, cm, cm_dev = compute_accuracy(args.output, args.input_fn, args.classes)
    print("-----------------------------")
    print(acc)
    print(cm)
    print(cm_dev)


if __name__ == "__main__":
    main()
