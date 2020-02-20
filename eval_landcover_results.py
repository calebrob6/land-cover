import os
import argparse

import numpy as np

from landcover.helpers import get_logger

logger = get_logger(__name__)

TEST_SPLITS = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]
EXP_BASE = "~/land-cover/results/results_sr_epochs_100_0/"


def do_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-splits",
        nargs="+",
        default=TEST_SPLITS,
        help="Name of datasets where tests were ran.",
    )
    parser.add_argument(
        "--exp-base",
        default=EXP_BASE,
        type=str,
        help="Base directory where experiments were saved.",
    )
    return parser.parse_args()


def eval_landcover_results(acc_file):
    with open(acc_file, "r") as src:
        lines = src.readlines()
        # All
        logger.info("Accuracy and jaccard of all pixels")
        all_acc, all_jac = accuracy_jaccard(lines)
        # NLCD
        logger.info("Accuracy and jaccard of pixels with developed NLCD classes")
        dev_acc, dev_jac = accuracy_jaccard(lines, s=-4, f=None)
    return all_acc, all_jac, dev_acc, dev_jac


def accuracy_jaccard(lines, s=-8, f=-4):
    lines_8 = lines[s:f]
    arr = process_line(lines_8)
    j = 0
    all_n = np.sum(arr)
    diag = np.sum(np.diagonal(arr))
    acc = diag / all_n
    for c in range(len(lines_8)):
        j += arr[c, c] / np.sum(arr[c, :] + arr[:, c])
    jaccard = j / len(lines_8)
    logger.info("Accuracy: %.6f, Jaccard: %.6f" % (acc, jaccard))
    return acc, jaccard


def process_line(lines):
    str_list = "".join(lines)
    str_list = str_list.replace(" ", ",")
    str_list = str_list.replace("\n", ",")
    return np.squeeze(np.array(eval(str_list)))


def eval_all_landcover_results(test_splits=TEST_SPLITS, exp_base=EXP_BASE, sr_hr="sr"):
    for split in test_splits:
        acc_file_path = os.path.join(
            exp_base,
            "train-hr_%s_train-%s_%s/" % (split, sr_hr, split),
            "test-output_%s/log_acc_%s.txt" % (split, split),
        )
        try:
            eval_landcover_results(acc_file_path)
        except FileNotFoundError:
            logger.debug("Could not find %s - continuing with next file", acc_file_path)


def main():
    args = do_args()
    print(args.test_splits)
    eval_all_landcover_results(args.test_splits, args.exp_base)


if __name__ == "__main__":
    main()
