import os
import time
from collections import defaultdict

import numpy as np

import keras


NLCD_CLASSES = [
    0,
    11,
    12,
    21,
    22,
    23,
    24,
    31,
    41,
    42,
    43,
    51,
    52,
    71,
    72,
    73,
    74,
    81,
    82,
    90,
    95,
    255,
]
NLCD_CLASSES_TO_IDX_MAP = defaultdict(
    lambda: 0, {cl: i for i, cl in enumerate(NLCD_CLASSES)}
)


def nlcd_classes_to_idx(nlcd):
    return np.vectorize(NLCD_CLASSES_TO_IDX_MAP.__getitem__)(np.squeeze(nlcd)).astype(
        np.uint8
    )


def humansize(nbytes):
    suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ("%.2f" % nbytes).rstrip("0").rstrip(".")
    return "%s %s" % (f, suffixes[i])


def schedule_decay(epoch, lr, decay=0.001):
    if epoch >= 10:
        lr = lr * 1 / (1 + decay * epoch)
    return lr


def schedule_stepped(epoch, lr, step_size=10):
    if epoch > 0:
        if epoch % step_size == 0:
            return lr / 10.0
        else:
            return lr
    else:
        return lr


def load_nlcd_stats():
    nlcd_means = np.concatenate(
        [np.zeros((22, 1)), np.loadtxt("data/nlcd_mu.txt")], axis=1
    )
    nlcd_means[nlcd_means == 0] = 0.000001
    nlcd_means[:, 0] = 0
    nlcd_means[2:, 1] -= 0
    nlcd_means[3:7, 4] += 0.25
    nlcd_means = nlcd_means / np.maximum(0, nlcd_means).sum(axis=1, keepdims=True)
    nlcd_means[0, :] = 0
    nlcd_means[-1, :] = 0

    nlcd_vars = np.concatenate(
        [np.zeros((22, 1)), np.loadtxt("data/nlcd_sigma.txt")], axis=1
    )
    nlcd_vars[nlcd_vars < 0.0001] = 0.0001
    nlcd_class_weights = np.ones((22,))

    # Taken from the training script
    nlcd_class_weights = np.array(
        [
            0.0,
            1.0,
            0.5,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.5,
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.5,
            1.0,
            1.0,
            0.0,
        ]
    )

    return nlcd_class_weights, nlcd_means, nlcd_vars


def find_key_by_str(keys, needle):
    for key in keys:
        if needle in key:
            return key
    raise ValueError("%s not found in keys" % (needle))


class LandcoverResults(keras.callbacks.Callback):
    def __init__(self, log_dir=None, verbose=False):

        self.mb_log_keys = None
        self.epoch_log_keys = None

        self.verbose = verbose
        self.log_dir = log_dir

        self.batch_num = 0
        self.epoch_num = 0

        if self.log_dir is not None:
            self.train_mb_fn = os.path.join(log_dir, "minibatch_history.txt")
            self.train_epoch_fn = os.path.join(log_dir, "epoch_history.txt")

    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()

    def on_batch_begin(self, batch, logs={}):
        self.mb_start_time = float(time.time())

    def on_batch_end(self, batch, logs={}):
        t = time.time() - self.mb_start_time

        if self.mb_log_keys is None and self.log_dir is not None:
            self.mb_log_keys = [
                key for key in list(logs.keys()) if key != "batch" and key != "size"
            ]
            f = open(self.train_mb_fn, "w")
            f.write("Batch Number,Time Elapsed")
            for key in self.mb_log_keys:
                f.write(",%s" % (key))
            f.write("\n")
            f.close()

        if self.log_dir is not None:
            f = open(self.train_mb_fn, "a")
            f.write("%d,%f" % (self.batch_num, t))
            for key in self.mb_log_keys:
                f.write(",%f" % (logs[key]))
            f.write("\n")
            f.close()

        self.batch_num += 1

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = float(time.time())

    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time

        if self.epoch_log_keys is None and self.log_dir is not None:
            self.epoch_log_keys = [key for key in list(logs.keys()) if key != "epoch"]
            f = open(self.train_epoch_fn, "w")
            f.write("Epoch Number,Time Elapsed")
            for key in self.epoch_log_keys:
                f.write(",%s" % (key))
            f.write("\n")
            f.close()

        if self.log_dir is not None:
            f = open(self.train_epoch_fn, "a")
            f.write("%d,%f" % (self.epoch_num, t))
            for key in self.epoch_log_keys:
                f.write(",%f" % (logs[key]))
            f.write("\n")
            f.close()

        self.epoch_num += 1
