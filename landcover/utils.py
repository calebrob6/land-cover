import os
import time

import numpy as np

import keras

import config


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


def load_nlcd_stats(
    stats_mu=config.LR_STATS_MU,
    stats_sigma=config.LR_STATS_SIGMA,
    class_weights=config.LR_CLASS_WEIGHTS,
    lr_classes=config.LR_NCLASSES,
    hr_classes=config.HR_NCLASSES,
):
    stats_mu = np.loadtxt(stats_mu)
    assert lr_classes == stats_mu.shape[0]
    assert hr_classes == (stats_mu.shape[1] + 1)
    nlcd_means = np.concatenate([np.zeros((lr_classes, 1)), stats_mu], axis=1)
    nlcd_means[nlcd_means == 0] = 0.000001
    nlcd_means[:, 0] = 0
    if stats_mu == "data/nlcd_mu.txt":
        nlcd_means = do_nlcd_means_tuning(nlcd_means)

    stats_sigma = np.loadtxt(stats_sigma)
    assert lr_classes == stats_sigma.shape[0]
    assert hr_classes == (stats_sigma.shape[1] + 1)
    nlcd_vars = np.concatenate([np.zeros((lr_classes, 1)), stats_sigma], axis=1)
    nlcd_vars[nlcd_vars < 0.0001] = 0.0001

    if not class_weights:
        nlcd_class_weights = np.ones((lr_classes,))
    else:
        nlcd_class_weights = np.loadtxt(class_weights)
        assert lr_classes == nlcd_class_weights.shape[0]

    return nlcd_class_weights, nlcd_means, nlcd_vars


def do_nlcd_means_tuning(nlcd_means):
    nlcd_means[2:, 1] -= 0
    nlcd_means[3:7, 4] += 0.25
    nlcd_means = nlcd_means / np.maximum(0, nlcd_means).sum(axis=1, keepdims=True)
    nlcd_means[0, :] = 0
    nlcd_means[-1, :] = 0
    return nlcd_means


def find_key_by_str(keys, needle):
    for key in keys:
        if needle in key:
            return key
    raise ValueError("%s not found in keys" % (needle))


class LandcoverResults(keras.callbacks.Callback):
    # pylint: disable=too-many-instance-attributes,super-init-not-called,dangerous-default-value
    def __init__(self, log_dir=None, verbose=False):

        self.mb_log_keys = None
        self.epoch_log_keys = None

        self.verbose = verbose
        self.log_dir = log_dir

        self.batch_num = 0
        self.epoch_num = 0

        self.train_start_time = None
        self.mb_start_time = None
        self.epoch_start_time = None

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
                key for key in list(logs.keys()) if key not in ["batch", "size"]
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
        # total_time = time.time() - self.train_start_time

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


def handle_labels(arr, key_txt):
    key_array = np.loadtxt(key_txt)
    trans_arr = arr

    for translation in key_array:
        # translation is (src label, dst label)
        scr_l, dst_l = translation
        if scr_l != dst_l:
            trans_arr[trans_arr == scr_l] = dst_l

    # translated array
    return trans_arr


def classes_in_key(key_txt):
    key_array = np.loadtxt(key_txt)
    return len(np.unique(key_array[:, 1]))
