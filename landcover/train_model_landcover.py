#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Copyright © 2018 Caleb Robinson <calebrob6@gmail.com>
#
# Distributed under terms of the MIT license.
"""Training CVPR models
"""
import sys
import os

# Here we look through the args to find which GPU we should use
# We must do this before importing keras, which is super hacky
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
# TODO: This _really_ should be part of the normal argparse code.
def parse_args(args, key):
    def is_int(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    for i, arg in enumerate(args):
        if arg == key:
            if i + 1 < len(sys.argv):
                if is_int(args[i + 1]):
                    return args[i + 1]
    return None


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = parse_args(sys.argv, "--gpu")
if GPU_ID is not None:  # if we passed `--gpu INT`, then set the flag, else don't
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import shutil
import time
import argparse
import datetime

import numpy as np
import pandas as pd

import utils
import models
import datagen

from keras.optimizers import RMSprop, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint


def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument(
        "-v",
        "--verbose",
        action="store",
        dest="verbose",
        type=int,
        help="Verbosity of keras.fit",
        default=2,
    )
    parser.add_argument(
        "--output",
        action="store",
        dest="output",
        type=str,
        help="Output base directory",
        required=True,
    )
    parser.add_argument(
        "--name",
        action="store",
        dest="name",
        type=str,
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "--gpu",
        action="store",
        dest="gpu",
        type=int,
        help="GPU id to use",
        required=False,
    )

    parser.add_argument(
        "--data_dir",
        action="store",
        dest="data_dir",
        type=str,
        help="Path to data directory containing the splits CSV files",
        required=True,
    )

    parser.add_argument(
        "--training_states",
        action="store",
        dest="training_states",
        nargs="+",
        type=str,
        help="States to use as training",
        required=True,
    )
    parser.add_argument(
        "--validation_states",
        action="store",
        dest="validation_states",
        nargs="+",
        type=str,
        help="States to use as validation",
        required=True,
    )
    parser.add_argument(
        "--superres_states",
        action="store",
        dest="superres_states",
        nargs="+",
        type=str,
        help="States to use only superres loss with",
        default="",
    )

    parser.add_argument(
        "--color", action="store_true", help="Enable color augmentation", default=False
    )

    parser.add_argument(
        "--model_type",
        action="store",
        dest="model_type",
        type=str,
        choices=["unet", "unet_large", "fcdensenet", "fcn_small"],
        help="Model architecture to use",
        required=True,
    )
    parser.add_argument(
        "--epochs", action="store", type=int, help="Number of epochs", default=100
    )
    parser.add_argument(
        "--learning_rate",
        action="store",
        type=float,
        help="Learning rate",
        default=0.001,
    )
    parser.add_argument(
        "--loss",
        action="store",
        type=str,
        help="Loss function",
        choices=["crossentropy", "jaccard", "superres"],
        required=True,
    )

    parser.add_argument(
        "--batch_size", action="store", type=eval, help="Batch size", default="128"
    )

    return parser.parse_args(arg_list)


def main():
    prog_name = sys.argv[0]
    args = do_args(sys.argv[1:], prog_name)

    verbose = args.verbose
    output = args.output
    name = args.name

    data_dir = args.data_dir
    training_states = args.training_states
    validation_states = args.validation_states
    superres_states = args.superres_states

    num_epochs = args.epochs
    model_type = args.model_type
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    loss = args.loss
    do_color_aug = args.color
    do_superres = loss == "superres"

    log_dir = os.path.join(output, name)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    f = open(os.path.join(log_dir, "args.txt"), "w")
    for k, v in args.__dict__.items():
        f.write("%s,%s\n" % (str(k), str(v)))
    f.close()

    print("Starting %s at %s" % (prog_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    # ------------------------------
    # Step 1, load data
    # ------------------------------

    training_patches = []
    for state in training_states:
        print("Adding training patches from %s" % (state))
        fn = os.path.join(data_dir, "%s_extended-train_patches.csv" % (state))
        df = pd.read_csv(fn)
        for fn in df["patch_fn"].values:
            training_patches.append((os.path.join(data_dir, fn), state))

    validation_patches = []
    for state in validation_states:
        print("Adding validation patches from %s" % (state))
        fn = os.path.join(data_dir, "%s_extended-val_patches.csv" % (state))
        df = pd.read_csv(fn)
        for fn in df["patch_fn"].values:
            validation_patches.append((os.path.join(data_dir, fn), state))

    print(
        "Loaded %d training patches and %d validation patches"
        % (len(training_patches), len(validation_patches))
    )

    if do_superres:
        print("Using %d states in superres loss:" % (len(superres_states)))
        print(superres_states)

    # ------------------------------
    # Step 2, run experiment
    # ------------------------------
    # training_steps_per_epoch = 1#len(training_patches) // batch_size
    # validation_steps_per_epoch = 1 #len(validation_patches) // batch_size

    training_steps_per_epoch = 300
    validation_steps_per_epoch = 39
    # TODO: For testing - best if adaptable
    if training_steps_per_epoch * batch_size > len(training_patches):
        training_steps_per_epoch = 1
        validation_steps_per_epoch = 1

    print(
        "Number of training/validation steps per epoch: %d/%d"
        % (training_steps_per_epoch, validation_steps_per_epoch)
    )

    # Build the model
    optimizer = RMSprop(learning_rate)
    if model_type == "unet":
        model = models.unet((240, 240, 4), 5, optimizer, loss)
    elif model_type == "unet_large":
        model = models.unet_large((240, 240, 4), 5, optimizer, loss)
    elif model_type == "fcdensenet":
        model = models.fcdensenet((240, 240, 4), 5, optimizer, loss)
    elif model_type == "fcn_small":
        model = models.fcn_small((240, 240, 4), 5, optimizer, loss)
    model.summary()

    validation_callback = utils.LandcoverResults(log_dir=log_dir, verbose=verbose)
    learning_rate_callback = LearningRateScheduler(
        utils.schedule_stepped, verbose=verbose
    )

    model_checkpoint_callback = ModelCheckpoint(
        os.path.join(log_dir, "model_{epoch:02d}.h5"),
        verbose=verbose,
        save_best_only=False,
        save_weights_only=False,
        period=20,
    )

    training_generator = datagen.DataGenerator(
        training_patches,
        batch_size,
        training_steps_per_epoch,
        240,
        240,
        4,
        do_color_aug=do_color_aug,
        do_superres=do_superres,
        superres_only_states=superres_states,
    )
    validation_generator = datagen.DataGenerator(
        validation_patches,
        batch_size,
        validation_steps_per_epoch,
        240,
        240,
        4,
        do_color_aug=do_color_aug,
        do_superres=do_superres,
        superres_only_states=[],
    )

    model.fit_generator(
        training_generator,
        steps_per_epoch=training_steps_per_epoch,
        # epochs=10**6,
        epochs=num_epochs,
        verbose=verbose,
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        max_queue_size=256,
        workers=4,
        use_multiprocessing=True,
        callbacks=[
            validation_callback,
            # learning_rate_callback,
            model_checkpoint_callback,
        ],
        initial_epoch=0,
    )

    # ------------------------------
    # Step 3, save models
    # ------------------------------
    model.save(os.path.join(log_dir, "final_model.h5"))

    model_json = model.to_json()
    with open(os.path.join(log_dir, "final_model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(log_dir, "final_model_weights.h5"))

    print("Finished in %0.4f seconds" % (time.time() - start_time))
    del model, training_generator, validation_generator


if __name__ == "__main__":
    main()
    sys.exit(0)
