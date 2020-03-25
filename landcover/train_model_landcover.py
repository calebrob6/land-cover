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

import shutil
import time
import argparse
import datetime
from typing import Tuple

import numpy as np
import pandas as pd

import utils
import models
import datagen

from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

from helpers import get_logger
import config

logger = get_logger(__name__)


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
        "--do_color",
        action="store_true",
        help="Enable color augmentation",
        default=False,
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
        "--batch_size", action="store", type=int, help="Batch size", default=128
    )
    parser.add_argument(
        "--preload_weights",
        type=str,
        default="",
        help="Path to H5 containing weights to preload for training. Make sure architecture is the same.",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="int8",
        help="Data type of imagery patches. int8 or int16",
    )

    return parser.parse_args(arg_list)


class Train:
    """
    Wrapper class for all training parameters and methods.
    """

    def __init__(
        self,
        output: str,
        name: str,
        data_dir: str,
        training_states: list,
        validation_states: list,
        superres_states: list,
        model_type: str,
        loss: str,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        do_color: bool = False,
        do_superres: bool = False,
        input_shape: tuple = (240, 240, 4),
        classes: int = config.HR_NCLASSES,
        verbose: int = 2,
        lr_num_classes: int = config.LR_NCLASSES,
        hr_labels_index: int = config.HR_LABEL_INDEX,
        lr_labels_index: int = config.LR_LABEL_INDEX,
        hr_label_key: str = config.HR_LABEL_KEY,
        lr_label_key: str = config.LR_LABEL_KEY,
        preload_weights: str = config.PRELOAD_WEIGHTS,
        data_type: str = config.DATA_TYPE,
    ):
        """Constructor for Train object.

        Parameters
        ----------
        output : str
            Directory to place all output files of training runs.
        name : str
            Name of experiment or training run (should be unique).
        data_dir : str
            Path to data directory containing the CSV file pointing to train
            and validation patches.
        training_states : list
            States (datasets) to use as training.
        validation_states : list
            States (datasets) to use as validation.
        superres_states : list
            States (datasets) to use only superres loss with.
        epochs : int
            Number of epochs for training.
        batch_size : int
            Batch size for training.
        model_type : str
            Model architecture to use - one of ["unet", "unet_large",
            "fcdensenet", "fcn_small"].
        learning_rate : float
            Learning rate for model training.
        loss : str
            Loss function to use - one of ["crossentropy", "jaccard", "superres"].
        do_color : bool
            Use color augmentation method.
        do_superres : bool
            Use superresolution augmentation method.
        input_shape : tuple
            Shape of input data (w, h, c).
        classes : int
            Number of target classes (number of classes in training data).
        verbose : int
            Level of verbosity of fit method (passed to keras) 0 to 2 (silent to verbose).
        preload_weights : str
            Path to H5 containing weights to preload for training. Make sure architecture is the same.
        data_type: str
            Data type of imagery patches. int8 or int16
        """
        self.verbose = verbose
        self.output = output
        self.name = name

        self.data_dir = data_dir
        self.training_states = training_states
        self.validation_states = validation_states
        self.superres_states = superres_states

        self.input_shape = input_shape
        self.classes = classes

        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.loss = loss

        self.do_color = do_color
        self.do_superres = loss == "superres"

        self.log_dir = os.path.join(output, name)

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.training_steps_per_epoch = 300
        self.validation_steps_per_epoch = 39

        self.lr_num_classes = lr_num_classes
        self.hr_labels_index = hr_labels_index
        self.lr_labels_index = lr_labels_index
        self.hr_label_key = hr_label_key
        self.lr_label_key = lr_label_key

        self.preload_weights = preload_weights
        self.data_type = data_type

        self.write_args()

        self.start_time = None
        self.end_time = None

    def write_args(self):
        """Write arguments into file.

        """
        f = open(os.path.join(self.log_dir, "args.txt"), "w")
        for k, v in self.__dict__.items():
            f.write("%s,%s\n" % (str(k), str(v)))
        f.close()

    def load_data(self) -> Tuple[datagen.DataGenerator, datagen.DataGenerator]:
        """Load patches from csv file for training and validation.

        Returns
        -------
        Tuple[datagen.DataGenerator, datagen.DataGenerator]
            Generator of data for training [0] and validation [1]
        """
        training_patches = []
        for state in self.training_states:
            logger.info("Adding training patches from %s" % (state))
            fn = os.path.join(self.data_dir, "%s_extended-train_patches.csv" % (state))
            if not os.path.isfile(fn):
                fn = os.path.join(self.data_dir, "%s-train_patches.csv" % (state))
            df = pd.read_csv(fn)
            for fn in df["patch_fn"].values:
                training_patches.append((os.path.join(self.data_dir, fn), state))

        validation_patches = []
        for state in self.validation_states:
            logger.info("Adding validation patches from %s" % (state))
            fn = os.path.join(self.data_dir, "%s_extended-val_patches.csv" % (state))
            if not os.path.isfile(fn):
                fn = os.path.join(self.data_dir, "%s-val_patches.csv" % (state))
            df = pd.read_csv(fn)
            for fn in df["patch_fn"].values:
                validation_patches.append((os.path.join(self.data_dir, fn), state))

        logger.info(
            "Loaded %d training patches and %d validation patches"
            % (len(training_patches), len(validation_patches))
        )

        if self.training_steps_per_epoch * self.batch_size > len(training_patches):
            logger.info("Number of train patches is insufficient. Assuming testing...")
            self.training_steps_per_epoch = 1
            self.validation_steps_per_epoch = 1

        if self.do_superres:
            logger.info(
                "Using %d states in superres loss:" % (len(self.superres_states))
            )
            logger.info(self.superres_states)

        training_generator = datagen.DataGenerator(
            training_patches,
            self.batch_size,
            self.training_steps_per_epoch,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            num_classes=self.classes,
            lr_num_classes=self.lr_num_classes,
            hr_labels_index=self.hr_labels_index,
            lr_labels_index=self.lr_labels_index,
            hr_label_key=self.hr_label_key,
            lr_label_key=self.lr_label_key,
            do_color_aug=self.do_color,
            do_superres=self.do_superres,
            superres_only_states=self.superres_states,
            data_type=self.data_type,
        )
        validation_generator = datagen.DataGenerator(
            validation_patches,
            self.batch_size,
            self.validation_steps_per_epoch,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            num_classes=self.classes,
            lr_num_classes=self.lr_num_classes,
            hr_labels_index=self.hr_labels_index,
            lr_labels_index=self.lr_labels_index,
            hr_label_key=self.hr_label_key,
            lr_label_key=self.lr_label_key,
            do_color_aug=self.do_color,
            do_superres=self.do_superres,
            superres_only_states=[],
            data_type=self.data_type,
        )
        return training_generator, validation_generator

    def get_model(self) -> Model:
        """Get selected model.

        Returns
        -------
        Model
            Keras model object, compiled.

        """
        # Build the model
        optimizer = RMSprop(self.learning_rate)
        if self.model_type == "unet":
            model = models.unet(self.input_shape, self.classes, optimizer, self.loss)
        elif self.model_type == "unet_large":
            model = models.unet_large(
                self.input_shape, self.classes, optimizer, self.loss
            )
        elif self.model_type == "fcdensenet":
            model = models.fcdensenet(
                self.input_shape, self.classes, optimizer, self.loss
            )
        elif self.model_type == "fcn_small":
            model = models.fcn_small(
                self.input_shape, self.classes, optimizer, self.loss
            )
        if self.preload_weights:
            logger.info("=====================================================")
            logger.info(f"Using weights from {self.preload_weights}")
            logger.info("=====================================================")
            model.load_weights(self.preload_weights)
        model.run_eagerly = True
        model.summary()
        return model

    def save_model(self, model: Model):
        """Save Keras trained model and weights into files.

        Parameters
        ----------
        model : Model
            Keras model object, trained.

        """
        model.save(os.path.join(self.log_dir, "final_model.h5"))

        model_json = model.to_json()
        with open(os.path.join(self.log_dir, "final_model.json"), "w") as json_file:
            json_file.write(model_json)
        model.save_weights(os.path.join(self.log_dir, "final_model_weights.h5"))

    def run_experiment(
        self,
        learning_rate_flag: bool = False,
        max_queue_size: int = 256,
        workers: int = 4,
    ):
        """Run training job.

        Parameters
        ----------
        learning_rate_flag : bool
            Set to true to use learning rate callback.
        max_queue_size : int
            Maximum queue size for Keras fit_generator.
        workers : int
            Number of workers for Keras fit_generator.
        """
        logger.info("Starting %s at %s" % (self.name, str(datetime.datetime.now())))
        self.start_time = float(time.time())

        logger.info(
            "Number of training/validation steps per epoch: %d/%d"
            % (self.training_steps_per_epoch, self.validation_steps_per_epoch)
        )

        model = self.get_model()

        validation_callback = utils.LandcoverResults(
            log_dir=self.log_dir, verbose=self.verbose
        )
        learning_rate_callback = LearningRateScheduler(
            utils.schedule_stepped, verbose=self.verbose
        )

        model_checkpoint_callback = ModelCheckpoint(
            os.path.join(self.log_dir, "model_{epoch:02d}.h5"),
            verbose=self.verbose,
            save_best_only=False,
            save_weights_only=False,
            period=20,
        )
        log_dir_tag = f"logs/{self.name} {str(datetime.datetime.now())}"
        tensorboard_callback = TensorBoard(log_dir=log_dir_tag)

        training_generator, validation_generator = self.load_data()

        model_diagnoser = utils.ModelDiagnoser(
            training_generator,
            self.batch_size,
            1,
            f"{log_dir_tag}/images",
            self.do_superres,
        )

        callbacks = [
            validation_callback,
            model_checkpoint_callback,
            tensorboard_callback,
            model_diagnoser,
        ]

        if learning_rate_flag:
            callback.append(learning_rate_callback)

        model.fit_generator(
            training_generator,
            steps_per_epoch=self.training_steps_per_epoch,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=validation_generator,
            validation_steps=self.validation_steps_per_epoch,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=True,
            callbacks=callbacks,
            initial_epoch=0,
        )

        self.save_model(model)

        self.end_time = float(time.time())
        logger.info("Finished in %0.4f seconds" % (self.end_time - self.start_time))


def main():
    prog_name = sys.argv[0]
    args = do_args(sys.argv[1:], prog_name)
    vars_args = vars(args)
    Train(**vars_args).run_experiment()


if __name__ == "__main__":
    main()
