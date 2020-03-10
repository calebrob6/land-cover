#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Caleb Robinson <calebrob6@gmail.com>
# and
# Le Hou <lehou0312@gmail.com>
"""Script for running a saved model file on a list of tiles.
"""
# Stdlib imports
import sys
import os

import time
import subprocess
import datetime
import argparse

# Library imports
import numpy as np
import pandas as pd

import rasterio

import keras
from keras.models import Model
import keras.models
import keras.metrics

from helpers import get_logger
from utils import to_float

logger = get_logger(__name__)


def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument(
        "--input",
        action="store",
        dest="input_fn",
        type=str,
        required=True,
        help="Path to filename that lists tiles",
    )
    parser.add_argument(
        "--output",
        action="store",
        dest="output_base",
        type=str,
        required=True,
        help="Output directory to store predictions",
    )
    parser.add_argument(
        "--model",
        action="store",
        dest="model_fn",
        type=str,
        required=True,
        help="Path to Keras .h5 model file to use",
    )
    parser.add_argument(
        "--save_probabilities",
        action="store_true",
        default=False,
        help="Enable outputing grayscale probability maps for each class",
    )
    parser.add_argument(
        "--superres",
        action="store_true",
        dest="superres",
        default=False,
        help="Is this a superres model",
    )
    parser.add_argument(
        "--classes", dest="classes", default=5, help="Number of target classes",
    )

    return parser.parse_args(arg_list)


class Test:
    def __init__(
        self,
        input_fn: str,
        output_base: str,
        model_fn: str,
        save_probabilities: bool,
        superres: bool,
        classes: int = 5,
    ):
        """Constructor for Test object.

        Parameters
        ----------
        input_fn : str
            Path to csv filename that lists tiles.
        output_base : str
            Output directory to store predictions.
        model_fn : str
            Path to Keras .h5 model file to use.
        save_probabilities : bool
            Enable outputing grayscale probability maps for each class.
        superres : bool
            Is this a superres model?.

        """
        self.input_fn = input_fn
        self.data_dir = os.path.dirname(input_fn)
        self.output_base = output_base
        self.model_fn = model_fn
        self.save_probabilities = save_probabilities
        self.superres = superres
        self.classes = 5
        self.start_time = None
        self.end_time = None

    @staticmethod
    def run_model_on_tile(
        model: Model,
        naip_tile: np.array,
        inpt_size: int,
        output_size: int,
        batch_size: int,
    ):
        """Run a model on imagery tile.

        Parameters
        ----------
        model : Model
            Keras Model object.
        naip_tile : np.array
            numpy array with imagery.
        inpt_size : int
            Height and width of input.
        output_size : int
            Number of channels of output - # classes.
        batch_size : int
            Batch size to predict with.

        Returns
        -------
        np.array
            Array with predictions. Each channel is probability of each class.

        """
        down_weight_padding = 40
        height = naip_tile.shape[0]
        width = naip_tile.shape[1]

        stride_x = inpt_size - down_weight_padding * 2
        stride_y = inpt_size - down_weight_padding * 2

        output = np.zeros((height, width, output_size), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32) + 0.000000001
        kernel = np.ones((inpt_size, inpt_size), dtype=np.float32) * 0.1
        kernel[10:-10, 10:-10] = 1
        kernel[
            down_weight_padding : down_weight_padding + stride_y,
            down_weight_padding : down_weight_padding + stride_x,
        ] = 5

        batch = []
        batch_indices = []
        batch_count = 0

        for y_index in list(range(0, height - inpt_size, stride_y)) + [
            height - inpt_size,
        ]:
            for x_index in list(range(0, width - inpt_size, stride_x)) + [
                width - inpt_size,
            ]:
                naip_im = naip_tile[
                    y_index : y_index + inpt_size, x_index : x_index + inpt_size, :
                ]

                batch.append(naip_im)
                batch_indices.append((y_index, x_index))
                batch_count += 1

        model_output = model.predict(np.array(batch), batch_size=batch_size, verbose=0)

        for i, (y, x) in enumerate(batch_indices):
            output[y : y + inpt_size, x : x + inpt_size] += (
                model_output[i] * kernel[..., np.newaxis]
            )
            counts[y : y + inpt_size, x : x + inpt_size] += kernel

        return output / counts[..., np.newaxis]

    def load_tiles(self) -> list:
        """Load list of tiles.

        Returns
        -------
        list
            List of lists of imagery, HR labels and LR labels file names.

        """
        try:
            df = pd.read_csv(self.input_fn)
            fns = df[["naip-new_fn", "lc_fn", "nlcd_fn"]].values
            return fns
        except Exception as e:
            logger.error("Could not load the input file")
            logger.error(e)
            return

    def load_model(self) -> Model:
        """Load model from file.

        Returns
        -------
        Model
            Loaded previously trained model.

        """
        model = keras.models.load_model(
            self.model_fn,
            custom_objects={
                "jaccard_loss": keras.metrics.mean_squared_error,
                "loss": keras.metrics.mean_squared_error,
            },
        )

        if self.superres:
            model = keras.models.Model(input=model.inputs, outputs=[model.outputs[0]])
            model.compile("sgd", "mse")

        output_shape = model.output_shape[1:]
        input_shape = model.input_shape[1:]
        model_input_size = input_shape[0]
        assert (
            len(model.outputs) == 1
        ), "The loaded model has multiple outputs. You need to specify --superres if this model was trained with the superres loss."
        return model

    def run_on_tiles(self):
        """Run inference on list of tiles.

        """
        logger.info(
            "Starting %s at %s"
            % ("Model inference script", str(datetime.datetime.now()))
        )
        self.start_time = float(time.time())

        fns = self.load_tiles()
        model = self.load_model()

        for i in range(len(fns)):
            tic = float(time.time())
            naip_fn = os.path.join(self.data_dir, fns[i][0])
            lc_fn = os.path.join(self.data_dir, fns[i][1])
            nlcd_fn = os.path.join(self.data_dir, fns[i][2])

            logger.info("Running model on %s\t%d/%d" % (naip_fn, i + 1, len(fns)))

            naip_fid = rasterio.open(naip_fn, "r")
            naip_profile = naip_fid.meta.copy()
            naip_tile = to_float(naip_fid.read().astype(np.float32))
            naip_tile = np.rollaxis(naip_tile, 0, 3)
            naip_fid.close()

            output = self.run_model_on_tile(
                model,
                naip_tile,
                model.input_shape[1:][0],
                model.output_shape[1:][2],
                16,
            )
            output = output[:, :, : self.classes]

            # ----------------------------------------------------------------
            # Write out each softmax prediction to a separate file
            # ----------------------------------------------------------------
            if self.save_probabilities:
                output_fn = os.path.basename(naip_fn)[:-4] + "_prob.tif"
                current_profile = naip_profile.copy()
                current_profile["driver"] = "GTiff"
                current_profile["dtype"] = "uint8"
                current_profile["count"] = self.classes
                current_profile["compress"] = "lzw"

                # quantize the probabilities
                bins = np.arange(256)
                bins = bins / 255.0
                output = np.digitize(output, bins=bins, right=True).astype(np.uint8)

                with rasterio.open(
                    os.path.join(self.output_base, output_fn), "w", **current_profile
                ) as f:
                    for i in range(self.classes):
                        f.write(output[:, :, i], i + 1)

            # ----------------------------------------------------------------
            # Write out the class predictions
            # ----------------------------------------------------------------
            output_classes = np.argmax(output, axis=2).astype(np.uint8)
            output_class_fn = os.path.basename(naip_fn)[:-4] + "_class.tif"

            current_profile = naip_profile.copy()
            current_profile["driver"] = "GTiff"
            current_profile["dtype"] = "uint8"
            current_profile["count"] = 1
            current_profile["compress"] = "lzw"
            f = rasterio.open(
                os.path.join(self.output_base, output_class_fn), "w", **current_profile
            )
            f.write(output_classes, 1)
            f.close()

            logger.info("Finished iteration in %0.4f seconds" % (time.time() - tic))

        self.end_time = float(time.time())
        logger.info(
            "Finished %s in %0.4f seconds"
            % ("Model inference script", self.end_time - self.start_time)
        )


def main():
    program_name = "Model inference script"
    args = do_args(sys.argv[1:], program_name)
    vars_args = vars(args)
    Test(**vars_args).run_on_tiles()


if __name__ == "__main__":
    main()
