import argparse
from pathlib import Path
import time
import datetime
import sys
import numpy as np

# pylint: disable=wrong-import-position
sys.path.append("landcover")
from train_model_landcover import Train
from testing_model_landcover import Test
from compute_accuracy import compute_accuracy
from eval_landcover_results import accuracy_jaccard_np
import config

from helpers import get_logger

logger = get_logger(__name__)


def do_args():
    parser = argparse.ArgumentParser(
        description="Wrapper utility for training and testing land cover models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help="Verbosity of keras.fit",
        default=config.VERBOSE,
    )
    parser.add_argument("--name", type=str, help="Experiment name", required=True)
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to data directory containing the splits CSV files",
        default=config.DATA_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output base directory",
        default=config.OUTPUT_DIR,
    )
    parser.add_argument(
        "--training-states",
        nargs="+",
        type=str,
        help="States to use as training",
        default=config.TRAINING_STATES,
    )
    parser.add_argument(
        "--validation-states",
        nargs="+",
        type=str,
        help="States to use as validation",
        default=config.VALIDATION_STATES,
    )
    parser.add_argument(
        "--superres-states",
        nargs="+",
        type=str,
        help="States to use only superres loss with",
        default=config.SUPERRES_STATES,
    )
    parser.add_argument(
        "--test-states",
        nargs="+",
        type=str,
        help="States to test model with",
        default=config.TEST_STATES,
    )
    parser.add_argument(
        "--do-color",
        action="store_true",
        help="Enable color augmentation",
        default=config.DO_COLOR,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL,
        choices=["unet", "unet_large", "fcdensenet", "fcn_small"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs", default=config.EPOCHS
    )
    parser.add_argument(
        "--loss",
        type=str,
        help="Loss function",
        default=config.LOSS,
        choices=["crossentropy", "jaccard", "superres"],
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate",
        default=config.LEARNING_RATE,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size", default=config.BATCH_SIZE
    )
    return parser.parse_args()


def main():
    # Read arguments
    args = do_args()
    start_time = float(time.time())
    logger.info("Starting at %s", str(datetime.datetime.now()))
    logger.info(args)

    # Ensure folders are there and no overwrite
    logger.info("Ensuring all folders are there...")
    assert Path(args.data_dir).is_dir(), (
        "DATA_DIR (%s) does not exist. Make sure path is correct." % args.data_dir
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    assert Path(args.output_dir).is_dir(), (
        "OUTPUT_DIR (%s) does not exist. Make sure path is correct." % args.output_dir
    )
    assert not (Path(args.output_dir) / Path(args.name)).is_dir(), (
        "EXPERIMENT_DIR (%s) already exists. Change name or delete directory."
        % (args.output_dir + args.name)
    )

    # Run training
    train = Train(
        name=args.name,
        output=args.output_dir,
        data_dir=args.data_dir,
        training_states=args.training_states,
        validation_states=args.validation_states,
        superres_states=args.superres_states,
        model_type=args.model,
        loss=args.loss,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        do_color=args.do_color,
        batch_size=args.batch_size,
    )
    train.run_experiment()

    cm = np.zeros((config.HR_NCLASSES - 1, config.HR_NCLASSES - 1), dtype=np.float32)
    cm_dev = np.zeros(
        (config.HR_NCLASSES - 1, config.HR_NCLASSES - 1), dtype=np.float32
    )
    for test_state in args.test_states:
        # Run testing
        ## Get test file name
        input_fn = Path(args.data_dir) / ("%s_extended-test_tiles.csv" % test_state)

        ## Get model file name
        model_fn = Path(args.output_dir) / args.name / "final_model.h5"

        prediction_dir = (
            Path(args.output_dir) / args.name / ("test-output_%s" % test_state)
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)

        test = Test(
            input_fn=input_fn,
            output_base=prediction_dir,
            model_fn=model_fn,
            save_probabilities=False,
            superres=args.loss == "superres",
        )
        test.run_on_tiles()

        # Run accuracy
        acc, cm_s, cm_dev_s = compute_accuracy(
            pred_dir=prediction_dir, input_fn=input_fn
        )
        logger.info("Overall accuracy for %s: %.4f", test_state, acc)

        # Confusion matrices
        cm += cm_s
        cm_dev += cm_dev_s

    # Run eval
    logger.info("-----------------------------")
    logger.info("OVERALL METRICS")
    logger.info("-----------------------------")
    logger.info("Accuracy and jaccard of all pixels")
    accuracy_jaccard_np(cm)
    logger.info("Accuracy and jaccard of pixels with developed NLCD classes")
    accuracy_jaccard_np(cm_dev)

    logger.info("Finished at %s", str(datetime.datetime.now()))
    logger.info("Finished in %0.4f seconds", float(time.time()) - start_time)


if __name__ == "__main__":
    main()
