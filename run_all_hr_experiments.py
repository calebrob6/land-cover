#!/usr/bin/env python
import os
import datetime
import subprocess
import multiprocessing


DATASET_DIR = "chesapeake_data/"
OUTPUT_DIR = "results/results_epochs_20_5/"

_GPU_IDS = [0, 1]
NUM_GPUS = len(_GPU_IDS)
JOBS_PER_GPU = [[] for i in range(NUM_GPUS)]

# pylint: disable=redefined-outer-name
def run_jobs(jobs):
    print("Starting job runner")
    for (command, args) in jobs:
        print(datetime.datetime.now(), command)

        output_dir = os.path.join(args["output"], args["exp_name"])
        os.makedirs(output_dir, exist_ok=True)

        process = subprocess.Popen(
            command.split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        with open(os.path.join(output_dir, args["log_name"]), "w") as f:
            while process.returncode is None:
                for line in process.stdout:
                    f.write(line.decode("utf-8").strip() + "\n")
                process.poll()


TRAIN_STATE_LIST = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]
TEST_STATE_LIST = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]

GPU_IDX = 0
for train_state in TRAIN_STATE_LIST:
    gpu_id = _GPU_IDS[GPU_IDX]

    args = {
        "output": OUTPUT_DIR,
        "exp_name": "train-output_%s" % (train_state),
        "TRAIN_STATE_LIST": train_state,
        "val_state_list": train_state,
        "superres_state_list": "",
        "gpu": gpu_id,
        "data_dir": DATASET_DIR,
        "log_name": "log.txt",
        "learning_rate": 0.001,
        "loss": "crossentropy",
        "batch_size": 16,
        "model_type": "unet_large",
    }

    command_train = (
        "python landcover/train_model_landcover.py "
        "--output {output} "
        "--name {exp_name} "
        "--gpu {gpu} "
        "--verbose 2 "
        "--data_dir {data_dir} "
        "--training_states {TRAIN_STATE_LIST} "
        "--validation_states {val_state_list} "
        "--model_type {model_type} "
        "--learning_rate {learning_rate} "
        "--loss {loss} "
        "--batch_size {batch_size} "
    ).format(**args)
    JOBS_PER_GPU[GPU_IDX].append((command_train, args))

    for test_state in TEST_STATE_LIST:

        args = {
            "test_csv": "{}/{}_extended-test_tiles.csv".format(DATASET_DIR, test_state),
            "output": "{}/train-output_{}/".format(OUTPUT_DIR, train_state),
            "exp_name": "test-output_{}".format(test_state),
            "gpu": gpu_id,
            "log_name": "log_test_{}.txt".format(test_state),
        }
        command_test = (
            "python landcover/test_model_landcover.py "
            "--input {test_csv} "
            "--output {output}/{exp_name}/ "
            "--model {output}/final_model.h5 "
            "--gpu {gpu}"
        ).format(**args)
        JOBS_PER_GPU[GPU_IDX].append((command_test, args))

        args = args.copy()
        args["log_name"] = "log_acc_{}.txt".format(test_state)
        command_acc = (
            "python landcover/compute_accuracy.py "
            "--input {test_csv} "
            "--output {output}/{exp_name}/"
        ).format(**args)
        JOBS_PER_GPU[GPU_IDX].append((command_acc, args))

    GPU_IDX = (GPU_IDX + 1) % NUM_GPUS


POLL_SZ = NUM_GPUS
POLL = multiprocessing.Poll(NUM_GPUS + 1)
POLL.map(run_jobs, JOBS_PER_GPU)
POLL.close()
POLL.join()
