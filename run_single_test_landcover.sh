#!/bin/bash

LOSSES=(
    "crossentropy"
    "jaccard"
    "superres"
)

MODEL_TYPES=(
    "unet"
    "unet_large"
    "fcdensenet"
)

TEST_SPLITS=(
    ny_1m_2013
)

GPU_ID=3
LOSS=${LOSSES[0]}
MODEL_TYPE=${MODEL_TYPES[0]}

BATCH_SIZE=16
LEARNING_RATE=0.001

TRAIN_STATE_LIST="md_1m_2013"
VAL_STATE_LIST="ny_1m_2013"
SUPERRES_STATE_LIST="ny_1m_2013"

MODEL_FN="model_10.h5"
MODEL_FN_INST=${MODEL_FN%.*}

EXP_NAME=CVPR-for_github-loss-${LOSS}-model-${MODEL_TYPE}-training_states-${TRAIN_STATE_LIST// /-}
EXP_NAME_OUT=${EXP_NAME}-instance-${MODEL_FN_INST}
OUTPUT=/results/train-output/
PRED_OUTPUT=/results/pred-output/

if [ ! -f "${OUTPUT}/${EXP_NAME}/${MODEL_FN}" ]; then
    echo "This experiment hasn't been trained! Exiting..."
    exit
fi

if [ -d "${PRED_OUTPUT}/${EXP_NAME_OUT}" ]; then
    echo "Experiment output ${PRED_OUTPUT}/${EXP_NAME_OUT} exists"
    while true; do
        read -p "Do you wish to overwrite this experiment? [y/n]" yn
        case $yn in
            [Yy]* ) rm -rf ${PRED_OUTPUT}/${EXP_NAME_OUT}; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
fi

mkdir -p ${PRED_OUTPUT}/${EXP_NAME_OUT}

echo ${MODEL_FN} > ${PRED_OUTPUT}/${EXP_NAME_OUT}/model_fn.txt

for TEST_SPLIT in "${TEST_SPLITS[@]}"
do
	echo $TEST_SPLIT
    TEST_CSV=/home/caleb/data//${TEST_SPLIT}_test_tiles.csv
    echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt
    unbuffer python -u landcover/test_model_landcover.py \
        --input ${TEST_CSV} \
        --output ${PRED_OUTPUT}/${EXP_NAME_OUT}/ \
        --model ${OUTPUT}/${EXP_NAME}/${MODEL_FN} \
        --gpu ${GPU_ID} \
        &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt
        #--superres \

    echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt
    unbuffer python -u compute_accuracy.py \
        --input ${TEST_CSV} \
        --output ${PRED_OUTPUT}/${EXP_NAME_OUT} \
        &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt &
done

wait;

echo "./eval_all_landcover_results.sh ${PRED_OUTPUT}/${EXP_NAME_OUT}"

exit 0
