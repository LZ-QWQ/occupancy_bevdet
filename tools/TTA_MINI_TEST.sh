#!/usr/bin/env bash

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500} #29500
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# CHECKPOINT=${CHECKPOINT:"work_dirs/bevdet-occ-tenimage_B_custom_decay-4d-stereo-512x1408-24e-labelsmooth_0.00001-load_cbgs/epoch_14_ema.pth "}
# SAVE_DIR=${SAVE_DIR:"submission_prefix=./results/merge_logit/internimage_14ema_TTA"}
CHECKPOINT=$1
GPUS=$2


CONFIG_PATH="./configs/bevdet_occ/TTA_MINI_FOR_SAVE/"
ls $CONFIG_PATH | while read CONFIG_NAME
do
    echo "[INFO]" ${CONFIG_NAME} "test start"
    CONFIG=${CONFIG_PATH}${CONFIG_NAME}
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch \
        --nnodes=$NNODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --nproc_per_node=$GPUS \
        --master_port=$PORT \
        $(dirname "$0")/test.py \
        $CONFIG \
        $CHECKPOINT \
        --launcher pytorch \
        ${@:3}
    echo "[INFO]" ${CONFIG_NAME} "test done"
done