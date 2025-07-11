#!/usr/bin/env bash

set -x

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
WORKDIR=$4                                                                                                                                         
NNODES=${NNODES:-1}     
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# Arguments starting from the forth one are captured by ${@:4}
python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --dump --work-dir=$WORKDIR --launcher pytorch ${@:5}
