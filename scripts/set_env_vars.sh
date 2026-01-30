# ----------------------------------------------------------------------------------------
# setup environment variables
# disable TF verbose logging
TF_CPP_MIN_LOG_LEVEL=2
# fix known issues for pytorch-1.5.1 accroding to 
# https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
MKL_THREADING_LAYER=GNU
# set NCCL envs for disributed communication
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
NCCL_DEBUG=INFO
ARNOLD_FRAMEWORK=pytorch

# get distributed training parameters 
NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


###############################
### Set the torchrun related parameters. The default is for a single node.
###############################
export GPUS=$NV_GPUS
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export PORT=${PORT:-55565}
