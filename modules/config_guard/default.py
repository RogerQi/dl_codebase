import os
import sys

from yacs.config import CfgNode as CN

# ----------------------------
# | Start Default Config
# ----------------------------

# Root Config Node
_C = CN()

# DL System Setting
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1		# Number of GPUs to use
_C.SYSTEM.NUM_WORKERS = 4	# Number of CPU workers for errands

# Dataset Settings
_C.DATASET = CN()
_C.DATASET.DATASET = 'cifar10'

# Training Settings
_C.TRAIN = CN()
_C.TRAIN.initial_lr = 0.01	# Initial Learning Rate
# Learning Rate decay method. Options:
#	- Step Down
#	- Exponential
#	- Cosine
_C.TRAIN.lr_decay = 'none'

# ---------------------------
# | End Default Config
# ---------------------------

def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.freeze()

if __name__ == "__main__":
    # debug print
    print(_C)
