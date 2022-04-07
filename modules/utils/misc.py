import os
import torch

def get_dataset_root():
    try:
        return os.environ['DATASET_ROOT']
    except KeyError:
        return "/Users/rogerq/data"

def guess_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
