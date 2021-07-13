import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    task_name = cfg.task
    assert task_name != "none"
    if task_name == "classification":
        from .clf_trainer import clf_trainer
        return clf_trainer
    else:
        raise NotImplementedError