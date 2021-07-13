import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg):
    task_name = cfg.task
    assert task_name != "none"
    if task_name == "classification":
        from .clf_trainer import clf_trainer as clf_trainer_fn
        return clf_trainer_fn
    elif task_name == "semantic_segmentation":
        from .seg_trainer import seg_trainer as seg_trainer_fn
        return seg_trainer_fn
    else:
        raise NotImplementedError