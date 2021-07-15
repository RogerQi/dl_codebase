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
    elif task_name == "few_shot_semantic_segmentation_fine_tuning":
        from .fs_ft_seg_trainer import fs_ft_seg_trainer as fs_ft_seg_trainer_fn
        return fs_ft_seg_trainer_fn
    elif task_name == "incremental_few_shot_segmentation":
        from .GIFS_seg_trainer import GIFS_seg_trainer as GIFS_seg_trainer_fn
        return GIFS_seg_trainer_fn
    else:
        raise NotImplementedError