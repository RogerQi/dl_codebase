def dispatcher(cfg):
    dataset_name = cfg.DATASET.dataset
    if dataset_name == "mnist":
        from .mnist import get_train_set, get_val_set
    elif dataset_name == "cifar10":
        from .cifar10 import get_train_set, get_val_set
    elif dataset_name == "imagenet":
        raise NotImplementedError
    elif dataset_name == "numpy":
        from .generic_np_dataset import get_train_set, get_val_set
    elif dataset_name == "coco2017":
        from .coco import get_train_set, get_val_set
    elif dataset_name == "ade20k":
        from .ade20k import get_train_set, get_val_set
    else:
        raise NotImplementedError
    return [get_train_set(cfg), get_val_set(cfg)]