def meta_dispatcher(cfg):
    dataset_name = cfg.DATASET.dataset
    if dataset_name == "pascal_5i":
        from .pascal_5i import get_meta_train_set, get_meta_test_set
    elif dataset_name == "mini_imagenet":
        from .mini_imagenet import get_meta_train_set, get_meta_test_set
    elif dataset_name == "coco_20i":
        from .coco_20i import get_meta_train_set, get_meta_test_set
    else:
        raise NotImplementedError
    return [get_meta_train_set(cfg), get_meta_test_set(cfg)]
