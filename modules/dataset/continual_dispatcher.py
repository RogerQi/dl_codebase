def continual_dispatcher(cfg):
    dataset_name = cfg.DATASET.dataset
    if dataset_name == "pascal_5i":
        from .pascal_5i import get_continual_vanilla_train_set, get_continual_aug_train_set, get_continual_test_set
    else:
        raise NotImplementedError
    return [get_continual_vanilla_train_set(cfg), get_continual_aug_train_set(cfg), get_continual_test_set(cfg)]
