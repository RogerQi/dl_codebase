def dispatcher(cfg):
    if cfg.BACKBONE.network == "dropout_lenet":
        from .classification import dropout_lenet
        return dropout_lenet.net