def dispatcher(cfg):
    network_name = cfg.BACKBONE.network
    if network_name == "lenet":
        from backbone.lenet import net
        return net
    elif network_name == "resnet18":
        from backbone.resnet import resnet18
        return resnet18
    else:
        raise NotImplementedError