def dispatcher(cfg):
    network_name = cfg.BACKBONE.network
    if network_name == "lenet":
        from backbone.lenet import net
        return net
    else:
        raise NotImplementedError