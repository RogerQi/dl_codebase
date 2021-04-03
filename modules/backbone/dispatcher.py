def dispatcher(cfg):
    network_name = cfg.BACKBONE.network
    if network_name == "lenet":
        from backbone.lenet import net
        return net
    elif network_name == "resnet18":
        from backbone.resnet import resnet18
        return resnet18
    elif network_name == "resnet32_cifar":
        from backbone.resnet_cifar import resnet32
        return resnet32
    elif network_name == "vgg16":
        # Note that there is NO batch normalization in this VGG16
        # implementation, which hurts performance. This is for
        # consistency with the PANet implementation
        from backbone.vgg import vgg16
        return vgg16
    elif network_name == "vgg16_seg":
        from backbone.vgg import vgg16_seg
        return vgg16_seg
    else:
        raise NotImplementedError