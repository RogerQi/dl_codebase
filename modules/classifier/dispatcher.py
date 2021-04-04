import torch
import torch.nn as nn
import torch.nn.functional as F

class identity_mod(nn.Module):
    def forward(self, x):
        return x

def dispatcher(cfg, feature_shape):
    classifier_name = cfg.CLASSIFIER.classifier
    assert classifier_name != "none"
    if classifier_name == "fc":
        import classifier.fc as fc
        fc_classifier = fc.fc(cfg, feature_shape)
        return fc_classifier
    elif classifier_name == "c1":
        import classifier.c1 as c1
        c1_seghead = c1.c1(cfg, feature_shape)
        return c1_seghead
    elif classifier_name == "fcn32s":
        import classifier.fcn as fcn
        fcn32s_head = fcn.fcn32s(cfg, feature_shape)
        return fcn32s_head
    elif classifier_name == "fcn32s_cos":
        import classifier.fcn_cos as fcn_cos
        fcn32s_cos_head = fcn_cos.fcn32s_cos(cfg, feature_shape)
        return fcn32s_cos_head
    elif classifier_name == "identity":
        identity_module = identity_mod()
        return identity_module
    else:
        raise NotImplementedError