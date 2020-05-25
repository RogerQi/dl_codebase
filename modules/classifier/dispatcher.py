import torch
import torch.nn as nn
import torch.nn.functional as F

class identity_mod(nn.Module):
    def forward(self, x):
        return x

def dispatcher(cfg, feature_size):
    classifier_name = cfg.CLASSIFIER.classifier
    assert classifier_name != "none"
    if classifier_name == "fc":
        import classifier.fc as fc
        fc_classifier = fc.net(cfg, feature_size)
        return fc_classifier
    elif classifier_name == "identity":
        identity_module = identity_mod()
        return identity_module
    else:
        raise NotImplementedError