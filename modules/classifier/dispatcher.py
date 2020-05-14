import torch
import torch.nn as nn
import torch.nn.functional as F

def dispatcher(cfg, feature_size):
    classifier_name = cfg.CLASSIFIER.classifier
    assert classifier_name != "none"
    if classifier_name == "fc":
        import classifier.fc as fc
        fc_classifier = fc.net(cfg, feature_size)
        return fc_classifier
    else:
        raise NotImplementedError