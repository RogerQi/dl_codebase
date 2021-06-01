import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm

class pixel_classifier(nn.Module):
    def __init__(self, cfg, feature_shape, in_channel, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.class_mat = nn.Conv2d(in_channel, self.num_classes, 1, bias = False)
        self.weight_norm = cfg.CLASSIFIER.SEGHEAD.COSINE.weight_norm
        if self.weight_norm:
            # conv weight shape: (num_classes, in_channel, 1, 1)
            WeightNorm.apply(self.class_mat, 'weight', dim=0)
        self.train_scale_factor = cfg.CLASSIFIER.SEGHEAD.COSINE.train_scale_factor
        self.val_scale_factor = cfg.CLASSIFIER.SEGHEAD.COSINE.val_scale_factor
    
    def forward(self, x):
        '''
        x: (B, in_channel, H, W)
        '''
        # x_norm: (B, in_channel, H, W) where x_norm[i, :, H, W] is the norm of
        # x[i, :, H, W]. That is, x/x_norm yields normalized value along channel axis
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5) # avoid div by zero
        if not self.weight_norm:
            class_mat_norm = torch.norm(self.class_mat.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.class_mat.weight.data)
            self.class_mat.weight.data = self.class_mat.weight.data.div(class_mat_norm + 1e-5)
        cos_dist = self.class_mat(x_normalized)
        if self.training:
            scores = self.train_scale_factor * cos_dist
        else:
            scores = self.val_scale_factor * cos_dist

        return scores

class seg_cos(nn.Module):
    # Treat segmentation as a classification problem for every spatial pixel
    def __init__(self, cfg, feature_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = feature_shape[1]

        self.pixel_classifier = pixel_classifier(cfg, feature_shape, self.in_channels, self.num_classes)
    
    def forward(self, x, size_ = None):
        x = self.pixel_classifier(x)
        x = F.interpolate(x, size = size_, mode = 'bilinear')
        return x
