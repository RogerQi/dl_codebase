import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.weight_norm import WeightNorm

class pixel_classifier(nn.Module):
    def __init__(self, cfg, feature_shape, in_channel, weight_norm = False):
        super().__init__()
        self.num_classes = cfg.num_classes
        self.class_mat = nn.Conv2d(in_channel, self.num_classes, 1, bias = False)
        self.weight_norm = weight_norm
        if weight_norm:
            # conv weight shape: (num_classes, in_channel, 1, 1)
            WeightNorm.apply(self.class_mat, 'weight', dim=0)
        self.scale_factor = 50
    
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
        scores = self.scale_factor * (cos_dist) 

        return scores

class fcn32s_cos(nn.Module):

    def __init__(self, cfg, feature_shape):
        super().__init__()
        self.num_classes = cfg.num_classes
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.pixel_classifier = pixel_classifier(cfg, feature_shape, 4096)

    def forward(self, x, size_ = None):
        x = self.relu6(self.fc6(x))
        x = self.drop6(x)

        x = self.relu7(self.fc7(x))
        x = self.drop7(x)

        x = self.pixel_classifier(x)

        # Origianl FCN paper uses transpose Conv to upscale the image
        # However, experiments showed that it doesn't work well with cosine similarity.
        # ... So we changed it to bilinear interpolation

        x = F.interpolate(x, size = size_, mode = 'bilinear', align_corners=False)

        return x