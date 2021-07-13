import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class pixel_classifier(nn.Module):
    def __init__(self, cfg, feature_shape, in_channel, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.class_mat = nn.Conv2d(in_channel, self.num_classes, 1, bias = False)
        self.scale_factor = 50
    
    def forward(self, x, scale_factor=None):
        '''
        x: (B, in_channel, H, W)
        '''
        # x_norm: (B, in_channel, H, W) where x_norm[i, :, H, W] is the norm of
        # x[i, :, H, W]. That is, x/x_norm yields normalized value along channel axis
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5) # avoid div by zero
        class_mat_norm = torch.norm(self.class_mat.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.class_mat.weight.data)
        self.class_mat.weight.data = self.class_mat.weight.data.div(class_mat_norm + 1e-5)
        cos_dist = self.class_mat(x_normalized)
        if scale_factor is not None:
            return scale_factor * cos_dist
        else:
            return self.scale_factor * cos_dist

class fcn32s_cos(nn.Module):

    def __init__(self, cfg, feature_shape, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # fc6
        self.fc6 = nn.Conv2d(feature_shape[1], 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)

        self.pixel_classifier = pixel_classifier(cfg, feature_shape, 4096, self.num_classes)

    def forward(self, x, size_ = None):
        x = self.logit_forward(x)
        x = self.cosine_forward(x, size_)

        return x
    
    def replace_binary_head(self, cfg, feature_shape):
        # Use for binary classification
        self.pixel_classifier = pixel_classifier(cfg, feature_shape, 4096, 2)
    
    def logit_forward(self, x):
        x = self.relu6(self.fc6(x))

        x = self.fc7(x)

        return x
    
    def cosine_forward(self, x, size_ = None):
        x = self.pixel_classifier(x)

        # Origianl FCN paper uses transpose Conv to upscale the image
        # However, experiments showed that it doesn't work well with cosine similarity.
        # ... So we changed it to bilinear interpolation

        x = F.interpolate(x, size = size_, mode = 'bilinear', align_corners=False)

        return x
