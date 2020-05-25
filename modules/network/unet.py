import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    '''
    Implementation from Figure 1. in https://arxiv.org/pdf/1505.04597.pdf

    Support for bilinear upsampling. Note that in the original UNet paper,
    deconv was used. But bilinear upsampling seems to be a popular choice.
    '''
    def __init__(self, cfg, bilinear = True):
        super(UNet, self).__init__()
        self.input_channels = cfg.input_dim[0]
        self.num_classes = cfg.num_classes

        # Input Convolution
        self.input_conv = unet_double_cbr(self.input_channels, 64)
        # Start Sampling
        self.down1 = unet_downsample(64, 128)
        self.down2 = unet_downsample(128, 256)
        self.down3 = unet_downsample(256, 512)
        # lower bottle neck
        # Maybe the CBR module in the middle should have 1024 channels?
        self.down4 = unet_downsample(512, 512, 1024)
        # Start Upsampling
        self.up1 = unet_upsample(1024, 256, bilinear)
        self.up2 = unet_upsample(512, 128, bilinear)
        self.up3 = unet_upsample(256, 64, bilinear)
        self.up4 = unet_upsample(128, 64, bilinear)
        # Output Convolution
        self.output_conv = nn.Conv2d(64, self.num_classes, kernel_size = 1)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        # Start upsampling
        # Note that skip connections happen here...
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.output_conv(x)
        return x

class unet_double_cbr(nn.Module):
    '''
    Implementation from Figure 1. in https://arxiv.org/pdf/1505.04597.pdf
    '''
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(unet_double_cbr, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.cbr1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True)
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )
    
    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        return x

class unet_downsample(nn.Module):
    '''
    Implementation from Figure 1. in https://arxiv.org/pdf/1505.04597.pdf
    '''
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super(unet_downsample, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_cbr = unet_double_cbr(in_channels, out_channels, mid_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_cbr(x)
        return x

class unet_upsample(nn.Module):
    '''
    Implementation from Figure 1. in https://arxiv.org/pdf/1505.04597.pdf

    Support for bilinear resizing. Note that in the original UNet paper,
    deconv was used. But bilinear upsampling seems to be a popular choice.
    '''
    def __init__(self, in_channels, out_channels, bilinear = True):
        super(unet_upsample, self).__init__()
        self.bilinear = bilinear

        self.double_cbr = unet_double_cbr(in_channels, out_channels)
        if not self.bilinear:
            # Deconv (or conv transpose) was used in the original UNet paper
            self.upsample_conv = nn.ConvTranspose2d(in_channels , out_channels, kernel_size = 2, stride = 2)


    def forward(self, x, skip_tensor):
        assert x.shape[0] == skip_tensor.shape[0]
        assert x.shape[1] == skip_tensor.shape[1]
        B, C, H, W = skip_tensor.shape
        if self.bilinear:
            x = F.interpolate(x, size = (H, W), mode = 'bilinear')
        else:
            x = self.upsample_conv(x)

        x = torch.cat([skip_tensor, x], dim=1)
        x = self.double_cbr(x)
        return x

if __name__ == '__main__':
    class dummy_cfg(object):
        input_dim = (3, 256, 256)
        num_classes = 10

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    my_net = UNet(dummy_cfg).to(device)
    from torchsummary import summary
    print(summary(my_net, dummy_cfg.input_dim, device = device_str))
