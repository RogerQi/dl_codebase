import torch
import numpy as np
import matplotlib.pyplot as plt

# A generalized imshow helper function which supports displaying (CxHxW) tensor
def generalized_imshow(cfg, arr):
    if isinstance(arr, torch.Tensor) and arr.shape[0] == 3:
        ori_rgb_np = np.array(arr.permute((1, 2, 0)).cpu())
        if 'normalize' in cfg.DATASET.TRANSFORM.TEST.transforms:
            rgb_mean = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.mean
            rgb_sd = cfg.DATASET.TRANSFORM.TEST.TRANSFORMS_DETAILS.NORMALIZE.sd
            ori_rgb_np = (ori_rgb_np * rgb_sd) + rgb_mean
        assert ori_rgb_np.max() <= 1.1, "Max is {}".format(ori_rgb_np.max())
        ori_rgb_np[ori_rgb_np >= 1] = 1
        arr = (ori_rgb_np * 255).astype(np.uint8)
    plt.imshow(arr)
    plt.show()