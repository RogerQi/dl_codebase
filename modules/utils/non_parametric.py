import numpy as np
import cv2
import torch
import torch.nn.functional as F

def masked_average_pooling(mask_b1hw, feature_bchw, normalization):
    '''
    Params
        - mask_b1hw: a binary mask whose element-wise value is either 0 or 1
        - feature_bchw: feature map obtained from the backbone
    
    Return: Mask-average-pooled vector of shape 1 x C
    '''
    if len(mask_b1hw.shape) == 3:
        mask_b1hw = mask_b1hw.view((mask_b1hw.shape[0], 1, mask_b1hw.shape[1], mask_b1hw.shape[2]))

    # Assert remove mask is not in mask provided
    assert -1 not in mask_b1hw

    # Spatial resolution mismatched. Interpolate feature to match mask size
    if mask_b1hw.shape[-2:] != feature_bchw.shape[-2:]:
        feature_bchw = F.interpolate(feature_bchw, size=mask_b1hw.shape[-2:], mode='bilinear')
    
    if normalization:
        feature_norm = torch.norm(feature_bchw, p=2, dim=1).unsqueeze(1).expand_as(feature_bchw)
        feature_bchw = feature_bchw.div(feature_norm + 1e-5) # avoid div by zero

    batch_pooled_vec = torch.sum(feature_bchw * mask_b1hw, dim = (2, 3)) / (mask_b1hw.sum(dim = (2, 3)) + 1e-5) # B x C
    return torch.mean(batch_pooled_vec, dim=0)

def crop_partial_img(img_chw, mask_hw, cls_id=1):
    binary_mask_hw = (mask_hw == cls_id)
    binary_mask_hw_np = binary_mask_hw.numpy().astype(np.uint8)
    # RETR_EXTERNAL to keep online the outer contour
    contours, _ = cv2.findContours(binary_mask_hw_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crop annotated objects off the image
    # Compute a minimum rectangle containing the object
    assert len(contours) != 0
    cnt = contours[0]
    x_min = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
    x_max = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
    y_min = tuple(cnt[cnt[:,:,1].argmin()][0])[1]
    y_max = tuple(cnt[cnt[:,:,1].argmax()][0])[1]
    for cnt in contours:
        x_min = min(x_min, tuple(cnt[cnt[:,:,0].argmin()][0])[0])
        x_max = max(x_max, tuple(cnt[cnt[:,:,0].argmax()][0])[0])
        y_min = min(y_min, tuple(cnt[cnt[:,:,1].argmin()][0])[1])
        y_max = max(y_max, tuple(cnt[cnt[:,:,1].argmax()][0])[1])
    # Index of max bounding rect are inclusive so need 1 offset
    x_max += 1
    y_max += 1
    # mask_roi is a boolean arrays
    mask_roi = binary_mask_hw[y_min:y_max,x_min:x_max]
    img_roi = img_chw[:,y_min:y_max,x_min:x_max]
    return (img_roi, mask_roi)

def semantic_seg_CRF(pred_bhw):
    raise NotImplementedError
