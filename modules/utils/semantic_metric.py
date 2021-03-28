import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

def compute_pixel_acc(pred, label, fg_only=True):
    '''
    pred: BHW
    label: BHW
    '''
    assert pred.shape == label.shape
    if fg_only:
        valid = (label >= 0)
        acc_sum = (valid * (pred == label)).sum()
        valid_sum = valid.sum()
        acc = float(acc_sum) / (valid_sum + 1e-10)
        return acc, valid_sum
    else:
        acc_sum = (pred == label).sum()
        acc = float(acc_sum) / (np.prod(pred.shape))
        return acc, 0

def compute_iou(pred_map, label_map, num_classes, fg_only=True):
    pred_map = np.asarray(pred_map).copy()
    label_map = np.asarray(label_map).copy()

    assert pred_map.shape == label_map.shape

    # When computing intersection, all pixels that are not
    # in the intersection are masked with zeros.
    # So we add 1 to the existing mask so that background pixels can be computed
    pred_map += 1
    label_map += 1

    # Compute area intersection:
    intersection = pred_map * (pred_map == label_map)
    (area_intersection, _) = np.histogram(
        intersection, bins=num_classes, range=(1, num_classes))

    # Compute area union:
    (area_pred, _) = np.histogram(pred_map, bins=num_classes, range=(1, num_classes))
    (area_lab, _) = np.histogram(label_map, bins=num_classes, range=(1, num_classes))
    area_union = area_pred + area_lab - area_intersection

    if fg_only:
        # Remove first bg channel
        return np.sum(area_intersection[1:]) / (np.sum(area_union[1:]) + 1e-10)
    else:
        return np.sum(area_intersection) / np.sum(area_union)
