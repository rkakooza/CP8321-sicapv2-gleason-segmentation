import torch
import torch.nn.functional as F


def _one_hot(labels, num_classes):
    one_hot = F.one_hot(labels, num_classes=num_classes)
    return one_hot.permute(0, 3, 1, 2).float() 


def dice_per_class(preds, targets, num_classes, smooth=1e-5):
    preds_one_hot = _one_hot(preds, num_classes)
    targets_one_hot = _one_hot(targets, num_classes)

    dims = (0, 2, 3) 
    intersection = torch.sum(preds_one_hot * targets_one_hot, dims)
    cardinality = torch.sum(preds_one_hot + targets_one_hot, dims)

    dice = (2. * intersection + smooth) / (cardinality + smooth)
    return dice


def iou_per_class(preds, targets, num_classes, smooth=1e-5):
    preds_one_hot = _one_hot(preds, num_classes)
    targets_one_hot = _one_hot(targets, num_classes)

    dims = (0, 2, 3)
    intersection = torch.sum(preds_one_hot * targets_one_hot, dims)
    union = torch.sum(preds_one_hot + targets_one_hot, dims) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou

# this can be misleading for imbalanced data. we'll decide to keep or leave
def pixel_accuracy(preds, targets):
    correct = (preds == targets).float()
    return correct.mean()

# Computes the  per-class precision, recall, and F1.
def precision_recall_f1(preds, targets, num_classes, smooth=1e-5):
    preds_one_hot = _one_hot(preds, num_classes)
    targets_one_hot = _one_hot(targets, num_classes)

    dims = (0, 2, 3)

    tp = torch.sum(preds_one_hot * targets_one_hot, dims)
    fp = torch.sum(preds_one_hot * (1 - targets_one_hot), dims)
    fn = torch.sum((1 - preds_one_hot) * targets_one_hot, dims)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)

    return precision, recall, f1

# All evaluation metrics
def aggregate_metrics(preds, targets, num_classes):
    dices = dice_per_class(preds, targets, num_classes)
    ious = iou_per_class(preds, targets, num_classes)
    precision, recall, f1 = precision_recall_f1(preds, targets, num_classes)

    return {
        'pixel_accuracy': pixel_accuracy(preds, targets),
        'dice_per_class': dices,
        'dice_macro': dices.mean(),
        'iou_per_class': ious,
        'iou_macro': ious.mean(),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1.mean(),
    }
