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

    # Convert to lists for JSON compatibility
    dices_list = dices.detach().cpu().tolist()
    ious_list = ious.detach().cpu().tolist()
    precision_list = precision.detach().cpu().tolist()
    recall_list = recall.detach().cpu().tolist()
    f1_list = f1.detach().cpu().tolist()

    return {
        'pixel_accuracy': pixel_accuracy(preds, targets).detach().cpu().item(),
        
        'dice_per_class': dices_list,
        'dice_macro': float(dices.mean().detach().cpu()),

        'iou_per_class': ious_list,
        'iou_macro': float(ious.mean().detach().cpu()),

        'precision': precision_list,
        'recall': recall_list,
        'f1': f1_list,

        'f1_macro': float(f1.mean().detach().cpu()),
    }
