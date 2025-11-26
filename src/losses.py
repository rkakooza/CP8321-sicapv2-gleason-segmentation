import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coefficient(logits, targets, num_classes, smooth=1e-5):
    probs = F.softmax(logits, dim=1)  # (B, C, H, W)
    targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    dims = (0, 2, 3) 
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality = torch.sum(probs + targets_one_hot, dims)

    dice_per_class = (2. * intersection + smooth) / (cardinality + smooth)
    return dice_per_class.mean()


def dice_loss(logits, targets, num_classes, smooth=1e-5):
    dice = dice_coefficient(logits, targets, num_classes, smooth)
    return 1.0 - dice

#Combined CrossEntropy + Dice loss.
class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        d_loss = dice_loss(logits, targets, self.num_classes)
        return self.ce_weight * ce_loss + self.dice_weight * d_loss
