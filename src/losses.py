import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: (B, C, H, W) logits from network
        targets: (B, H, W) integer class labels
        """
        preds = F.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)

        intersection = torch.sum(preds * targets_one_hot, dims)
        union = torch.sum(preds + targets_one_hot, dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice loss is (1 - dice) averaged across classes
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, dice_weight=0.5, ce_weight=0.5):
        super().__init__()

        self.dice = DiceLoss(num_classes=num_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, preds, targets):
        ce = self.ce(preds, targets)
        dice = self.dice(preds, targets)

        return self.ce_weight * ce + self.dice_weight * dice