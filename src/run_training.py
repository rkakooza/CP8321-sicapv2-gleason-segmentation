import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from PIL import Image

from .dataset import load_partition, SICAPv2Dataset
from .attention_unet import AttentionUNet
from .losses import CombinedLoss
from .metrics import aggregate_metrics
from .train import train, get_device
from .transforms import get_train_transform, get_val_transform


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None

    def step(self, val_loss):
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention U-Net on SICAPv2")

    parser.add_argument("--fold", type=str, required=True,
                        help="Validation fold: Val1, Val2, Val3, or Val4")

    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")

    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of DataLoader workers")

    parser.add_argument("--data_root", type=str, default="data/SICAPv2",
                        help="Root folder of the SICAPv2 dataset")

    parser.add_argument("--out_dir", type=str, default="experiments",
                        help="Where to save results")

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory for a fold
    fold_dir = os.path.join(args.out_dir, args.fold)
    os.makedirs(fold_dir, exist_ok=True)

    # Excel partition paths
    if args.fold.lower() == "final":
        train_xlsx = os.path.join(args.data_root,"partition","Test","Train.xlsx")
        val_xlsx   = os.path.join(args.data_root,"partition","Test","Test.xlsx")
    else:
        train_xlsx = os.path.join(args.data_root,"partition","Validation",args.fold,"Train.xlsx")
        val_xlsx   = os.path.join(args.data_root,"partition","Validation",args.fold,"Test.xlsx")

    images_root = os.path.join(args.data_root, "images")
    masks_root = os.path.join(args.data_root, "masks")

    # Load partitions
    train_list = load_partition(train_xlsx, images_root=images_root, masks_root=masks_root)
    val_list = load_partition(val_xlsx, images_root=images_root, masks_root=masks_root)

    # Build datasets
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_imgs, train_masks, train_labels = train_list
    val_imgs, val_masks, val_labels = val_list

    train_ds = SICAPv2Dataset(train_imgs, train_masks, train_labels, transform=train_transform)
    val_ds = SICAPv2Dataset(val_imgs, val_masks, val_labels, transform=val_transform)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=args.num_workers)

    device = get_device()

    # Model
    model = AttentionUNet(in_channels=3, num_classes=2).to(device)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss() # Another option i tried, produced good Values
    # loss_fn = CombinedLoss(num_classes=2)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    early_stopper = EarlyStopping(patience=5)

    # Best checkpoint tracking
    best_dice = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(fold_dir, "best_model.pth")

    # Training loop with per-epoch validation
    history = {"train_loss": [], "val_loss": [], "val_dice_macro": []}
    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, val_loss, val_dice, stop_flag = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_classes=2,
            scheduler=scheduler,
            early_stopper=early_stopper
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice_macro"].append(val_dice)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice (macro): {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model at epoch {epoch} (dice_macro={val_dice:.4f}). Saved to {best_ckpt_path}")
        if stop_flag:
            print("Early stopping triggered.")
            break

    # Load the best model and compute final validation metrics
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()
    all_metrics = []
    with torch.inference_mode():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            m = aggregate_metrics(preds.cpu(), masks.cpu(), num_classes=2)
            all_metrics.append(m)

    final_metrics = {}
    first = all_metrics[0]

    for key, first_val in first.items():
        vals = [m[key] for m in all_metrics]

        # Per-class metrics are lists ([c0, c1, c2, c3])
        if isinstance(first_val, (list, tuple)):
            t = torch.tensor(vals, dtype=torch.float32)   
            final_metrics[key] = t.mean(dim=0).tolist()  

        # Scalar metrics (pixel_accuracy, dice_macro, iou_macro, f1_macro)
        else:
            t = torch.tensor(vals, dtype=torch.float32)   # shape: [N]
            final_metrics[key] = t.mean().item()          # single scalar

    # Save metrics JSON
    metrics_path = os.path.join(fold_dir, "val_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    history_path = os.path.join(fold_dir,"history.json")
    with open(history_path,"w") as f:
        json.dump(history,f,indent=4)
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()