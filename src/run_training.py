import argparse
import os
import json
import torch
from torch.utils.data import DataLoader

from .dataset import build_dataset_from_partition
from .attention_unet import AttentionUNet
from .losses import CombinedLoss
from .metrics import aggregate_metrics
from .train import train, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention U-Net on SICAPv2")

    parser.add_argument("--fold", type=str, required=True,
                        help="Validation fold: Val1, Val2, Val3, or Val4")

    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, default=4,
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

    # Create output directory for this fold
    fold_dir = os.path.join(args.out_dir, args.fold)
    os.makedirs(fold_dir, exist_ok=True)

    # Excel partition paths
    if args.fold.lower() == "final":
        train_xlsx = os.path.join(args.data_root,"partition","Test","Train.xlsx")
        val_xlsx   = os.path.join(args.data_root,"partition","Test","Test.xlsx")
    else:
        train_xlsx = os.path.join(args.data_root,"partition","Validation",args.fold,"Train.xlsx")
        val_xlsx   = os.path.join(args.data_root,"partition","Validation",args.fold,"Test.xlsx")

    # Build datasets
    train_ds = build_dataset_from_partition(train_xlsx, args.data_root)
    val_ds = build_dataset_from_partition(val_xlsx, args.data_root)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=args.num_workers)

    device = get_device()

    # Model
    model = AttentionUNet(in_channels=3, num_classes=4).to(device)

    # Loss function
    loss_fn = CombinedLoss(num_classes=4)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    history = train(model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    num_classes=4,
                    epochs=args.epochs)

    # Final validation pass for saving metrics
    model.eval()
    all_metrics = []

    with torch.inference_mode():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            m = aggregate_metrics(preds.cpu(), masks.cpu(), num_classes=4)
            all_metrics.append(m)

    # Average metrics across validation samples
    final_metrics = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics]
        final_metrics[key] = (torch.stack(vals).mean().item()
                              if torch.is_tensor(vals[0]) else sum(vals) / len(vals))

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