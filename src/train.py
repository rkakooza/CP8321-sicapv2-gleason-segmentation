import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .attention_unet import AttentionUNet
from .losses import CombinedLoss
from .metrics import aggregate_metrics

import json
import os


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple Metal (MPS) GPU")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device, num_classes):
    model.eval()
    all_metrics = []

    pbar = tqdm(dataloader, desc="Val", leave=False)
    
    with torch.inference_mode():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)
            preds = torch.argmax(logits, dim=1)

            batch_metrics = aggregate_metrics(preds.cpu(), masks.cpu(), num_classes)
            batch_metrics["loss"] = loss.detach().cpu()
            all_metrics.append(batch_metrics)

    final_metrics = {}
    for key in all_metrics[0].keys():
        vals = []
        for m in all_metrics:
            v = m[key]
            v = v if torch.is_tensor(v) else torch.tensor(v)
            vals.append(v)
        final_metrics[key] = torch.stack(vals).mean()

    return final_metrics


def train(model, train_loader, val_loader, optimizer, loss_fn, device, num_classes, epochs=20):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice_macro": []
    }
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, num_classes)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice (macro): {val_metrics['dice_macro']:.4f}")

        # Log into history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.get("loss", 0.0).item())
        history["val_dice_macro"].append(val_metrics["dice_macro"].item())

    return history


if __name__ == "__main__":
    print("train.py is a module and should be imported and used from run scripts or notebooks.")
