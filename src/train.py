import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

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


def train_epoch(model, dataloader, optimizer, loss_fn, device, use_amp=False, scaler=None):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        # Forward + loss with optional mixed precision
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)

        # Backward + optimizer step (with or without AMP)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device,
    num_classes,
    epochs=20,
    use_amp=False,
    scaler=None,
    checkpoint_path=None,
    resume=False,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice_macro": []
    }
    best_metric = None 

    # Resume Training (if enabled and checkpoint exists): This is for colab if it crushes
    start_epoch = 1
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt.get("history", history)
        best_metric = ckpt.get("best_dice_macro", None)
        start_epoch = ckpt.get("epoch", 0) + 1

        print(f" → Resumed from epoch {start_epoch-1} (best dice: {best_metric:.4f})")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = validate_epoch(model, val_loader, loss_fn, device, num_classes)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice (macro): {val_metrics['dice_macro']:.4f}")

        # Log into history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics.get("loss", 0.0).item())
        history["val_dice_macro"].append(val_metrics["dice_macro"].item())

        #saving best checkpoint based on validation Dice
        if checkpoint_path is not None:
            current_metric = val_metrics["dice_macro"].item()
            if (best_metric is None) or (current_metric > best_metric):
                best_metric = current_metric
                state = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "history": history,
                    "best_dice_macro": best_metric,
                }
                torch.save(state, checkpoint_path)
                print(
                    f"Saved new best checkpoint to {checkpoint_path} "
                    f"(epoch {epoch}, dice_macro={current_metric:.4f})"
                )

    return history


if __name__ == "__main__":
    print("train.py is a module and should be imported and used from run scripts or notebooks.")
