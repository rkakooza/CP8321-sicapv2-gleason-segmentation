import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import amp

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


def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler, device_type):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        with amp.autocast(device_type=device_type):
            logits = model(images)
            loss = loss_fn(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return running_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device, num_classes, device_type):
    model.eval()
    all_metrics = []

    pbar = tqdm(dataloader, desc="Val", leave=False)
    
    with torch.inference_mode():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            with amp.autocast(device_type=device_type):
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


def train(model, train_loader, val_loader, optimizer, loss_fn, device, num_classes):
    device_type = device.type
    scaler = amp.GradScaler() if device_type == "cuda" else None
    # perform one epoch
    train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler, device_type)
    val_metrics = validate_epoch(model, val_loader, loss_fn, device, num_classes, device_type)
    return train_loss, val_metrics.get("loss", torch.tensor(0.0)).item(), val_metrics["dice_macro"].item()


if __name__ == "__main__":
    print("train.py is a module and should be imported and used from run scripts or notebooks.")
