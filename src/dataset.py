import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def get_patch_label(row):
    if row.get("NC", 0) == 1:
        return 0
    if row.get("G3", 0) == 1:
        return 1
    if row.get("G4", 0) == 1 or row.get("G4C", 0) == 1:
        return 2
    if row.get("G5", 0) == 1:
        return 3
    raise ValueError("Invalid Gleason label row: NC/G3/G4/G4C/G5 are all zero.")


def load_partition(excel_path, images_root, masks_root):
    df = pd.read_excel(excel_path)

    image_paths = []
    mask_paths = []
    labels = []

    for _, row in df.iterrows():
        name = str(row["image_name"])
        if not name.endswith(".jpg"):
            name = name + ".jpg"

        image_paths.append(os.path.join(images_root, name))
        mask_paths.append(os.path.join(masks_root, name))
        labels.append(get_patch_label(row))

    return image_paths, mask_paths, labels


class SICAPv2Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, labels, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Load mask
        mask_raw = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Mask not found: {self.mask_paths[idx]}")
        mask_class = mask_raw.astype(np.uint8) 

        mask_class = (mask_raw > 50).astype(np.uint8) 

        # mask_binary = (mask_raw > 50).astype(np.uint8)

        # # Patch-level Gleason label (0–3)
        # patch_label = self.labels[idx] if self.labels is not None else 0

        # # Attempt: binary mask to multi-class: background=0, tissue=patch_label
        # mask_class = mask_binary * patch_label
     


        # Augments
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask_class)
            image = transformed["image"]
            mask_class = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        if not isinstance(mask_class, torch.Tensor):
            mask_class = torch.from_numpy(mask_class).long()

        return image, mask_class.long()