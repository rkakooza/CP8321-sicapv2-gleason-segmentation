
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


# Utility functions for partition handling
def load_partition_excel(excel_path):
    df = pd.read_excel(excel_path)
    # assume first column contains patch names
    col = df.columns[0]
    names = df[col].astype(str).tolist()

    # ensure .jpg extension
    filenames = []
    for n in names:
        if not n.endswith(".jpg"):
            filenames.append(n + ".jpg")
        else:
            filenames.append(n)
    return filenames


def match_paths(filenames, images_root, masks_root):
    image_paths = [os.path.join(images_root, f) for f in filenames]
    mask_paths = [os.path.join(masks_root, f) for f in filenames]
    return image_paths, mask_paths


def build_dataset_from_partition(excel_path, sicap_root, transform=None):
    filenames = load_partition_excel(excel_path)
    images_root = os.path.join(sicap_root, "images")
    masks_root = os.path.join(sicap_root, "masks")

    image_paths, mask_paths = match_paths(filenames, images_root, masks_root)

    return SICAPv2Dataset(image_paths, mask_paths, transform)

class SICAPv2Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        image_paths: list of full paths to image files
        mask_paths: list of full paths to mask files
        transform: Augmentation pipeline (for later if needed)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # To numpy arrays
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.int64)

        # Map all unexpected mask values (>3) to 0
        mask[mask > 3] = 0

        # Apply transform later (if we decide to implement it)
        if self.transform:
            # transform expects combined dict or custom pipeline
            raise NotImplementedError("Transforms not implemented yet")

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return image, mask