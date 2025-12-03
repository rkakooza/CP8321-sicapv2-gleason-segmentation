import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization (commonly used for histopathology)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.2),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3),

        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        A.HueSaturationValue(
            hue_shift_limit=5,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3),

        A.GaussNoise(p=0.2),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

# Validation transformations: no augmentation.
def get_val_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
