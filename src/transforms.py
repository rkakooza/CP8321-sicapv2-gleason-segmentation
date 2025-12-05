import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet normalization (commonly used for histopathology)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform():
    return A.Compose([
        A.RandomRotate90(p=0.1),
        A.HorizontalFlip(p=0.1),
        A.VerticalFlip(p=0.1),

        A.OneOf([
            A.GaussianBlur(blur_limit=(0, 3), p=1.0),  
            A.MedianBlur(blur_limit=3, p=1.0),        
            A.GaussNoise(var_limit=(0.05, 0.05), p=1.0), 
            A.ColorJitter(
                brightness=(229/255, 281/255),
                contrast=(0.95, 1.1),
                saturation=(0.8, 1.2),
                hue=(-0.04, 0.04),
                p=1.0),
        ], p=0.1),

        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

# Validation transformations: no augmentation.
def get_val_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
