import albumentations as A
from albumentations.pytorch import ToTensorV2



def get_train_transform():
    return A.Compose([
        A.RandomRotate90(p=0.1),
        A.HorizontalFlip(p=0.1),
        A.VerticalFlip(p=0.1),
        ToTensorV2(),
    ])

# Validation transformations: no augmentation.
def get_val_transform():
    return A.Compose([
        ToTensorV2(),
    ])
