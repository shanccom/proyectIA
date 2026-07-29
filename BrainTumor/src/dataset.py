from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms.functional as TF

from config import IMAGE_SIZE, BATCH_SIZE, DATASET_PATH


def _base_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _augmented_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _square_center_crop(img):
    w, h = img.size
    size = min(w, h)
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _normalized_train_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Lambda(_square_center_crop),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Lambda(lambda img: TF.equalize(img)),
        transforms.RandomApply([
            transforms.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor=1.2)),
            transforms.Lambda(lambda img: TF.adjust_brightness(img, brightness_factor=0.8)),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _normalized_eval_transform():
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Lambda(_square_center_crop),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda img: TF.equalize(img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(augment=True, normalized=False):
    root = Path(DATASET_PATH)

    if normalized:
        train_transform = _normalized_train_transform()
        val_transform = _normalized_eval_transform()
        test_transform = _normalized_eval_transform()
    elif augment:
        train_transform = _augmented_transform()
        val_transform = _eval_transform()
        test_transform = _eval_transform()
    else:
        train_transform = _base_transform()
        val_transform = _eval_transform()
        test_transform = _eval_transform()

    train_dataset = ImageFolder(root=str(root / "train"),
                                transform=train_transform)
    val_dataset = ImageFolder(root=str(root / "val"),
                              transform=val_transform)
    test_dataset = ImageFolder(root=str(root / "test"),
                               transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
