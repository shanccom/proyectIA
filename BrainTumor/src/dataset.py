"""
Data-loading pipeline.

Expects a pre-split dataset at ``DATASET_PATH`` with the structure::

    DATASET_PATH/
    ├── train/  {class_0/, class_1/}
    ├── val/    {class_0/, class_1/}
    └── test/   {class_0/, class_1/}
"""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from config import IMAGE_SIZE, BATCH_SIZE, DATASET_PATH


def _train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
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


def get_dataloaders():
    """
    Build train, validation and test DataLoaders from the pre-split dataset.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
    """
    root = Path(DATASET_PATH)

    train_dataset = ImageFolder(root=str(root / "train"),
                                transform=_train_transform())
    val_dataset = ImageFolder(root=str(root / "val"),
                              transform=_eval_transform())
    test_dataset = ImageFolder(root=str(root / "test"),
                               transform=_eval_transform())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader
