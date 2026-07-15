"""Validate images and check class distribution in the dataset."""

from pathlib import Path

from PIL import Image
from tqdm import tqdm

DATASET_PATH = "dataset/BrainTumor_Dataset"


def validate_images(root: Path, split: str) -> tuple[list, list, list]:
    split_path = root / split
    if not split_path.exists():
        return [], [], []

    good, corrupt, invalid_ext = [], [], []
    valid_exts = {".jpg", ".jpeg", ".png"}

    for img_path in tqdm(
        sorted(split_path.rglob("*")),
        desc=f"Checking {split}",
        unit="img",
    ):
        if img_path.is_dir():
            continue

        if img_path.suffix.lower() not in valid_exts:
            invalid_ext.append(img_path)
            continue

        try:
            with Image.open(img_path) as img:
                img.load()
            good.append(img_path)
        except Exception:
            corrupt.append(img_path)

    return good, corrupt, invalid_ext


def summary(root: Path):
    print(f"{'Split':<8} {'Class':<6} {'Count':<8}")
    print("-" * 22)
    for split in ["train", "val", "test"]:
        split_path = root / split
        if not split_path.exists():
            continue
        for class_dir in sorted(split_path.iterdir()):
            if class_dir.is_dir():
                count = len(list(class_dir.iterdir()))
                print(f"{split:<8} {class_dir.name:<6} {count:<8}")


def main():
    root = Path(DATASET_PATH)
    print(f"Dataset path: {root.resolve()}")
    print(f"Exists: {root.exists()}\n")

    if not root.exists():
        print("Dataset not found. Nothing to clean.")
        return

    print("=== Class Distribution ===")
    summary(root)
    print()

    total_good, total_corrupt, total_invalid = 0, 0, 0
    for split in ["train", "val", "test"]:
        good, corrupt, invalid_ext = validate_images(root, split)
        total_good += len(good)
        total_corrupt += len(corrupt)
        total_invalid += len(invalid_ext)

        if corrupt:
            print(f"\n  Corrupt images in {split}:")
            for p in corrupt:
                print(f"     {p}")
        if invalid_ext:
            print(f"\n  Invalid extensions in {split}:")
            for p in invalid_ext:
                print(f"     {p}")

        print(f"  {split}: {len(good)} good, {len(corrupt)} corrupt, {len(invalid_ext)} invalid")

    print(f"\n=== Summary ===")
    print(f"Total good:   {total_good}")
    print(f"Total corrupt: {total_corrupt}")
    print(f"Total invalid: {total_invalid}")

    if total_corrupt == 0 and total_invalid == 0:
        print("\nDataset looks clean. Ready to train.")
    else:
        print("\nIssues found. Review the list above.")


if __name__ == "__main__":
    main()
