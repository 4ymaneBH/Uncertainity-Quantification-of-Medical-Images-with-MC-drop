from __future__ import annotations
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mc_dropout.config import Config

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader]:
    dataset_dir = Path(config.dataset.dir)
    categories = {"yes": 1, "no": 0}

    image_paths: List[str] = []
    labels: List[int] = []

    for category, label in categories.items():
        category_path = dataset_dir / category
        if not category_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {category_path}\n"
                f"Expected sub-folders 'yes' and 'no' inside: {dataset_dir}"
            )
        for img_name in os.listdir(category_path):
            image_paths.append(str(category_path / img_name))
            labels.append(label)

    size = config.dataset.image_size
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels,
        test_size=config.dataset.test_split,
        random_state=42,
    )

    train_ds = BrainTumorDataset(X_train, y_train, transform)
    test_ds = BrainTumorDataset(X_test, y_test, transform)

    return (
        DataLoader(train_ds, batch_size=config.dataset.batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=config.dataset.batch_size, shuffle=False),
    )
