import os
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from mc_dropout.config import Config, DatasetConfig
from mc_dropout.dataset import BrainTumorDataset, get_dataloaders


def make_fake_dataset(root, n_yes=4, n_no=4):
    for category, count in [("yes", n_yes), ("no", n_no)]:
        folder = os.path.join(root, category)
        os.makedirs(folder, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (50, 50), color=(i * 30, i * 20, 100))
            img.save(os.path.join(folder, f"img_{i}.jpg"))


def test_dataset_len(tmp_path):
    make_fake_dataset(str(tmp_path))
    paths = [str(tmp_path / "yes" / f"img_{i}.jpg") for i in range(4)]
    labels = [1] * 4
    ds = BrainTumorDataset(paths, labels)
    assert len(ds) == 4


def test_dataset_returns_image_and_label(tmp_path):
    make_fake_dataset(str(tmp_path))
    paths = [str(tmp_path / "yes" / "img_0.jpg")]
    ds = BrainTumorDataset(paths, [1])
    img, label = ds[0]
    assert label == 1
    assert img is not None


def test_get_dataloaders_splits_correctly(tmp_path):
    make_fake_dataset(str(tmp_path), n_yes=8, n_no=8)
    config = Config(dataset=DatasetConfig(
        dir=str(tmp_path), test_split=0.5, image_size=50, batch_size=4
    ))
    train_loader, test_loader = get_dataloaders(config)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    total = len(train_loader.dataset) + len(test_loader.dataset)
    assert total == 16


def test_get_dataloaders_raises_on_missing_dir():
    config = Config(dataset=DatasetConfig(dir="/nonexistent/path"))
    with pytest.raises(FileNotFoundError, match="yes"):
        get_dataloaders(config)
