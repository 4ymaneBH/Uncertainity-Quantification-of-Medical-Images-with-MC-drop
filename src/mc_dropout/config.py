from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class DatasetConfig:
    dir: str = "./data/Brain_Tumor_Detection"
    test_split: float = 0.2
    image_size: int = 150
    batch_size: int = 32


@dataclass
class ModelConfig:
    path: str = "./models/monte_carlo_trained_model.pth"
    dropout_rate: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 0.001


@dataclass
class InferenceConfig:
    num_mc_samples: int = 100
    threshold: float = 0.5


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    api: ApiConfig = field(default_factory=ApiConfig)


def load_config(path: str = "config.yaml") -> Config:
    config_path = Path(path)
    if not config_path.exists():
        return Config()

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(
        dataset=DatasetConfig(**data.get("dataset") or {}),
        model=ModelConfig(**data.get("model") or {}),
        training=TrainingConfig(**data.get("training") or {}),
        inference=InferenceConfig(**data.get("inference") or {}),
        api=ApiConfig(**data.get("api") or {}),
    )
