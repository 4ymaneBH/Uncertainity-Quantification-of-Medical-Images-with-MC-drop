import pytest
from mc_dropout.config import (
    Config, DatasetConfig, ModelConfig, TrainingConfig,
    InferenceConfig, ApiConfig, load_config,
)

def test_load_config_returns_defaults_when_file_missing():
    config = load_config("nonexistent_file_xyz.yaml")
    assert isinstance(config, Config)
    assert config.training.epochs == 5
    assert config.inference.num_mc_samples == 100
    assert config.dataset.image_size == 150

def test_load_config_reads_yaml(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("training:\n  epochs: 99\n")
    config = load_config(str(yaml_file))
    assert config.training.epochs == 99
    assert config.training.learning_rate == 0.001  # default preserved

def test_load_config_partial_section(tmp_path):
    yaml_file = tmp_path / "cfg.yaml"
    yaml_file.write_text("dataset:\n  batch_size: 64\n")
    config = load_config(str(yaml_file))
    assert config.dataset.batch_size == 64
    assert config.dataset.image_size == 150  # default preserved
