from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from mc_dropout.config import load_config, Config
from mc_dropout.dataset import get_dataloaders
from mc_dropout.model import CNNModel


def train_model(config: Config | None = None) -> CNNModel:
    if config is None:
        config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available — training on CPU (this will be slow)")

    train_loader, test_loader = get_dataloaders(config)
    model = CNNModel(
        dropout_rate=config.model.dropout_rate,
        image_size=config.dataset.image_size,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(config.training.epochs):
        model.train()
        running_loss = correct = total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predicted = (outputs.view(-1) > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(100.0 * correct / total)

        model.eval()
        t_loss = t_correct = t_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                t_loss += criterion(outputs.view(-1), labels).item()
                predicted = (outputs.view(-1) > 0.5).float()
                t_correct += (predicted == labels).sum().item()
                t_total += labels.size(0)

        test_losses.append(t_loss / len(test_loader))
        test_accs.append(100.0 * t_correct / t_total)

        print(
            f"Epoch [{epoch + 1}/{config.training.epochs}] "
            f"Train Loss: {train_losses[-1]:.4f}  Acc: {train_accs[-1]:.2f}% | "
            f"Test Loss:  {test_losses[-1]:.4f}  Acc: {test_accs[-1]:.2f}%"
        )

    model_path = Path(config.model.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved → {model_path}")

    _save_training_curves(train_losses, test_losses, train_accs, test_accs, model_path.parent)
    return model


def _save_training_curves(
    train_losses: list[float],
    test_losses: list[float],
    train_accs: list[float],
    test_accs: list[float],
    output_dir: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, test_losses, label="Test Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Accuracy")
    ax2.plot(epochs, test_accs, label="Test Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Training curves saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MC Dropout CNN for brain tumor detection")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, help="Override training.epochs")
    parser.add_argument("--lr", type=float, help="Override training.learning_rate")
    parser.add_argument("--data-dir", help="Override dataset.dir")
    parser.add_argument("--model-path", help="Override model.path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.data_dir:
        config.dataset.dir = args.data_dir
    if args.model_path:
        config.model.path = args.model_path

    train_model(config)


if __name__ == "__main__":
    main()
