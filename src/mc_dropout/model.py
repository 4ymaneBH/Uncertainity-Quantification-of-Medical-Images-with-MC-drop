import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """3-block CNN with dropout for MC Dropout uncertainty estimation.

    image_size must match the size used during training when loading saved weights.
    Default image_size=150 matches the provided monte_carlo_trained_model.pth.
    """

    def __init__(self, dropout_rate: float = 0.5, image_size: int = 150) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        flat_size = self._compute_flat_size(image_size)
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def _compute_flat_size(self, image_size: int) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            dummy = self.pool(torch.relu(self.conv1(dummy)))
            dummy = self.pool(torch.relu(self.conv2(dummy)))
            dummy = self.pool(torch.relu(self.conv3(dummy)))
            return dummy.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))
