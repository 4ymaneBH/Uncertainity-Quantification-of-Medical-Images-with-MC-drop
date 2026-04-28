import torch
from mc_dropout.model import CNNModel


def test_forward_output_shape():
    model = CNNModel()
    x = torch.zeros(2, 3, 150, 150)
    out = model(x)
    assert out.shape == (2, 1), f"Expected (2,1), got {out.shape}"


def test_output_in_probability_range():
    model = CNNModel()
    x = torch.rand(4, 3, 150, 150)
    out = model(x)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_dropout_active_in_train_mode():
    torch.manual_seed(0)
    model = CNNModel(dropout_rate=0.9)
    model.train()
    x = torch.ones(1, 3, 150, 150)
    out1 = model(x).item()
    out2 = model(x).item()
    assert out1 != out2, "Outputs should differ under active dropout"


def test_dropout_inactive_in_eval_mode():
    torch.manual_seed(0)
    model = CNNModel(dropout_rate=0.9)
    model.eval()
    x = torch.ones(1, 3, 150, 150)
    with torch.no_grad():
        out1 = model(x).item()
        out2 = model(x).item()
    assert out1 == out2, "Outputs should be deterministic in eval mode"


def test_custom_image_size():
    model = CNNModel(image_size=224)
    x = torch.zeros(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 1)
