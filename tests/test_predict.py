import base64
import torch
from PIL import Image
from mc_dropout.model import CNNModel
from mc_dropout.predict import mc_predict


def _blank_image(size: int = 150) -> Image.Image:
    return Image.new("RGB", (size, size), color=(128, 128, 128))


def test_mc_predict_returns_required_keys():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    assert set(result.keys()) == {"prediction", "mean_probability", "uncertainty", "histogram_b64"}


def test_mc_predict_prediction_label_is_valid():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    assert result["prediction"] in {"Tumor", "No Tumor"}


def test_mc_predict_probability_in_range():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    assert 0.0 <= result["mean_probability"] <= 1.0
    assert result["uncertainty"] >= 0.0


def test_mc_predict_histogram_is_valid_base64_png():
    model = CNNModel()
    result = mc_predict(_blank_image(), model, num_samples=5, device=torch.device("cpu"))
    raw = base64.b64decode(result["histogram_b64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "histogram_b64 should decode to a PNG"


def test_mc_predict_threshold_respected():
    model = CNNModel()
    result = mc_predict(
        _blank_image(), model, num_samples=5,
        threshold=0.0,
        device=torch.device("cpu"),
    )
    assert result["prediction"] == "Tumor"

    result2 = mc_predict(
        _blank_image(), model, num_samples=5,
        threshold=1.1,
        device=torch.device("cpu"),
    )
    assert result2["prediction"] == "No Tumor"
