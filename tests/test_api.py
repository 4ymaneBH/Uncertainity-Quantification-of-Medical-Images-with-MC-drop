import io
import base64
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image

from mc_dropout.api.main import create_app
from mc_dropout.config import Config


def _jpeg_bytes(color=(200, 100, 50), size=(100, 100)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def test_index_returns_html():
    app = create_app()
    app.state.model = MagicMock()
    app.state.config = Config()
    app.state.device = __import__("torch").device("cpu")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_predict_returns_503_when_model_not_loaded():
    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)
    img_bytes = _jpeg_bytes()
    response = client.post("/predict", files={"file": ("tumor.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 503


def test_predict_returns_422_for_non_image():
    app = create_app()
    app.state.model = MagicMock()
    app.state.config = Config()
    app.state.device = __import__("torch").device("cpu")
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"not an image", "application/pdf")},
    )
    assert response.status_code == 422


def test_predict_response_schema():
    with patch("mc_dropout.api.routes.mc_predict") as mock_pred:
        mock_pred.return_value = {
            "prediction": "Tumor",
            "mean_probability": 0.82,
            "uncertainty": 0.04,
            "histogram_b64": base64.b64encode(b"fake_png").decode(),
        }
        app = create_app()
        app.state.model = MagicMock()
        app.state.config = Config()
        app.state.device = __import__("torch").device("cpu")
        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = _jpeg_bytes()
        response = client.post(
            "/predict",
            files={"file": ("mri.jpg", img_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Tumor"
    assert "mean_probability" in data
    assert "uncertainty" in data
    assert "histogram_b64" in data


def test_predict_response_includes_analysis_fields():
    """The /predict response must always include the 4 analysis fields (may be None)."""
    with patch("mc_dropout.api.routes.mc_predict") as mock_pred:
        mock_pred.return_value = {
            "prediction": "No Tumor",
            "mean_probability": 0.35,
            "uncertainty": 0.05,
            "histogram_b64": base64.b64encode(b"fake_png").decode(),
            "contour_b64": None,
            "scatter_b64": None,
            "area_px": None,
            "mc_area_samples": None,
        }
        app = create_app()
        app.state.model = MagicMock()
        app.state.config = Config()
        app.state.device = __import__("torch").device("cpu")
        client = TestClient(app, raise_server_exceptions=False)
        img_bytes = _jpeg_bytes()
        response = client.post(
            "/predict",
            files={"file": ("scan.png", img_bytes, "image/png")},
        )
    assert response.status_code == 200
    body = response.json()
    for key in ("contour_b64", "scatter_b64", "area_px", "mc_area_samples"):
        assert key in body, f"Missing field: {key}"
