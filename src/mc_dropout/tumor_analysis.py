from __future__ import annotations
import base64
import io
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mc_dropout.model import CNNModel

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_GRADCAM_PERCENTILE = 75
_MC_AREA_SAMPLES    = 50_000
_CONTOUR_STROKE     = 2
_ZOOM_PADDING       = 20
_DOT_RADIUS         = 1
_MAX_SCATTER_DOTS   = 5_000


def gradcam_mask(
    image: Image.Image,
    model: CNNModel,
    device: torch.device,
    image_size: int = 150,
) -> np.ndarray:
    """Return a binary uint8 mask (0/255) of shape (image_size, image_size).

    Uses GradCAM on the last conv layer (conv3) to find the attention region,
    then refines with Otsu thresholding inside that region.
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0).to(device)

    # Storage for hooks
    activations: list[torch.Tensor] = []
    gradients:   list[torch.Tensor] = []

    fwd_hook = model.conv3.register_forward_hook(
        lambda _m, _i, out: activations.append(out)
    )
    bwd_hook = model.conv3.register_full_backward_hook(
        lambda _m, _gi, go: gradients.append(go[0])
    )

    try:
        model.eval()
        model.zero_grad()
        out = model(tensor)
        out[0, 0].backward()
    finally:
        fwd_hook.remove()
        bwd_hook.remove()

    # GradCAM: weight channels by global-average-pooled gradient
    act  = activations[0].squeeze(0)          # (C, H, W)
    grad = gradients[0].squeeze(0)            # (C, H, W)
    weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
    cam = torch.relu((act * weights).sum(dim=0))    # (H, W)

    cam_np = cam.detach().cpu().numpy()
    if cam_np.max() > 0:
        cam_np = cam_np / cam_np.max()

    # Upsample to image_size
    cam_resized = cv2.resize(cam_np, (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)

    # Coarse attention mask: top GRADCAM_PERCENTILE activations
    threshold_val = float(np.percentile(cam_resized, _GRADCAM_PERCENTILE))
    attention_mask = (cam_resized >= threshold_val).astype(np.uint8) * 255

    if attention_mask.sum() == 0:
        # Fallback: use full image
        attention_mask = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Grayscale of the resized original image
    gray = np.array(image.resize((image_size, image_size)).convert("L"))

    # Otsu threshold inside attention region
    masked_gray = cv2.bitwise_and(gray, gray, mask=attention_mask)
    _, binary = cv2.threshold(masked_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary
