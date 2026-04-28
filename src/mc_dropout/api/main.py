from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI

from mc_dropout.config import load_config
from mc_dropout.model import CNNModel


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(config.model.path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"Run `mc-train` first to generate the checkpoint."
        )

    model = CNNModel(
        dropout_rate=config.model.dropout_rate,
        image_size=config.dataset.image_size,
    ).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    print(f"Model loaded from {model_path} on {device}")

    app.state.model = model
    app.state.config = config
    app.state.device = device

    yield

    del app.state.model


def create_app() -> FastAPI:
    app = FastAPI(
        title="MC Dropout UQ — Brain Tumor Detection",
        description="Upload a brain MRI image to receive a tumor prediction with uncertainty score.",
        version="0.1.0",
        lifespan=lifespan,
    )
    from mc_dropout.api.routes import router
    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn
    config = load_config()
    uvicorn.run(
        "mc_dropout.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
