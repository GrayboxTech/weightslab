"""Shared utilities for the ViT model-editing experiments."""

from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torchvision.models import ViT_B_16_Weights, vit_b_16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "This experiment supports CPU and CUDA only because ModelInterface "
            "currently normalizes other devices to CPU."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_vit_b_16(
    *,
    image_size: int,
    num_classes: int,
    pretrained: bool,
) -> tuple[torch.nn.Module, int]:
    """Build ViT-B/16, the torchvision ViT-Base architecture with 12 blocks."""
    if pretrained:
        if image_size != 224:
            raise ValueError("Pretrained ViT-B/16 requires --image-size 224.")
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = torch.nn.Linear(model.hidden_dim, num_classes)
    else:
        model = vit_b_16(
            weights=None,
            image_size=image_size,
            num_classes=num_classes,
        )

    block_count = len(model.encoder.layers)
    if block_count != 12:
        raise AssertionError(f"Expected 12 encoder blocks, found {block_count}.")
    return model, block_count


def make_pattern_images(
    *,
    samples: int,
    image_size: int,
    num_classes: int,
    seed: int,
    normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic, learnable image patterns without downloading data."""
    if samples < num_classes:
        raise ValueError("samples must be at least num_classes.")

    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % num_classes
    images = torch.rand(samples, 3, image_size, image_size, generator=generator) * 0.08
    band = max(1, image_size // 4)

    for index, label_tensor in enumerate(labels):
        label = int(label_tensor)
        channel = label % 3
        position = (label // 3) % 4
        if label % 2 == 0:
            start = min(position * band, image_size - band)
            images[index, channel, start : start + band, :] += 0.85
        else:
            start = min(position * band, image_size - band)
            images[index, channel, :, start : start + band] += 0.85

    images.clamp_(0.0, 1.0)
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std
    return images, labels.long()


@torch.no_grad()
def extract_embeddings(
    backbone: torch.nn.Module,
    images: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    backbone.eval()
    outputs = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        outputs.append(backbone(batch).cpu())
    return torch.cat(outputs)


def environment_info(device: torch.device) -> dict[str, Any]:
    import torchvision

    import weightslab

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "weightslab": getattr(weightslab, "__version__", "unknown"),
        "device": str(device),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def optimizer_parameter_ids(optimizer: Any) -> set[int]:
    raw_optimizer = getattr(optimizer, "optimizer", optimizer)
    return {
        id(parameter)
        for group in raw_optimizer.param_groups
        for parameter in group["params"]
    }
