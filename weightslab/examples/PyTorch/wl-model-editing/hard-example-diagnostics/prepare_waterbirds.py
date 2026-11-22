"""Step 1: cache frozen ViT-B/16 features for the official Waterbirds splits.

Install this checkout and extract the Waterbirds archive first (see README.md
beside this script). The dataset root must contain metadata.csv and its image
paths. This script does not download Waterbirds; torchvision downloads the
configured pretrained ViT weights if they are not already cached.

Run from the repository root, using a new output directory:
    export PYTHONPATH=. WL_NO_TELEMETRY=1 WEIGHTSLAB_OPENCODE_AUTOINSTALL=0
    experiment_dir=weightslab/examples/PyTorch/wl-model-editing/hard-example-diagnostics
    python "$experiment_dir/prepare_waterbirds.py" --dataset-root /data/waterbirds --output-dir /tmp/wb-cache

The default experiment.json selects ImageNet weights, preprocessing and device
(CUDA when available, otherwise CPU). features.pt contains one embedding per
metadata row, including its original split and stable image identity;
provenance.json records the weights, preprocessing and metadata hashes. Pass
features.pt to run_waterbirds.py next. No backbone weights are trained here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ViT_B_16_Weights, vit_b_16

import weightslab as wl


def sha256(path: Path) -> str:
    """Hash an artifact without loading it into RAM, including on Python 3.10."""
    with path.open("rb") as stream:
        file_digest = getattr(hashlib, "file_digest", None)
        if file_digest is not None:
            return file_digest(stream, "sha256").hexdigest()
        # hashlib.file_digest was added in 3.11; the package still supports 3.10.
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(root: Path) -> list[dict]:
    """Preserve official splits (0=train, 1=validation, 2=test) and image IDs.

    Groups encode bird label:background; both use 0=land and 1=water. Validate
    identities and paths before extraction so later comparisons align examples.
    """
    rows = []
    with (root / "metadata.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            path = (root / row["img_filename"]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError("Image path is missing or escapes the dataset root")
            label, place, split = int(row["y"]), int(row["place"]), int(row["split"])
            if label not in (0, 1) or place not in (0, 1) or split not in (0, 1, 2):
                raise ValueError("Unexpected Waterbirds label, background or split")
            rows.append({"sample_id": row["img_filename"], "image": row["img_filename"],
                         "label": label, "place": place, "split": split,
                         "group": f"{label}:{place}"})
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate image identity across official splits")
    if {row["split"] for row in rows} != {0, 1, 2}:
        raise ValueError("All three official splits are required")
    return rows


class Images(Dataset):
    """Read RGB images in metadata order using the pretrained weights' transform."""

    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["image"]) as image:
            return self.transform(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"),
                        help="Protocol JSON (default: experiment.json beside this script)")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Extracted Waterbirds directory containing metadata.csv and images")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for features.pt and provenance.json; must not exist")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(dataset_root=str(args.dataset_root.resolve()), root_log_dir=str(args.output_dir / "wl"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Register config before constructing objects, so their settings come from
    # the same live hyperparameter handle used by the training example.
    hp = wl.watch_or_edit(config, flag="hyperparameters", defaults=config)
    torch.set_num_threads(int(hp["num_threads"]))
    device = str(hp["device"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(hp["backbone"]["architecture"]) != "vit_b_16" or not bool(hp["backbone"]["frozen"]):
        raise ValueError("This cache supports a frozen vit_b_16 only")
    weights = ViT_B_16_Weights[str(hp["backbone"]["weights"])]
    model = vit_b_16(weights=weights).eval().to(device)
    model.requires_grad_(False)
    # Remove ImageNet's classifier to expose the 768-D CLS embedding. Every arm
    # reuses these features; only the small downstream head will be editable.
    model.heads = torch.nn.Identity()
    rows = load_rows(Path(str(hp["dataset_root"])))
    # Never shuffle this loader: features[i] must describe rows[i] in all splits.
    loader = DataLoader(Images(args.dataset_root, rows, weights.transforms()), shuffle=False,
                        batch_size=int(hp["feature_extraction"]["batch_size"]),
                        num_workers=int(hp["feature_extraction"]["num_workers"]))
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            chunks.append(model(batch.to(device)).cpu())
            print(f"Embedded {sum(len(chunk) for chunk in chunks)}/{len(rows)}", flush=True)
    features = torch.cat(chunks)
    if features.shape != (len(rows), int(hp["backbone"]["feature_width"])) or not torch.isfinite(features).all():
        raise AssertionError("Feature cache shape or finiteness check failed")
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    # Record artifact identity so attribution can reconstruct and verify the
    # same image-to-feature pipeline rather than silently using other weights.
    provenance = {"backbone": str(weights), "preprocessing": str(weights.transforms()),
                  "weights_sha256": sha256(weight_path),
                  "metadata_sha256": sha256(args.dataset_root / "metadata.csv"),
                  "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
                  "device": device, "rows": len(rows), "config": config}
    provenance["preprocessing_sha256"] = hashlib.sha256(provenance["preprocessing"].encode()).hexdigest()
    torch.save({"features": features, "rows": rows, "provenance": provenance}, args.output_dir / "features.pt")
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Feature cache ready: {args.output_dir / 'features.pt'}", flush=True)
    wl.clear_all()


if __name__ == "__main__":
    main()
