"""Cache official Waterbirds images using a pinned, frozen ViT-B/16."""

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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(root: Path) -> list[dict]:
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
    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = root, rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.root / self.rows[index]["image"]) as image:
            return self.transform(image.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    config.update(dataset_root=str(args.dataset_root.resolve()), root_log_dir=str(args.output_dir / "wl"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
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
    model.heads = torch.nn.Identity()
    rows = load_rows(Path(str(hp["dataset_root"])))
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
