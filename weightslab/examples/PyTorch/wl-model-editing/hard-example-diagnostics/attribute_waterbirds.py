"""Measured class-conditioned input attribution for locked validation cases."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from prepare_waterbirds import sha256
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16

import weightslab as wl


def integrated_gradients(model, image, target, steps, batch_size):
    """Gauss-Legendre IG from zero normalized input; return signed completeness error."""
    points, weights = np.polynomial.legendre.leggauss(steps)
    points = torch.tensor((points + 1) / 2, device=image.device, dtype=image.dtype)
    weights = torch.tensor(weights / 2, device=image.device, dtype=image.dtype)
    accumulated = torch.zeros_like(image)
    for start in range(0, steps, batch_size):
        scaled = (points[start:start + batch_size, None, None, None] * image).detach().requires_grad_(True)
        output = model(scaled)[:, target]
        gradient = torch.autograd.grad(output.sum(), scaled)[0]
        accumulated += (gradient * weights[start:start + batch_size, None, None, None]).sum(0, keepdim=True)
    attribution = image * accumulated
    with torch.no_grad():
        difference = float(model(image)[0, target] - model(torch.zeros_like(image))[0, target])
    delta = float(attribution.sum()) - difference
    if not torch.isfinite(attribution).all() or not np.isfinite(delta):
        raise AssertionError("Nonfinite attribution")
    return attribution.detach(), delta, difference


def image_uri(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    args = parser.parse_args()
    report = json.loads((args.run_dir / "report.json").read_text())
    config = json.loads(args.config.read_text())
    if config["backbone"] != report["protocol"]["config"]["backbone"]:
        raise ValueError("Attribution backbone must match the recorded experiment")
    config.update(root_log_dir=str(args.run_dir / "attribution-ledger"), dataset_root=str(args.dataset_root))
    hp = wl.watch_or_edit(config, flag="hyperparameters", defaults=config)
    torch.set_num_threads(int(hp["num_threads"]))
    device = str(hp["device"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ViT_B_16_Weights[str(hp["backbone"]["weights"])]
    backbone = vit_b_16(weights=weights).eval().to(device)
    backbone.heads = nn.Identity()
    backbone.requires_grad_(False)
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]
    if sha256(weight_path) != report["feature_provenance"]["weights_sha256"]:
        raise ValueError("Attribution backbone weights differ from the feature cache")
    seed = report["protocol"]["config"]["seeds"][0]
    candidates = report["runs"][f"{seed}-continue"]["demo_validation"]
    selected = []
    for group in report["group_names"]:
        selected.extend([case for case in candidates if case["group"] == group][:int(hp["attribution_cases_per_group"])])
    names = ["baseline"] + [arm["name"] for arm in report["protocol"]["config"]["arms"]]
    result = {"method": "Integrated Gradients / Gauss-Legendre", "seed": seed,
              "baseline": "zero in normalized image space (ImageNet mean-color image)",
              "target": "official dataset class logit, fixed across all branches; known label issues are not corrected",
              "selection": "first locked validation case per group in metadata order; no test cases",
              "scale": "shared maximum absolute signed channel-summed attribution across all five snapshots per image",
              "protocol_sha256": report["protocol_sha256"],
              "settings": {key: value for key, value in config.items() if key.startswith("attribution_")}, "cases": {}}
    for case in selected:
        with Image.open(Path(str(hp["dataset_root"])) / case["image"]) as image:
            tensor = weights.transforms()(image.convert("RGB")).unsqueeze(0).to(device)
        crop = tensor[0].detach().cpu() * torch.tensor(weights.transforms().std)[:, None, None] + torch.tensor(weights.transforms().mean)[:, None, None]
        crop_array = (crop.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        attribution_maps, entries = {}, {}
        for name in names:
            filename = f"baseline-{seed}.pt" if name == "baseline" else f"{seed}-{name}.pt"
            state = torch.load(args.run_dir / "checkpoints" / filename, map_location="cpu", weights_only=True)
            head = nn.Sequential(nn.Linear(state["0"]["weight"].shape[1], state["0"]["weight"].shape[0]),
                                 nn.LeakyReLU(float(hp["head"]["negative_slope"])),
                                 nn.Linear(state["2"]["weight"].shape[1], state["2"]["weight"].shape[0]))
            for path, parameters in state.items():
                head[int(path)].load_state_dict(parameters)
            model = nn.Sequential(backbone, head.to(device)).eval().requires_grad_(False)
            with torch.no_grad():
                logits = model(tensor)[0].cpu()
            expected = next(item for item in (report["baselines"][str(seed)]["validation"]["cases"] if name == "baseline"
                            else report["runs"][f"{seed}-{name}"]["demo_validation"]) if item["sample_id"] == case["sample_id"])
            if not torch.allclose(logits, torch.tensor(expected["logits"]), atol=1e-4, rtol=1e-4):
                raise AssertionError("Full image model disagrees with cached-feature predictions")
            steps = int(hp["attribution_steps"])
            while True:
                attribution, delta, difference = integrated_gradients(model, tensor, case["label"], steps,
                                                                      int(hp["attribution_batch_size"]))
                tolerance = float(hp["attribution_absolute_tolerance"]) + float(hp["attribution_relative_tolerance"]) * abs(difference)
                converged = abs(delta) <= tolerance
                if converged or steps >= int(hp["attribution_max_steps"]):
                    break
                steps = min(steps * 2, int(hp["attribution_max_steps"]))
            attribution_maps[name] = attribution[0].sum(0).cpu().numpy()
            entries[name] = {"steps": steps, "completeness_delta": delta, "logit_difference": difference,
                             "within_tolerance": converged, "tolerance": tolerance, "prediction": expected["prediction"],
                             "confidence": expected["confidence"], "true_label_margin": expected["true_label_margin"]}
            print(f"Attributed {case['sample_id']} / {name}: steps={steps}, delta={delta:.5f}, passed={converged}", flush=True)
        scale = max(float(np.abs(values).max()) for values in attribution_maps.values())
        for name, values in attribution_maps.items():
            strength = np.abs(values) / max(scale, np.finfo(float).eps)
            color = np.where((values >= 0)[..., None], np.array([240, 84, 44]), np.array([43, 106, 216]))
            overlay = crop_array * (1 - strength[..., None]) + color * strength[..., None]
            entries[name]["overlay"] = image_uri(Image.fromarray(overlay.astype("uint8")))
        result["cases"][case["sample_id"]] = {"group": case["group"], "label": case["label"],
                                               "crop": image_uri(Image.fromarray(crop_array)),
                                               "shared_scale": scale, "snapshots": entries}
        (args.run_dir / "attribution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    wl.clear_all()


if __name__ == "__main__":
    main()
