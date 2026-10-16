"""Exercise a paired WeightsLab edit workflow on synthetic vectors, not ViT."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

import weightslab as wl


def fixture(hp, split: str):
    generator = torch.Generator().manual_seed(int(hp[f"{split}_data_seed"]))
    count = int(hp[f"{split}_samples"])
    labels = torch.arange(count) % int(hp["num_classes"])
    rare = torch.arange(count) < round(count * float(hp[f"{split}_rare_fraction"]))
    features = torch.randn(count, int(hp["feature_width"]), generator=generator) * float(hp["feature_noise"])
    sign = labels.float() * 2 - 1
    features[:, 0] += sign
    features[:, 1] += sign * torch.where(rare, -1, 1) * float(hp["spurious_strength"])
    return features.to(hp["device"]), labels.to(hp["device"]), rare.to(hp["device"])


def register(config: dict, output: Path, phase: str, checkpoint=None):
    wl.clear_all()
    phase_config = copy.deepcopy(config)
    phase_config["root_log_dir"] = str(output / phase)
    hp = wl.watch_or_edit(phase_config, flag="hyperparameters", defaults=phase_config)
    torch.set_num_threads(int(hp["num_threads"]))
    torch.manual_seed(int(hp["seed"]))
    raw = nn.Sequential(
        nn.Linear(int(hp["feature_width"]), int(hp["hidden_width"])),
        nn.LeakyReLU(negative_slope=float(hp["negative_slope"])),
        nn.Linear(int(hp["hidden_width"]), int(hp["num_classes"])),
    )
    raw.task_type = "classification"
    if checkpoint is not None:
        raw[0].load_state_dict(checkpoint["hidden"])
        raw[2].load_state_dict(checkpoint["classifier"])
    model = wl.watch_or_edit(
        raw, flag="model", device=hp["device"],
        dummy_input=torch.zeros(1, int(hp["feature_width"]), device=hp["device"]),
        compute_dependencies=True, forced_model_wrapping=True,
        skip_previous_auto_load=True,
    )
    optimizer = wl.watch_or_edit(
        torch.optim.SGD(model.parameters(), lr=float(hp["optimizer"]["lr"])), flag="optimizer"
    )
    wl.start_training()
    return hp, model, optimizer


def linear_layers(model):
    layers = [layer for layer in model.get_model_graph()["layers"] if layer["type"] == "Linear"]
    if len(layers) != 2:
        raise AssertionError("Expected exactly two Linear layers in the smoke head")
    hidden = next(layer for layer in layers if layer["name"] == "0")
    classifier = next(layer for layer in layers if layer["name"] == "2")
    return hidden, classifier


def evaluate(model, data, batch_size: int) -> dict:
    features, labels, rare = data
    model.eval()
    chunks = []
    with wl.guard_testing_context, torch.no_grad():
        for start in range(0, len(features), batch_size):
            chunks.append(model(features[start:start + batch_size]).detach().cpu())
    logits = torch.cat(chunks)
    if len(logits) != len(labels) or not torch.isfinite(logits).all():
        raise AssertionError("Missing or non-finite evaluation predictions")
    labels, rare = labels.cpu(), rare.cpu()
    predictions = logits.argmax(dim=1)
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    probabilities = logits.softmax(dim=1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], float("-inf"))
    margins = logits.gather(1, labels[:, None]).squeeze(1) - other.max(dim=1).values
    metrics = {}
    for name, mask in (("ordinary", ~rare), ("rare", rare)):
        count = int(mask.sum())
        metrics[name] = {
            "count": count,
            "accuracy": float((predictions[mask] == labels[mask]).float().mean()) if count else None,
            "loss": float(losses[mask].mean()) if count else None,
        }
    return {
        "model_age": model.get_age(),
        "metrics": metrics,
        "cases": [
            {"sample_id": f"synthetic-test-{index}", "label": int(labels[index]),
             "group": "rare" if bool(rare[index]) else "ordinary",
             "prediction": int(predictions[index]), "logits": logits[index].tolist(),
             "confidence": float(probabilities[index].max()), "loss": float(losses[index]),
             "true_label_margin": float(margins[index])}
            for index in range(len(labels))
        ],
    }


def train(model, optimizer, data, hp, steps: int) -> list[dict]:
    features, labels, _ = data
    model.train()
    initial_age = model.get_age()
    history = []
    # Tracker is installed after any edit so its hooks address the new tensors.
    tracker = wl.track_model_signals(model, every_n_steps=int(hp["model_signals_every_n_steps"]))
    try:
        while model.get_age() - initial_age < steps:
            offset = model.get_age() - initial_age
            batch_size = int(hp["data"]["train_loader"]["batch_size"])
            indices = (torch.arange(batch_size, device=features.device) + offset * batch_size) % len(features)
            complete = False
            with wl.guard_training_context:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(features[indices]), labels[indices])
                loss.backward()
                optimizer.step()
                complete = True
            # The training guard may suppress errors: a skipped step is a failure.
            if not complete or not math.isfinite(float(loss.detach())):
                raise AssertionError("Training step failed or produced non-finite loss")
            history.append({"model_age": model.get_age(), "loss": float(loss.detach())})
    finally:
        tracker.remove()
    return history


def run(config: dict, output: Path) -> dict:
    hp, model, optimizer = register(config, output, "baseline")
    train_data, test_data = fixture(hp, "train"), fixture(hp, "test")
    train(model, optimizer, train_data, hp, int(hp["baseline_training_steps"]))
    baseline = evaluate(model, test_data, int(hp["data"]["test_loader"]["batch_size"]))
    hidden, classifier = linear_layers(model)
    checkpoint = {
        # Fork learned parameters only; WL's per-dataset tracker buffers are
        # runtime instrumentation and do not belong in a fresh nn.Linear.
        "hidden": {name: value.detach().cpu().clone() for name, value in
                   model.get_layer_by_id(hidden["id"]).named_parameters(recurse=False)},
        "classifier": {name: value.detach().cpu().clone() for name, value in
                       model.get_layer_by_id(classifier["id"]).named_parameters(recurse=False)},
        "parent_model_age": model.get_age(),
    }
    checkpoint_path = output / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    runs = {}
    for arm in ("continue", "widen"):
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        hp, model, optimizer = register(config, output, arm, restored)
        test_batch_size = int(hp["data"]["test_loader"]["batch_size"])
        before = evaluate(model, test_data, test_batch_size)
        if before["cases"] != baseline["cases"]:
            raise AssertionError("Branches must start from identical checkpoint predictions")
        graph_before = model.get_model_graph()
        hidden, classifier = linear_layers(model)
        events = []
        if arm == "widen":
            model.add_neurons(hidden["id"], count=int(hp["add_neurons"]))
            hidden_after = model.get_layer_info(hidden["id"], include_neurons=False)
            classifier_after = model.get_layer_info(classifier["id"], include_neurons=False)
            expected_width = int(hp["hidden_width"]) + int(hp["add_neurons"])
            if hidden_after["output_neurons"] != expected_width or classifier_after["input_neurons"] != expected_width:
                raise AssertionError("Head widening did not propagate to the classifier")
            events.append({
                "operation": "add_neurons", "layer_path": hidden["name"],
                "live_layer_id": hidden["id"], "count": int(hp["add_neurons"]),
                "before": hidden, "after": hidden_after,
                "optimizer_policy": "fresh_SGD_no_momentum_in_both_arms",
                "reason": "Synthetic plumbing check; no performance hypothesis tested",
            })
        optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if optimizer_ids != {id(p) for p in model.parameters()}:
            raise AssertionError("Optimizer points at stale or missing model parameters")
        immediate = evaluate(model, test_data, test_batch_size)
        history = train(model, optimizer, train_data, hp, int(hp["training_steps_to_do"]))
        after = evaluate(model, test_data, test_batch_size)
        runs[arm] = {
            "parent_checkpoint_sha256": checkpoint_hash,
            "parent_model_age": checkpoint["parent_model_age"],
            "branch_step_origin": 0,
            "before": before, "immediately_after_edit": immediate, "after": after,
            "graph_before": graph_before, "graph_after": model.get_model_graph(),
            "interventions": events, "training_history": history,
            "optimizer_binding_checked": True,
        }
    control_cases = {case["sample_id"]: case for case in runs["continue"]["after"]["cases"]}
    corrected, regressed = [], []
    for case in runs["widen"]["after"]["cases"]:
        control = control_cases[case["sample_id"]]
        was_correct = control["prediction"] == control["label"]
        is_correct = case["prediction"] == case["label"]
        if is_correct and not was_correct:
            corrected.append(case["sample_id"])
        elif was_correct and not is_correct:
            regressed.append(case["sample_id"])
    return {
        "schema_version": 0, "artifact_type": "synthetic_smoke_report",
        "status": "plumbing_passed", "created_at": datetime.now(timezone.utc).isoformat(),
        "claim": "Synthetic CPU workflow only; no ViT, real-data, or improvement claim",
        "config": config, "torch_version": torch.__version__,
        "checkpoint_sha256": checkpoint_hash, "baseline": baseline, "runs": runs,
        "comparison": {"corrected_ids": corrected, "regressed_ids": regressed},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("smoke_config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["num_classes"] != 2 or config["feature_width"] < 2:
        raise ValueError("The synthetic fixture requires two classes and at least two features")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        report = run(config, args.output_dir)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Synthetic workflow passed: {args.output_dir / 'report.json'}")
    finally:
        wl.clear_all()


if __name__ == "__main__":
    main()
