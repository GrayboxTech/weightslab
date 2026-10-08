"""Run the locked four-arm, three-seed Waterbirds head-editing pilot."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from plan import validate_config
from prepare_waterbirds import sha256
from torch import nn

import weightslab as wl

GROUP_NAMES = {"0:0": "Landbird / land", "0:1": "Landbird / water",
               "1:0": "Waterbird / land", "1:1": "Waterbird / water"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def linear_layers(model):
    return {item["name"]: item for item in model.get_model_graph()["layers"] if item["type"] == "Linear"}


def learned_parameters(model):
    return {path: {name: value.detach().cpu().clone()
                   for name, value in model.get_layer_by_id(info["id"]).named_parameters(recurse=False)}
            for path, info in linear_layers(model).items()}


def register(config, output, run_id, seed, checkpoint=None):
    wl.clear_all()
    local = copy.deepcopy(config)
    local.update(seed=seed, root_log_dir=str(output / "ledger" / run_id))
    hp = wl.watch_or_edit(local, flag="hyperparameters", defaults=local)
    device = str(hp["device"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(hp["num_threads"]))
    torch.manual_seed(int(hp["seed"]))
    raw = nn.Sequential(nn.Linear(int(hp["backbone"]["feature_width"]), int(hp["head"]["hidden_width"])),
                        nn.LeakyReLU(float(hp["head"]["negative_slope"])),
                        nn.Linear(int(hp["head"]["hidden_width"]), int(hp["head"]["classes"])))
    raw.task_type = "classification"
    if checkpoint is not None:
        for path, parameters in checkpoint.items():
            raw[int(path)].load_state_dict(parameters)
    model = wl.watch_or_edit(raw, flag="model", device=device, compute_dependencies=True,
                             forced_model_wrapping=True, skip_previous_auto_load=True,
                             dummy_input=torch.zeros(1, int(hp["backbone"]["feature_width"]), device=device))
    if str(hp["optimizer"]["name"]) != "SGD" or float(hp["optimizer"]["momentum"]) != 0:
        raise ValueError("This pilot intentionally uses fresh SGD without momentum in every arm")
    optimizer = wl.watch_or_edit(torch.optim.SGD(model.parameters(), lr=float(hp["optimizer"]["lr"]),
                                                momentum=float(hp["optimizer"]["momentum"])), flag="optimizer")
    wl.start_training()
    return hp, model, optimizer


def evaluate(model, features, rows, batch_size):
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    with wl.guard_testing_context, torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunks.append(model(features[start:start + batch_size].to(device)).detach().cpu())
    logits = torch.cat(chunks)
    if len(logits) != len(rows) or not torch.isfinite(logits).all():
        raise AssertionError("Invalid evaluation predictions")
    labels = torch.tensor([row["label"] for row in rows])
    losses = nn.functional.cross_entropy(logits, labels, reduction="none")
    probabilities = logits.softmax(1)
    cases = [{**row, "prediction": int(logits[i].argmax()), "logits": logits[i].tolist(),
              "confidence": float(probabilities[i].max()), "loss": float(losses[i]),
              "true_label_margin": float(logits[i, row["label"]] - logits[i, 1 - row["label"]])}
             for i, row in enumerate(rows)]
    groups = {}
    for group in GROUP_NAMES:
        selected = [case for case in cases if case["group"] == group]
        if not selected:
            raise ValueError(f"Official evaluation split missing group {group}")
        groups[group] = {"count": len(selected),
                         "accuracy": sum(case["prediction"] == case["label"] for case in selected) / len(selected),
                         "loss": sum(case["loss"] for case in selected) / len(selected)}
    return {"cases": cases, "groups": groups, "model_age": model.get_age(),
            "accuracy": sum(case["prediction"] == case["label"] for case in cases) / len(cases)}


def schedule(rows, seed, policy, steps, batch_size):
    counts = {group: sum(row["group"] == group for row in rows) for group in GROUP_NAMES}
    weights = torch.tensor([1 / counts[row["group"]] if policy == "group_balanced" else 1.0 for row in rows],
                           dtype=torch.float64)
    indices = torch.multinomial(weights, steps * batch_size, replacement=True,
                                generator=torch.Generator().manual_seed(seed))
    return indices.reshape(steps, batch_size)


def train(model, optimizer, data, validation, hp, steps, seed, policy):
    features, rows = data
    device = next(model.parameters()).device
    labels = torch.tensor([row["label"] for row in rows], device=device)
    features = features.to(device)
    batches = schedule(rows, seed, policy, steps, int(hp["data"]["train_loader"]["batch_size"]))
    initial_age = model.get_age()
    every = int(hp["eval_full_to_train_steps_ratio"])
    tracker = wl.track_model_signals(model, every_n_steps=int(hp["model_signals_every_n_steps"]))
    layers = {path: model.get_layer_by_id(info["id"]) for path, info in linear_layers(model).items()}
    activations, handles, history = {}, [], []
    for path, layer in layers.items():
        def capture(module, inputs, output, path=path):
            activations[path] = {"mean": float(output.detach().mean()), "std": float(output.detach().std(unbiased=False))}
        handles.append(layer.register_forward_hook(capture))
    try:
        while model.get_age() - initial_age < steps:
            offset = model.get_age() - initial_age
            record = (offset + 1) % every == 0 or offset + 1 == steps
            before = {path: layer.weight.detach().clone() for path, layer in layers.items()} if record else {}
            model.train()
            indices = batches[offset].to(device)
            complete = False
            with wl.guard_training_context:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(features[indices]), labels[indices])
                loss.backward()
                gradients = {path: float(layer.weight.grad.norm()) for path, layer in layers.items()} if record else {}
                optimizer.step()
                complete = True
            if not complete or not math.isfinite(float(loss.detach())):
                raise AssertionError("Training failed or produced a nonfinite loss")
            if record:
                diagnostics = {path: {"gradient_norm": gradients[path], "weight_norm": float(layer.weight.detach().norm()),
                                      "update_norm": float((layer.weight.detach() - before[path]).norm()),
                                      "activation": dict(activations[path])} for path, layer in layers.items()}
                valid = evaluate(model, *validation, int(hp["data"]["test_loader"]["batch_size"]))
                history.append({"step": offset + 1, "model_age": model.get_age(), "training_loss": float(loss.detach()),
                                "validation_groups": valid["groups"], "layers": diagnostics})
    finally:
        tracker.remove()
        for handle in handles:
            handle.remove()
    return {"history": history, "schedule_sha256": hashlib.sha256(batches.numpy().tobytes()).hexdigest(),
            "steps_completed": model.get_age() - initial_age}


def run(config, cache, output):
    validate_config(config)
    torch.use_deterministic_algorithms(True)
    features, rows = cache["features"], cache["rows"]
    if cache["provenance"]["config"]["backbone"] != config["backbone"]:
        raise ValueError("Feature cache backbone differs from protocol")
    splits = {}
    for number, name in enumerate(("train", "validation", "test")):
        indices = [i for i, row in enumerate(rows) if row["split"] == number]
        splits[name] = features[indices], [rows[i] for i in indices]
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    baselines = {}
    for seed in config["seeds"]:
        hp, model, optimizer = register(config, output, f"baseline-{seed}", seed)
        training = train(model, optimizer, splits["train"], splits["validation"], hp,
                         int(hp["baseline_training_steps"]), seed + 100, "original")
        valid = evaluate(model, *splits["validation"], int(hp["data"]["test_loader"]["batch_size"]))
        path = checkpoints / f"baseline-{seed}.pt"
        torch.save(learned_parameters(model), path)
        baselines[str(seed)] = {"validation": valid, "training": training,
                                "checkpoint_sha256": sha256(path), "graph": model.get_model_graph()}
        dump(output / "baseline-validation.json", baselines)
        print(f"Baseline {seed}: validation groups {valid['groups']}", flush=True)
    target = min(GROUP_NAMES, key=lambda group: sum(baselines[str(seed)]["validation"]["groups"][group]["accuracy"]
                                                   for seed in config["seeds"]))
    demo_cases = []
    reference = baselines[str(config["seeds"][0])]["validation"]["cases"]
    for group in GROUP_NAMES:
        ordered = sorted((case for case in reference if case["group"] == group), key=lambda case: (-case["loss"], case["sample_id"]))
        demo_cases.extend(case["sample_id"] for case in ordered[:config["demo_cases_per_group"]])
    # Locked BEFORE inspecting any test predictions or choosing an intervention.
    protocol = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "target_group": target, "target_selection": "lowest mean baseline validation accuracy across all three seeds",
                "common_groups": ["0:0", "1:1"], "demo_validation_ids": demo_cases,
                "case_manifest_sha256": digest(splits["test"][1]),
                "preprocessing_sha256": cache["provenance"]["preprocessing_sha256"],
                "dataset_metadata_sha256": cache["provenance"]["metadata_sha256"],
                "threshold_status": "proposed_descriptive_only_not_team_approved"}
    dump(output / "locked-protocol.json", protocol)
    protocol_hash = sha256(output / "locked-protocol.json")
    print(f"Protocol locked: target {GROUP_NAMES[target]}, {protocol_hash}", flush=True)
    runs, comparisons = {}, {}
    for seed in config["seeds"]:
        parent = checkpoints / f"baseline-{seed}.pt"
        expected_validation = baselines[str(seed)]["validation"]["cases"]
        for arm in config["arms"]:
            run_id = f"{seed}-{arm['name']}"
            checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
            hp, model, optimizer = register(config, output, run_id, seed, checkpoint)
            batch_size = int(hp["data"]["test_loader"]["batch_size"])
            if evaluate(model, *splits["validation"], batch_size)["cases"] != expected_validation:
                raise AssertionError("Checkpoint prediction parity failed")
            before_test = evaluate(model, *splits["test"], batch_size)
            graph_before = model.get_model_graph()
            layers = linear_layers(model)
            events = []
            if arm["add_neurons"]:
                model.add_neurons(layers["0"]["id"], count=arm["add_neurons"])
                after = learned_parameters(model)
                width = int(hp["head"]["hidden_width"])
                if after["0"]["weight"].shape[0] != width + arm["add_neurons"] or after["2"]["weight"].shape[1] != width + arm["add_neurons"]:
                    raise AssertionError("Dependency propagation failed")
                if not (torch.equal(after["0"]["weight"][:width], checkpoint["0"]["weight"])
                        and torch.equal(after["0"]["bias"][:width], checkpoint["0"]["bias"])
                        and torch.equal(after["2"]["weight"][:, :width], checkpoint["2"]["weight"])
                        and torch.equal(after["2"]["bias"], checkpoint["2"]["bias"])):
                    raise AssertionError("Widening changed retained parameters")
                events.append({"operation": "add_neurons", "layer_path": "0", "live_layer_id": layers["0"]["id"],
                               "count": arm["add_neurons"], "before_width": width, "after_width": width + arm["add_neurons"],
                               "reason": "Test added head capacity against matched continued training", "retained_parameters_unchanged": True})
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if optimizer_ids != {id(p) for p in model.parameters()}:
                raise AssertionError("Optimizer references stale or missing parameters")
            immediate = evaluate(model, *splits["test"], batch_size)
            training = train(model, optimizer, splits["train"], splits["validation"], hp,
                             int(hp["training_steps_to_do"]), seed + 1000, arm["sampling"])
            final = evaluate(model, *splits["test"], batch_size)
            valid = evaluate(model, *splits["validation"], batch_size)
            final_path = checkpoints / f"{run_id}.pt"
            torch.save(learned_parameters(model), final_path)
            runs[run_id] = {"run_id": run_id, "seed": seed, "arm": arm, "evaluation_split": "test",
                            "parent_checkpoint_sha256": sha256(parent), "checkpoint_sha256": sha256(final_path),
                            "case_manifest_sha256": protocol["case_manifest_sha256"],
                            "preprocessing_sha256": protocol["preprocessing_sha256"], "protocol_sha256": protocol_hash,
                            "training_steps": training["steps_completed"], "optimizer_policy": config["optimizer_at_fork"],
                            "cases": final.pop("cases"), "metrics": final, "baseline_test": before_test,
                            "immediate_test": immediate, "training": training,
                            "demo_validation": [case for case in valid["cases"] if case["sample_id"] in demo_cases],
                            "events": events, "graph_before": graph_before, "graph_after": model.get_model_graph(),
                            "checks": {"checkpoint_parity": True, "optimizer_binding": True, "finite_training": True}}
            if arm["name"] != "continue":
                control = runs[f"{seed}-continue"]
                comparisons[run_id] = wl.compare_predictions(control, runs[run_id])
                if arm["sampling"] == "original" and control["training"]["schedule_sha256"] != training["schedule_sha256"]:
                    raise AssertionError("Capacity comparison used mismatched batches")
            print(f"Finished {run_id}: test groups {final['groups']}", flush=True)
            dump(output / "runs.json", runs)
        if runs[f"{seed}-resample"]["training"]["schedule_sha256"] != runs[f"{seed}-widen_resample"]["training"]["schedule_sha256"]:
            raise AssertionError("Balanced capacity comparison used mismatched batches")
        comparisons[f"{seed}-capacity_balanced"] = wl.compare_predictions(runs[f"{seed}-resample"], runs[f"{seed}-widen_resample"])
    train_counts = {group: sum(row["group"] == group for row in splits["train"][1]) for group in GROUP_NAMES}
    for run in runs.values():
        run["metrics"]["training_weighted_accuracy"] = sum(run["metrics"]["groups"][g]["accuracy"] * count for g, count in train_counts.items()) / len(splits["train"][1])
    report = {"schema_version": 1, "artifact_type": "waterbirds_controlled_experiment", "status": "completed",
              "created_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "protocol_sha256": protocol_hash,
              "feature_provenance": cache["provenance"], "group_names": GROUP_NAMES, "train_group_counts": train_counts,
              "split_counts": {name: len(data[1]) for name, data in splits.items()}, "baselines": baselines,
              "runs": runs, "comparisons": comparisons,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
              "source_sha256": {str(path.relative_to(Path.cwd())): sha256(path)
                                for path in [Path(__file__).resolve(), Path(wl.__file__).resolve().with_name("diagnostics.py")]},
              "claim_boundary": "Frozen ViT plus editable head; recorded experiment, not live Studio integration or structural attention editing"}
    dump(output / "report.json", report)
    wl.clear_all()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Trusted local cache produced by prepare_waterbirds.py; never load arbitrary pickles.
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    try:
        run(config, cache, args.output_dir)
        print(f"Experiment complete: {args.output_dir / 'report.json'}", flush=True)
    finally:
        wl.clear_all()


if __name__ == "__main__":
    main()
