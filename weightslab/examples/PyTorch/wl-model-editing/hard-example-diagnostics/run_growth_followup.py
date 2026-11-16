"""Paired growth-initialization experiment and causal removal of added head units.

Exploratory follow-up to an already inspected benchmark, not a new untouched test.
No backbone editing, new data, or significance claim. Zeroing new outgoing weights
is an experiment-local initialization policy, not a new WeightsLab public option.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("WEIGHTSLAB_OPENCODE_AUTOINSTALL", "0")
os.environ.setdefault("WL_NO_TELEMETRY", "1")
os.environ.setdefault("WEIGHTSLAB_PROJECTION", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import weightslab as wl
from prepare_waterbirds import sha256
from run_waterbirds import dump, evaluate, learned_parameters, linear_layers, register, train

ARMS = {
    "continue": ("none", "original"), "resample": ("none", "group_balanced"),
    "widen": ("random", "original"), "widen_resample": ("random", "group_balanced"),
    "preserve": ("zero_outgoing", "original"), "preserve_resample": ("zero_outgoing", "group_balanced"),
}


def split_cache(cache):
    result = {}
    for number, name in enumerate(("train", "validation", "test")):
        indices = [i for i, row in enumerate(cache["rows"]) if row["split"] == number]
        result[name] = cache["features"][indices], [cache["rows"][i] for i in indices]
    return result


def assert_optimizer(model, optimizer):
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in model.parameters()}:
        raise AssertionError("Optimizer references do not match live model parameters")


def grow(model, optimizer, count, policy):
    if policy not in ("none", "random", "zero_outgoing"):
        raise ValueError("Unsupported initialization")
    before = learned_parameters(model)
    width = before["0"]["weight"].shape[0]
    if policy == "none":
        return {"operation": "no edit", "before_width": width, "after_width": width}
    if count <= 0:
        raise ValueError("Growth count must be positive")
    model.add_neurons(linear_layers(model)["0"]["id"], count=count)
    if policy == "zero_outgoing":
        # Keep random incoming weights so outgoing gradients can break symmetry.
        with torch.no_grad():
            model.get_layer_by_id(linear_layers(model)["2"]["id"]).weight[:, width:].zero_()
    after = learned_parameters(model)
    if after["0"]["weight"].shape[0] != width + count or after["2"]["weight"].shape[1] != width + count:
        raise AssertionError("Dependent classifier shape did not follow growth")
    for path, parameter, retained in (("0", "weight", after["0"]["weight"][:width]),
                                      ("0", "bias", after["0"]["bias"][:width]),
                                      ("2", "weight", after["2"]["weight"][:, :width]),
                                      ("2", "bias", after["2"]["bias"])):
        if not torch.equal(retained, before[path][parameter]):
            raise AssertionError("Growth changed a retained parameter")
    assert_optimizer(model, optimizer)
    return {"operation": "model.add_neurons", "initialization": policy, "count": count,
            "before_width": width, "after_width": width + count,
            "retained_parameters_unchanged": True, "optimizer_binding": True}


def prune_added(model, optimizer, original_width, count):
    width = learned_parameters(model)["0"]["weight"].shape[0]
    if type(count) is not int or count < 0 or count > width - original_width:
        raise ValueError("Can only prune the added suffix of units")
    if count:
        model.prune_neurons(linear_layers(model)["0"]["id"], list(range(width - count, width)))
    after = learned_parameters(model)
    if after["0"]["weight"].shape[0] != width - count or after["2"]["weight"].shape[1] != width - count:
        raise AssertionError("Dependent classifier shape did not follow pruning")
    assert_optimizer(model, optimizer)
    return {"operation": "model.prune_neurons" if count else "no edit", "count": count,
            "indices": list(range(width - count, width)), "before_width": width,
            "after_width": width - count, "optimizer_binding": True}


def summary(evaluation):
    return {key: value for key, value in evaluation.items() if key != "cases"}


def max_logit_delta(first, second):
    if [r["sample_id"] for r in first["cases"]] != [r["sample_id"] for r in second["cases"]]:
        raise ValueError("Evaluation order changed")
    return max(abs(a - b) for x, y in zip(first["cases"], second["cases"])
               for a, b in zip(x["logits"], y["logits"]))


def run(parent_dir, cache_path, output, source_commit):
    parent = json.loads((parent_dir / "report.json").read_text())
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["provenance"]["metadata_sha256"] != parent["feature_provenance"]["metadata_sha256"]:
        raise ValueError("Feature cache metadata differs from parent")
    if cache["provenance"]["weights_sha256"] != parent["feature_provenance"]["weights_sha256"]:
        raise ValueError("Feature cache backbone weights differ from parent")
    config = copy.deepcopy(parent["protocol"]["config"])
    config.update(experiment_name="waterbirds_growth_initialization", source_commit=source_commit,
                  prune_counts=[0, 4, 8, 16], preservation_tolerance=1e-5)
    output.mkdir(parents=True, exist_ok=False)
    (output / "checkpoints").mkdir()
    splits = split_cache(cache)
    torch.use_deterministic_algorithms(True)
    protocol = {"config": config, "parent_protocol_sha256": parent["protocol_sha256"],
                "cache_sha256": sha256(cache_path), "source_commit": source_commit,
                "runner_sha256": sha256(Path(__file__)),
                "helper_sha256": sha256(Path(__file__).with_name("run_waterbirds.py")),
                "target_group": parent["protocol"]["target_group"], "arms": ARMS,
                "claim_boundary": "Exploratory follow-up after inspecting E1 test results. No tuning on new test results; no fresh holdout or significance claim.",
                "pruning_policy": "Fixed suffix order, no ranking by evaluation performance; no retraining after removal."}
    dump(output / "protocol.json", protocol)
    result = {"status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol, "protocol_sha256": sha256(output / "protocol.json"),
              "torch_version": torch.__version__, "runs": {}}
    for seed in config["seeds"]:
        parent_path = parent_dir / "checkpoints" / f"baseline-{seed}.pt"
        if sha256(parent_path) != parent["baselines"][str(seed)]["checkpoint_sha256"]:
            raise ValueError("Parent checkpoint hash mismatch")
        state = torch.load(parent_path, map_location="cpu", weights_only=True)
        for arm, (policy, sampling) in ARMS.items():
            run_id = f"{seed}-{arm}"
            hp, model, optimizer = register(config, output, run_id, seed, state)
            batch = int(hp["data"]["test_loader"]["batch_size"])
            baseline = {name: evaluate(model, *splits[name], batch) for name in ("validation", "test")}
            expected = parent["baselines"][str(seed)]["validation"]
            if max_logit_delta(expected, baseline["validation"]) > float(hp["preservation_tolerance"]):
                raise AssertionError("Saved parent replay differs from previous experiment")
            event = grow(model, optimizer, int(hp["head"]["add_neurons"]), policy)
            immediate = {name: evaluate(model, *splits[name], batch) for name in baseline}
            deltas = {name: max_logit_delta(baseline[name], immediate[name]) for name in baseline}
            if policy == "zero_outgoing" and max(deltas.values()) > float(hp["preservation_tolerance"]):
                raise AssertionError("Output-preserving initialization failed")
            training = train(model, optimizer, splits["train"], splits["validation"], hp,
                             int(hp["training_steps_to_do"]), seed + 1000, sampling)
            final = {name: evaluate(model, *splits[name], batch) for name in baseline}
            checkpoint = learned_parameters(model)
            checkpoint_path = output / "checkpoints" / f"{run_id}.pt"
            torch.save(checkpoint, checkpoint_path)
            record = {"seed": seed, "arm": arm, "sampling": sampling, "initialization": policy,
                      "parent_checkpoint_sha256": sha256(parent_path), "checkpoint_sha256": sha256(checkpoint_path),
                      "event": event, "max_immediate_logit_delta": deltas,
                      "baseline": {k: summary(v) for k, v in baseline.items()},
                      "immediate": {k: summary(v) for k, v in immediate.items()},
                      "final": {k: summary(v) for k, v in final.items()}, "training": training,
                      "pruning": {}, "checks": {"retained_parameters": True, "optimizer_binding": True}}
            # Export all predictions for independent checks, not just screenshot cases.
            dump(output / f"predictions-{run_id}.json", {k: v["cases"] for k, v in final.items()})
            width = int(hp["head"]["hidden_width"])
            previous = 0
            for count in (hp["prune_counts"] if policy != "none" else [0]):
                pruning = prune_added(model, optimizer, width, count - previous)
                previous = count
                measured = {name: evaluate(model, *splits[name], batch) for name in baseline}
                record["pruning"][str(count)] = {"event": pruning, **{k: summary(v) for k, v in measured.items()}}
            if policy != "none":
                # Removal is not rollback: retained weights learned too. Verify only
                # that pruning equals the retained-subnetwork mathematical forward.
                features = splits["validation"][0]
                retained = torch.nn.Sequential(torch.nn.Linear(features.shape[1], width),
                                              torch.nn.LeakyReLU(float(hp["head"]["negative_slope"])),
                                              torch.nn.Linear(width, int(hp["head"]["classes"])))
                retained[0].load_state_dict({"weight": checkpoint["0"]["weight"][:width], "bias": checkpoint["0"]["bias"][:width]})
                retained[2].load_state_dict({"weight": checkpoint["2"]["weight"][:, :width], "bias": checkpoint["2"]["bias"]})
                with torch.no_grad():
                    predicted = retained(features).argmax(1).tolist()
                if predicted != [c["prediction"] for c in measured["validation"]["cases"]]:
                    raise AssertionError("Pruned API model differs from retained subnetwork")
                record["checks"]["pruned_subnetwork_parity"] = True
            result["runs"][run_id] = record
            dump(output / "followup.json", result)
            print(f"Finished {run_id}: target={record['final']['test']['groups'][protocol['target_group']]['accuracy']:.4f}; immediate_max_delta={deltas['test']:.6g}", flush=True)
        for sampling in ("original", "group_balanced"):
            hashes = {r["training"]["schedule_sha256"] for r in result["runs"].values() if r["seed"] == seed and r["sampling"] == sampling}
            if len(hashes) != 1:
                raise AssertionError("Matched arms saw different training batches")
    result["status"] = "completed"
    dump(output / "followup.json", result)
    wl.clear_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    run(args.parent_dir, args.cache, args.output, args.source_commit)
