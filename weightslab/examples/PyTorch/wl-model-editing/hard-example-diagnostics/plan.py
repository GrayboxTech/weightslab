"""Expand the proposal into a run matrix; never downloads data or trains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate_config(config: dict) -> None:
    """Reject plans that silently corrupt the paired experimental control."""
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported experiment schema version")
    seeds = config.get("seeds")
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Seeds must be nonempty, unique, nonnegative integers")
    for key in ("baseline_training_steps", "training_steps_to_do"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    arms = config.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("A nonempty list of intervention arms is required")
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("name"), str) or not arm["name"].strip():
            raise ValueError("Each arm needs a nonempty name")
        if type(arm.get("add_neurons")) is not int or arm["add_neurons"] < 0:
            raise ValueError("add_neurons must be a nonnegative integer")
        if arm.get("sampling") not in ("original", "group_balanced"):
            raise ValueError("Unsupported sampling policy")
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names) or "continue" not in names:
        raise ValueError("Unique arms including the continued-training control are required")
    control = next(arm for arm in arms if arm["name"] == "continue")
    if control["add_neurons"] != 0 or control["sampling"] != "original":
        raise ValueError("The continued-training control must not edit capacity or sampling")
    if config.get("optimizer_at_fork") != "fresh_identical_optimizer_in_all_arms":
        raise ValueError("This proposal requires an identical optimizer-reset policy in all arms")


def build_plan(config: dict) -> dict:
    """Produce paired jobs with a shared baseline identifier for each seed."""
    validate_config(config)
    seeds = config["seeds"]
    arms = config["arms"]
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    jobs = []
    for seed in seeds:
        for arm in arms:
            jobs.append({
                "run_id": f"{config['experiment_name']}-{fingerprint[:8]}-{seed}-{arm['name']}",
                "seed": seed,
                "parent_checkpoint_id": f"baseline-{fingerprint[:8]}-seed-{seed}",
                "parent_checkpoint_sha256": None,
                "case_manifest_sha256": config["dataset"]["case_manifest_sha256"],
                "training_steps_to_do": config["training_steps_to_do"],
                "optimizer_at_fork": config["optimizer_at_fork"],
                "intervention": arm,
                "status": "planned_not_executed",
            })
    return {
        "schema_version": 1,
        "artifact_type": "experiment_plan",
        "config_sha256": fingerprint,
        "config": config,
        "jobs": jobs,
        "next_gate": "Prepare dataset, validate baseline failure and lock case manifest",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing an already-reviewed plan.
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Planned {len(plan['jobs'])} runs (not executed): {args.output}")


if __name__ == "__main__":
    main()
