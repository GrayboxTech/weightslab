"""Expand the proposal into a run matrix; never downloads data or trains."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_plan(config: dict) -> dict:
    """Produce paired jobs with a shared baseline identifier for each seed."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported experiment schema version")
    seeds = config["seeds"]
    arms = config["arms"]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names) or "continue" not in names:
        raise ValueError("Unique arms including the continued-training control are required")
    if config["training_steps_to_do"] <= 0:
        raise ValueError("Each arm needs a positive, matched training budget")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
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
