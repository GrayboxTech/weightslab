"""Paired, provenance-checked prediction comparisons without a running ledger."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping


def _cases(snapshot: Mapping) -> dict:
    cases = snapshot.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a nonempty list")
    indexed = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise TypeError("Each case must be a mapping")
        sample_id, group = case.get("sample_id"), case.get("group")
        if not isinstance(sample_id, str) or not sample_id or sample_id in indexed:
            raise ValueError("sample_id must be a unique, nonempty string")
        if not isinstance(group, str) or not group:
            raise ValueError("group must be a nonempty string")
        for key in ("label", "prediction"):
            if type(case.get(key)) is not int or case[key] < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        margin = case.get("true_label_margin")
        if margin is not None and (type(margin) not in (int, float) or not math.isfinite(margin)):
            raise ValueError("true_label_margin must be finite or null")
        indexed[sample_id] = case
    return indexed


def compare_predictions(control: Mapping, intervention: Mapping) -> dict:
    """Compare the same held-out cases after equally budgeted branch training.

    Each input is a JSON-compatible snapshot with ``run_id``, ``cases`` and
    provenance: ``parent_checkpoint_sha256``, ``case_manifest_sha256``,
    ``preprocessing_sha256``, ``evaluation_split``, ``training_steps`` and
    ``optimizer_policy``. Each case has a unique string ``sample_id``, string
    ``group``, nonnegative integer ``label`` and ``prediction``; optional
    ``true_label_margin`` is finite or null. Case order need not match.

    Rejects changed cohorts/labels/groups, missing provenance and mismatched
    parents, preprocessing, evaluation splits, budgets or optimizer policies.
    This checks supplied provenance, not the checkpoint files or how a model
    was trained. The caller remains responsible for recording it truthfully.

    Returns per-group empirical accuracy/delta, corrected/regressed IDs and
    per-case outcomes. Accuracies use fractions, not percentages. No statistical
    significance, causal mechanism or dataset-weighted average is inferred.
    This pure function neither mutates inputs nor reads a global ledger.
    """
    if not isinstance(control, Mapping) or not isinstance(intervention, Mapping):
        raise TypeError("Both snapshots must be mappings")
    hash_keys = ("parent_checkpoint_sha256", "case_manifest_sha256", "preprocessing_sha256")
    for snapshot in (control, intervention):
        for key in hash_keys:
            if not isinstance(snapshot.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot[key]):
                raise ValueError(f"{key} must be a lowercase SHA-256 digest")
        for key in ("run_id", "evaluation_split", "optimizer_policy"):
            if not isinstance(snapshot.get(key), str) or not snapshot[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if type(snapshot.get("training_steps")) is not int or snapshot["training_steps"] < 0:
            raise ValueError("training_steps must be a nonnegative integer")
    if control["run_id"] == intervention["run_id"]:
        raise ValueError("Control and intervention must have distinct run IDs")
    for key in (*hash_keys, "evaluation_split", "training_steps", "optimizer_policy"):
        if control[key] != intervention[key]:
            raise ValueError(f"Unpaired comparison: {key} differs")
    before, after = _cases(control), _cases(intervention)
    if before.keys() != after.keys():
        raise ValueError("Case cohorts differ; silently intersecting is not allowed")
    rows = []
    for sample_id in sorted(before):
        left, right = before[sample_id], after[sample_id]
        if (left["label"], left["group"]) != (right["label"], right["group"]):
            raise ValueError(f"Case label/group changed: {sample_id}")
        old, new = left["prediction"] == left["label"], right["prediction"] == right["label"]
        outcome = "corrected" if new and not old else "regressed" if old and not new else "unchanged"
        left_margin, right_margin = left.get("true_label_margin"), right.get("true_label_margin")
        margin_delta = None if left_margin is None or right_margin is None else right_margin - left_margin
        if margin_delta is not None and not math.isfinite(margin_delta):
            raise ValueError("Margin difference overflowed")
        rows.append({"sample_id": sample_id, "group": left["group"], "label": left["label"],
                     "control_prediction": left["prediction"], "intervention_prediction": right["prediction"],
                     "control_correct": old, "intervention_correct": new, "outcome": outcome,
                     "margin_delta": margin_delta})

    def summarize(subset):
        count = len(subset)
        old = sum(row["control_correct"] for row in subset)
        new = sum(row["intervention_correct"] for row in subset)
        return {"count": count, "control_accuracy": old / count, "intervention_accuracy": new / count,
                "accuracy_delta": (new - old) / count,
                "corrected_count": sum(row["outcome"] == "corrected" for row in subset),
                "regressed_count": sum(row["outcome"] == "regressed" for row in subset)}

    return {"schema_version": 1, "artifact_type": "paired_prediction_comparison",
            "control_run_id": control["run_id"], "intervention_run_id": intervention["run_id"],
            "provenance": {key: control[key] for key in (*hash_keys, "evaluation_split", "training_steps", "optimizer_policy")},
            "overall": summarize(rows),
            "groups": {group: summarize([row for row in rows if row["group"] == group])
                       for group in sorted({row["group"] for row in rows})},
            "corrected_ids": [row["sample_id"] for row in rows if row["outcome"] == "corrected"],
            "regressed_ids": [row["sample_id"] for row in rows if row["outcome"] == "regressed"],
            "cases": rows}
