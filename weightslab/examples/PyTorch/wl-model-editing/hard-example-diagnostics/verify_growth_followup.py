"""Replay saved heads with plain PyTorch, independently of the editing engine."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, cache_path, output):
    report = json.loads((directory / "followup.json").read_text())
    assert report["status"] == "completed"
    assert sha(cache_path) == report["protocol"]["cache_sha256"]
    assert sha(directory / "protocol.json") == report["protocol_sha256"]
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    torch.set_num_threads(2)
    slope = report["protocol"]["config"]["head"]["negative_slope"]
    evaluations, saved_cases, maximum = 0, 0, 0.0
    records = []
    for run_id, run in report["runs"].items():
        checkpoint = directory / "checkpoints" / f"{run_id}.pt"
        assert sha(checkpoint) == run["checkpoint_sha256"]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        predictions_path = directory / f"predictions-{run_id}.json"
        saved = json.loads(predictions_path.read_text())
        for split_number, split in ((1, "validation"), (2, "test")):
            indices = [i for i, r in enumerate(cache["rows"]) if r["split"] == split_number]
            rows = [cache["rows"][i] for i in indices]
            features = cache["features"][indices]
            assert len({r["sample_id"] for r in saved[split]}) == len(rows) == len(saved[split])
            for observed, expected in zip(saved[split], rows):
                assert all(observed[k] == expected[k] for k in ("sample_id", "label", "group"))
            for count, recorded in run["pruning"].items():
                width = state["0"]["weight"].shape[0] - int(count)
                with torch.no_grad():
                    hidden = F.leaky_relu(F.linear(features, state["0"]["weight"][:width], state["0"]["bias"][:width]), slope)
                    logits = F.linear(hidden, state["2"]["weight"][:, :width], state["2"]["bias"])
                predicted = logits.argmax(1).tolist()
                for group, metric in recorded[split]["groups"].items():
                    members = [i for i, row in enumerate(rows) if row["group"] == group]
                    accuracy = sum(predicted[i] == rows[i]["label"] for i in members) / len(members)
                    assert len(members) == metric["count"]
                    assert abs(accuracy - metric["accuracy"]) < 1e-12, (run_id, split, count, group)
                evaluations += 1
                if int(count) == 0:
                    assert predicted == [row["prediction"] for row in saved[split]]
                    delta = float((logits - torch.tensor([row["logits"] for row in saved[split]])).abs().max())
                    assert delta <= 1e-4, (run_id, split, delta)
                    maximum = max(maximum, delta)
                    saved_cases += len(rows)
        records.append({"run": run_id, "checkpoint_sha256": sha(checkpoint), "predictions_sha256": sha(predictions_path)})
    evidence = {"passed": True, "verified_at": datetime.now(timezone.utc).isoformat(),
                "method": "Plain PyTorch functional forward; no WeightsLab editing or evaluation helpers",
                "report_sha256": sha(directory / "followup.json"), "verifier_sha256": sha(Path(__file__)),
                "protocol_sha256": report["protocol_sha256"], "checkpoints": len(records),
                "full_split_evaluations": evaluations, "saved_prediction_rows": saved_cases,
                "maximum_replay_logit_delta": maximum, "logit_tolerance": 1e-4,
                "checks": ["cache and checkpoint hashes", "full ordered IDs, labels and groups", "exact saved predicted labels", "every group accuracy at every pruning level"],
                "records": records}
    with output.open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in evidence.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir, args.cache, args.output)
