"""Replay every saved final head and verify reports against a second full run."""

import argparse
import json
from pathlib import Path

import torch
from prepare_waterbirds import sha256
from run_waterbirds import digest
from torch import nn

import weightslab as wl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run_dir / "report.json").read_text())
    reference = json.loads((args.reference_run / "report.json").read_text())
    config = report["protocol"]["config"]
    hp = wl.watch_or_edit(config, flag="hyperparameters", defaults=config)
    torch.set_num_threads(int(hp["num_threads"]))
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    indices = [i for i, row in enumerate(cache["rows"]) if row["split"] == 2]
    rows = [cache["rows"][i] for i in indices]
    features = cache["features"][indices]
    assert digest(rows) == report["protocol"]["case_manifest_sha256"]
    assert sha256(args.run_dir / "locked-protocol.json") == report["protocol_sha256"]
    checks = {}
    for run_id, run in report["runs"].items():
        assert run["cases"] == reference["runs"][run_id]["cases"], f"Predictions changed on repeat: {run_id}"
        assert run["metrics"] == reference["runs"][run_id]["metrics"]
        assert run["training"] == reference["runs"][run_id]["training"]
        path = args.run_dir / "checkpoints" / f"{run_id}.pt"
        assert sha256(path) == run["checkpoint_sha256"]
        assert sha256(args.run_dir / "checkpoints" / f"baseline-{run['seed']}.pt") == run["parent_checkpoint_sha256"]
        state = torch.load(path, map_location="cpu", weights_only=True)
        width = int(hp["head"]["hidden_width"]) + run["arm"]["add_neurons"]
        head = nn.Sequential(nn.Linear(int(hp["backbone"]["feature_width"]), width),
                             nn.LeakyReLU(float(hp["head"]["negative_slope"])),
                             nn.Linear(width, int(hp["head"]["classes"])))
        for layer, parameters in state.items():
            head[int(layer)].load_state_dict(parameters)
        with torch.no_grad():
            logits = head(features)
        expected = torch.tensor([case["logits"] for case in run["cases"]])
        assert torch.allclose(logits, expected, atol=1e-4, rtol=1e-4), f"Replay failed: {run_id}"
        assert logits.argmax(1).tolist() == [case["prediction"] for case in run["cases"]]
        for row, case in zip(rows, run["cases"]):
            assert all(row[key] == case[key] for key in row)
        if run["arm"]["name"] != "continue":
            assert wl.compare_predictions(report["runs"][f"{run['seed']}-continue"], run) == report["comparisons"][run_id]
        checks[run_id] = {"repeated_predictions_exact": True, "repeated_history_exact": True,
                          "saved_checkpoint_replayed": True, "max_logit_replay_error": float((logits - expected).abs().max()),
                          "cases_replayed": len(rows)}
    verification = {"status": "passed", "report_sha256": sha256(args.run_dir / "report.json"),
                    "reference_report_sha256": sha256(args.reference_run / "report.json"),
                    "protocol_sha256": report["protocol_sha256"], "checks": checks}
    (args.run_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Verified {len(checks)} saved heads, {len(rows)} cases each; exact repeated predictions and histories.")
    wl.clear_all()


if __name__ == "__main__":
    main()
