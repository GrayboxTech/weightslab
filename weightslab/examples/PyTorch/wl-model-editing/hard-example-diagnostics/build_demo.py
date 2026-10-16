"""Build an offline results page from measured report, images and attribution."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

CHART_SHA256 = "48444a82d4edcb5bec0f1965faacdde18d9c17db3063d042abada2f705c9f54a"


def build(report, attribution, dataset_root, chart_source, output):
    if report["status"] != "completed" or report["artifact_type"] != "waterbirds_controlled_experiment":
        raise ValueError("Demo requires a completed real-data report")
    if report["protocol_sha256"] != attribution["protocol_sha256"]:
        raise ValueError("Attribution and experiment protocols differ")
    if hashlib.sha256(chart_source).hexdigest() != CHART_SHA256:
        raise ValueError("Expected pinned Chart.js 4.5.1 UMD build")
    media_ids = set()
    # Deterministic post-hoc illustrations; never used to compute reported metrics.
    for comparison in report["comparisons"].values():
        for outcome in ("corrected", "regressed"):
            for group in report["group_names"]:
                selected = [row["sample_id"] for row in comparison["cases"] if row["outcome"] == outcome and row["group"] == group]
                media_ids.update(sorted(selected)[:3])
    media = {}
    for sample_id in sorted(media_ids):
        path = (dataset_root / sample_id).resolve()
        if not path.is_relative_to(dataset_root.resolve()):
            raise ValueError("Image path escapes dataset root")
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((224, 224))
            stream = io.BytesIO()
            image.save(stream, format="JPEG", quality=80)
        media[sample_id] = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()
    data = {key: report[key] for key in ("protocol", "protocol_sha256", "group_names", "train_group_counts", "split_counts", "created_at")}
    data.update(media=media, attribution=attribution, runs={}, comparisons={})
    for run_id, run in report["runs"].items():
        data["runs"][run_id] = {key: run[key] for key in ("seed", "arm", "metrics", "training", "events", "checks", "parent_checkpoint_sha256", "checkpoint_sha256")}
        data["runs"][run_id]["preview_cases"] = [case for case in run["cases"] if case["sample_id"] in media]
        data["runs"][run_id]["immediate_groups"] = run["immediate_test"]["groups"]
        data["runs"][run_id]["baseline_groups"] = run["baseline_test"]["groups"]
    for run_id, comparison in report["comparisons"].items():
        data["comparisons"][run_id] = {key: comparison[key] for key in ("overall", "groups")}
        data["comparisons"][run_id]["preview_cases"] = [case for case in comparison["cases"] if case["sample_id"] in media]
    template = Path(__file__).with_name("demo_template.html").read_text()
    serialized = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.replace("/*__CHART_JS__*/", chart_source.decode()).replace("/*__DATA__*/{}", serialized)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(html)
    print(f"Offline demo ready: {output} ({len(media)} illustrative test images; full split metrics)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--chart-js", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads((args.run_dir / "report.json").read_text()),
          json.loads((args.run_dir / "attribution.json").read_text()),
          args.dataset_root, args.chart_js.read_bytes(), args.output)


if __name__ == "__main__":
    main()
