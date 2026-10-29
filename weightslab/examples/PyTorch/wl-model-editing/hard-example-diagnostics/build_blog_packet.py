"""Package measured experiment evidence, a report and real browser screenshots.

Requires `pip install markdown`. Run capture_blog_packet.cjs first. Generated
images and the dataset-bearing offline demo belong in an artifact directory,
not in the source repository. Review image redistribution terms before sharing.
"""
import argparse
import hashlib
import html
import json
import shutil
import statistics
import zipfile
from pathlib import Path

import markdown


def pooled(metrics):
    groups = [metrics["groups"][key] for key in ("0:0", "1:1")]
    return sum(g["count"] * g["accuracy"] for g in groups) / sum(g["count"] for g in groups)


def build(evidence, report_path, demo, output):
    report = json.loads((evidence / "followup.json").read_text())
    verification = json.loads((evidence / "verification.json").read_text())
    browser = json.loads((output / "browser-evidence.json").read_text())
    figures = json.loads((output / "screenshots.json").read_text())["screenshots"]
    assert report["status"] == "completed" and verification["passed"] and browser["passed"]
    assert hashlib.sha256((evidence / "followup.json").read_bytes()).hexdigest() == verification["report_sha256"]
    target = report["protocol"]["target_group"]
    runs = []
    for run_id, record in report["runs"].items():
        runs.append({"run": run_id, "seed": record["seed"], "arm": record["arm"],
                     "target_accuracy_pct": 100 * record["final"]["test"]["groups"][target]["accuracy"],
                     "common_accuracy_pct": 100 * pooled(record["final"]["test"]),
                     "immediate_max_logit_delta": record["max_immediate_logit_delta"]["test"],
                     "pruning": {count: {"target_accuracy_pct": 100 * value["test"]["groups"][target]["accuracy"],
                                         "common_accuracy_pct": 100 * pooled(value["test"])}
                                 for count, value in record["pruning"].items()}})
    arms = {}
    for arm in report["protocol"]["arms"]:
        selected = [r for r in runs if r["arm"] == arm]
        assert sorted(r["seed"] for r in selected) == sorted(report["protocol"]["config"]["seeds"])
        values = [r["target_accuracy_pct"] for r in selected]
        arms[arm] = {"target_mean_pct": statistics.mean(values), "target_seed_range_pct": [min(values), max(values)],
                     "common_mean_pct": statistics.mean(r["common_accuracy_pct"] for r in selected),
                     "maximum_immediate_logit_delta": max(r["immediate_max_logit_delta"] for r in selected)}
    summary = {"protocol_sha256": report["protocol_sha256"], "source_commit": report["protocol"]["source_commit"],
               "claim_boundary": report["protocol"]["claim_boundary"], "aggregation": "Arithmetic mean across paired seeds; common accuracy pooled by group count within seed",
               "target_test_count_per_seed": 642, "common_test_count_per_seed": 2897, "arms": arms, "runs": runs}
    with (output / "experiment-summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    shutil.copy2(report_path, output / "report.md")
    for filename in ("followup.json", "verification.json"):
        shutil.copy2(evidence / filename, output / filename)
    shutil.copy2(demo, output / "recorded-demo.html")
    sections = []
    for figure in figures:
        relative = Path(figure["file"])
        if relative.is_absolute() or ".." in relative.parts or not (output / relative).is_file():
            raise ValueError("Invalid screenshot path")
        sections.append(f'<figure><a href="{html.escape(relative.as_posix(), quote=True)}"><img loading="lazy" src="{html.escape(relative.as_posix(), quote=True)}" alt="{html.escape(figure["caption"], quote=True)}"></a><figcaption>{html.escape(figure["caption"])}</figcaption></figure>')
    body = markdown.markdown(report_path.read_text(), extensions=["tables", "fenced_code"])
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WeightsLab experiment report — 29 October 2026</title><style>
body{margin:0;background:#f2f4f5;color:#21313d;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:0 auto;padding:32px}article{background:white;border:1px solid #d9e0e5;padding:32px;border-radius:8px}h1{font-size:30px;line-height:1.25}h2{font-size:23px;margin-top:36px}h3{font-size:18px}a{color:#126c65}nav{display:flex;flex-wrap:wrap;gap:12px 24px;margin-bottom:24px}table{display:block;overflow:auto;border-collapse:collapse;font-size:14px;width:100%}td,th{padding:10px 12px;border-bottom:1px solid #d9e0e5;text-align:left}th{background:#f2f6f5}code{font-size:13px;overflow-wrap:anywhere}pre{overflow:auto;padding:16px;background:#f2f6f5}figure{margin:24px 0;padding:16px;background:white;border:1px solid #d9e0e5;border-radius:8px}img{width:100%;height:auto}figcaption{margin-top:12px;font-size:14px}p,li{max-width:94ch}footer{font-size:13px;padding:20px 0}@media(max-width:600px){main{padding:12px}article{padding:18px}h1{font-size:25px}td,th{padding:8px}}@media print{nav{display:none}main{padding:0}figure{break-inside:avoid}article{border:0;padding:0}}
</style></head><body><main><nav><a href="#report">Report</a><a href="#screenshots">Screenshots</a><a href="recorded-demo.html">Interactive recorded inspector</a><a href="experiment-summary.json">Results JSON</a><a href="verification.json">Independent replay</a><a href="browser-evidence.json">Browser evidence</a><a href="report.md">Markdown report</a></nav><article id="report">'''
    page += body + '</article><section id="screenshots"><h2>Captured tool screens</h2>' + "".join(sections)
    page += '</section><footer>Recorded inspector works offline. Live editing requires the Python service and experiment artifacts described in the README. Review dataset image permissions before public redistribution.</footer></main></body></html>'
    with (output / "index.html").open("x") as stream:
        stream.write(page)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    with (output / "manifest-sha256.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    archive = output.parent / "weightslab-report-packet-2026-10-29.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, "weightslab-report-packet/" + path.relative_to(output).as_posix())
    print(json.dumps({"report": str(output / "index.html"), "archive": str(archive), "screenshots": len(figures), "bytes": archive.stat().st_size}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--recorded-demo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.evidence_dir, args.report_md, args.recorded_demo, args.output)
