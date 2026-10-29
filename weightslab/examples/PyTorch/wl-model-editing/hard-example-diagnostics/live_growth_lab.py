"""Loopback-only live editing lab: restore, edit through WL, reevaluate, audit.

This custom experiment UI is not Weights Studio or its gRPC transport. It does
not train in the request path or overwrite checkpoints. Cached frozen-ViT
features are used for inference; only the head is edited.
"""
from __future__ import annotations

import argparse
import copy
import json
import secrets
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from run_growth_followup import (ARMS, evaluate, grow, learned_parameters,
                                 max_logit_delta, prune_added, register,
                                 sha256, split_cache, summary, torch, wl)
from run_waterbirds import digest


def validate_request(data, seeds):
    if not isinstance(data, dict) or set(data) != {"seed", "sampling", "initialization", "stage", "prune", "split"}:
        raise ValueError("Expected exactly seed, sampling, initialization, stage, prune, split")
    if type(data["seed"]) is not int or data["seed"] not in seeds:
        raise ValueError("Unknown paired seed")
    if data["sampling"] not in ("original", "group_balanced") or data["initialization"] not in ("none", "random", "zero_outgoing"):
        raise ValueError("Unknown intervention")
    if data["stage"] not in ("immediate", "trained") or data["split"] not in ("validation", "test"):
        raise ValueError("Unknown stage or split")
    if type(data["prune"]) is not int or data["prune"] not in (0, 4, 8, 16):
        raise ValueError("Prune must be 0, 4, 8 or 16 added units")
    if data["initialization"] == "none" and data["prune"]:
        raise ValueError("An unchanged head has no added units to prune")
    return data


class Lab:
    def __init__(self, parent, followup, cache_path, demo_path, output):
        self.parent, self.followup, self.output = parent, followup, output
        self.report = json.loads((followup / "followup.json").read_text())
        if self.report["status"] != "completed":
            raise ValueError("Complete the follow-up experiment first")
        if sha256(cache_path) != self.report["protocol"]["cache_sha256"]:
            raise ValueError("Feature cache does not match the recorded experiment")
        self.splits = split_cache(torch.load(cache_path, map_location="cpu", weights_only=False))
        self.config = copy.deepcopy(self.report["protocol"]["config"])
        self.config.update(device="cpu", num_threads=2, experiment_name="live_growth_lab")
        self.preprocessing_sha256 = json.loads((parent / "locked-protocol.json").read_text())["preprocessing_sha256"]
        demo, _ = json.JSONDecoder().raw_decode(demo_path.read_text().partition("const DATA=")[2])
        self.media = demo["media"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.sequence = 0

    def checkpoint(self, seed, arm, stage):
        if stage == "immediate":
            path = self.parent / "checkpoints" / f"baseline-{seed}.pt"
            expected = self.report["runs"][f"{seed}-{arm}"]["parent_checkpoint_sha256"]
        else:
            path = self.followup / "checkpoints" / f"{seed}-{arm}.pt"
            expected = self.report["runs"][f"{seed}-{arm}"]["checkpoint_sha256"]
        if sha256(path) != expected:
            raise ValueError("Saved checkpoint changed since the experiment")
        return torch.load(path, map_location="cpu", weights_only=True), expected

    def load(self, state, seed, run_id):
        config = copy.deepcopy(self.config)
        config["head"]["hidden_width"] = state["0"]["weight"].shape[0]
        return register(config, self.output, run_id, seed, state)

    def execute(self, data):
        data = validate_request(data, self.config["seeds"])
        with self.lock:
            started = time.monotonic()
            self.sequence += 1
            request_id = f"edit-{self.sequence:04d}-{secrets.token_hex(3)}"
            seed, sampling, policy, stage, count, split = (data[key] for key in ("seed", "sampling", "initialization", "stage", "prune", "split"))
            arm = next(name for name, values in ARMS.items() if values == (policy, sampling))
            control_arm = "continue" if sampling == "original" else "resample"
            state, reference_sha = self.checkpoint(seed, control_arm, stage)
            hp, model, optimizer = self.load(state, seed, request_id + "-reference")
            batch = int(hp["data"]["test_loader"]["batch_size"])
            reference = evaluate(model, *self.splits[split], batch)
            state, checkpoint_sha = self.checkpoint(seed, arm, stage)
            hp, model, optimizer = self.load(state, seed, request_id)
            events = [{"operation": "restore immutable checkpoint", "sha256": checkpoint_sha}]
            if stage == "immediate":
                events.append(grow(model, optimizer, int(hp["head"]["add_neurons"]), policy))
            before = evaluate(model, *self.splits[split], batch)
            original_width = int(self.config["head"]["hidden_width"])
            events.append(prune_added(model, optimizer, original_width, count))
            after = evaluate(model, *self.splits[split], batch)
            params = learned_parameters(model)
            if model.get_age() != 0:
                raise AssertionError("Live inference unexpectedly trained the model")
            old = {row["sample_id"]: row for row in reference["cases"]}
            common = {"parent_checkpoint_sha256": self.report["runs"][f"{seed}-{arm}"]["parent_checkpoint_sha256"],
                      "case_manifest_sha256": digest(self.splits[split][1]),
                      "preprocessing_sha256": self.preprocessing_sha256,
                      "evaluation_split": split, "training_steps": 0 if stage == "immediate" else int(hp["training_steps_to_do"]),
                      "optimizer_policy": self.config["optimizer_at_fork"]}
            comparison = wl.compare_predictions({**common, "run_id": request_id + "-reference", "cases": reference["cases"]},
                                                 {**common, "run_id": request_id, "cases": after["cases"]})
            examples = []
            candidates = sorted(after["cases"], key=lambda r: (old[r["sample_id"]]["prediction"] == r["prediction"], r["sample_id"]))
            for row in candidates:
                if row["sample_id"] not in self.media:
                    continue
                left = old[row["sample_id"]]
                examples.append({"id": row["sample_id"], "label": row["label"], "group": row["group"],
                                 "before_prediction": left["prediction"], "after_prediction": row["prediction"],
                                 "before_confidence": left["confidence"], "after_confidence": row["confidence"],
                                 "image": self.media[row["sample_id"]]})
                if len(examples) == 6:
                    break
            result = {"request_id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(), "request": data,
                      "runtime": "live CPU inference using real WeightsLab edits; frozen ViT feature cache",
                      "arm": arm, "reference_arm": control_arm, "checkpoint_sha256": checkpoint_sha,
                      "reference_checkpoint_sha256": reference_sha,
                      "source_commit": self.config["source_commit"], "events": events,
                      "live_source_sha256": sha256(Path(__file__)),
                      "before": summary(before), "after": summary(after), "reference": summary(reference),
                      "max_reference_logit_delta": max_logit_delta(reference, after),
                      "max_pruning_logit_delta": max_logit_delta(before, after),
                      "shape": {"hidden": list(params["0"]["weight"].shape), "classifier": list(params["2"]["weight"].shape)},
                      "comparison": {key: comparison[key] for key in ("overall", "groups")},
                      "examples": examples, "elapsed_seconds": time.monotonic() - started}
            audit = {key: value for key, value in result.items() if key != "examples"}
            with (self.output / "audit.jsonl").open("a") as stream:
                stream.write(json.dumps(audit, allow_nan=False) + "\n")
            return result


def serve(lab, port):
    token = secrets.token_urlsafe(32)
    hosts = {f"localhost:{port}", f"127.0.0.1:{port}"}
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            encoded = body.encode() if isinstance(body, str) else json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.headers.get("Host") not in hosts:
                return self.send(403, {"error": "Loopback host required"})
            if self.path == "/":
                html = Path(__file__).with_name("live_growth_lab.html").read_text().replace("__REQUEST_TOKEN__", token)
                return self.send(200, html, "text/html; charset=utf-8")
            if self.path == "/api/meta":
                return self.send(200, {"seeds": lab.config["seeds"], "target_group": lab.report["protocol"]["target_group"],
                                       "protocol_sha256": lab.report["protocol_sha256"], "source_commit": lab.config["source_commit"]})
            self.send(404, {"error": "Not found"})

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (self.headers.get("Host") not in hosts or self.headers.get("X-Lab-Token") != token
                    or (origin is not None and origin not in {f"http://{host}" for host in hosts})):
                return self.send(403, {"error": "Same-origin request token required"})
            if self.path != "/api/edit":
                return self.send(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 4096:
                    raise ValueError("Invalid request size")
                result = lab.execute(json.loads(self.rfile.read(size)))
            except (ValueError, TypeError, KeyError) as error:
                return self.send(400, {"error": str(error)})
            except Exception as error:
                print(f"Live edit failed: {error}", flush=True)
                return self.send(500, {"error": "Live evaluation failed; see server log"})
            self.send(200, result)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Live editing lab: http://localhost:{port}/ (loopback only)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--followup-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--demo-html", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8878)
    args = parser.parse_args()
    serve(Lab(args.parent_dir, args.followup_dir, args.cache, args.demo_html, args.output), args.port)
