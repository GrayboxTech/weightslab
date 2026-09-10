"""Integration check: prove WeightsLab is fully wired for this tabular example.

Unlike ``test_fraud_detection.py`` (pure PyTorch/pandas, offline), this drives
the WeightsLab stack end-to-end — tracked dataloaders, watched loss/metrics,
the gRPC server, and the per-sample ledger — then asserts the sample dataframe
the UI reads contains, for tabular mode:

  * one row per sample (what you scroll/sort in the grid & List view),
  * every transaction feature as its own column (from ``get_items`` metadata),
  * ``target`` + ``prediction`` populated (so sort/filter/histograms work),
  * a per-sample training/eval loss column,

and that the gRPC server is actually listening (so Weights Studio can connect
live during the run).

Uses a small synthetic CSV with the real dataset's exact schema (same as
``test_fraud_detection.py``) rather than the real 284k-row download, so this
check runs offline/in CI without Kaggle credentials.

Run:  python verify_integration.py
Requires weightslab installed (not a pure-unit test).
"""

import glob
import os
import socket
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Precision, Recall, F1Score, AveragePrecision

sys.path.insert(0, os.path.dirname(__file__))

import weightslab as wl  # noqa: E402

from utils.data import (  # noqa: E402
    FEATURE_NAMES,
    NUM_FEATURES,
    compute_class_weights,
    load_creditcard_fraud,
)
from utils.kaggle_data import CSV_FILENAME  # noqa: E402
from utils.model import FraudMLP  # noqa: E402


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_synthetic_creditcard_csv(path, n_samples=1200, fraud_rate=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_samples * fraud_rate)))
    n_legit = n_samples - n_fraud

    def _rows(n, fraud):
        data = {"Time": rng.uniform(0, 172792, size=n)}
        for i in range(1, 29):
            shift = 2.5 if fraud and i <= 4 else 0.0
            data[f"V{i}"] = rng.normal(shift, 1.0, size=n)
        data["Amount"] = rng.gamma(2.0, 150.0 if fraud else 60.0, size=n)
        data["Class"] = np.full(n, int(fraud))
        return pd.DataFrame(data)

    df = pd.concat([_rows(n_legit, False), _rows(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def main() -> int:
    device = torch.device("cpu")
    log_dir = tempfile.mkdtemp(prefix="wl_fraud_verify_")
    data_dir = tempfile.mkdtemp(prefix="wl_fraud_verify_data_")
    csv_path = _make_synthetic_creditcard_csv(os.path.join(data_dir, CSV_FILENAME))

    grpc_port = _free_port()
    # The gRPC server reads its port from GRPC_BACKEND_PORT (same knob the E2E
    # harness uses); set it before serve() so we bind to a free ephemeral port.
    os.environ["GRPC_BACKEND_PORT"] = str(grpc_port)

    parameters = {
        "experiment_name": "fraud_verify",
        "device": "cpu",
        "root_log_dir": log_dir,
        "serving_grpc": True,
        "grpc_port": grpc_port,
        "is_training": True,
    }
    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    model = wl.watch_or_edit(FraudMLP(in_features=NUM_FEATURES, num_classes=2).to(device),
                             flag="model", device=device)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=0.005), flag="optimizer")

    train_ds, test_ds = load_creditcard_fraud(
        csv_path=csv_path, cache_dir=os.path.join(data_dir, ".cache"),
        seed=0, test_size=0.2, oversample_fraud_ratio=0.2,
    )

    train_loader = wl.watch_or_edit(
        train_ds, flag="data", loader_name="train_loader", batch_size=64, shuffle=True,
        is_training=True, compute_hash=False, preload_labels=True, preload_metadata=True,
        enable_h5_persistence=False)
    test_loader = wl.watch_or_edit(
        test_ds, flag="data", loader_name="test_loader", batch_size=128, shuffle=False,
        is_training=False, compute_hash=False, preload_labels=True, preload_metadata=True,
        enable_h5_persistence=False)

    class_weights = torch.tensor(compute_class_weights(train_ds.labels.numpy(), cap=20.0))
    train_crit = wl.watch_or_edit(nn.CrossEntropyLoss(weight=class_weights, reduction="none"),
                                  flag="loss", signal_name="train-loss-CE", log=True)
    test_crit = wl.watch_or_edit(nn.CrossEntropyLoss(weight=class_weights, reduction="none"),
                                 flag="loss", signal_name="test-loss-CE", log=True)
    metrics = {
        "precision": wl.watch_or_edit(Precision(task="binary"), flag="metric",
                                       signal_name="metric-Precision", log=True),
        "recall": wl.watch_or_edit(Recall(task="binary"), flag="metric",
                                    signal_name="metric-Recall", log=True),
        "f1": wl.watch_or_edit(F1Score(task="binary"), flag="metric",
                                signal_name="metric-F1", log=True),
        "ap": wl.watch_or_edit(AveragePrecision(task="binary"), flag="metric",
                                signal_name="metric-AveragePrecision", log=True),
    }

    wl.serve(serving_grpc=True, grpc_port=grpc_port)
    wl.start_training()

    # ---- drive a few real training steps ----
    for _ in range(120):
        with wl.guard_training_context:
            inputs, ids, labels = next(train_loader)
            optimizer.zero_grad()
            out = model(inputs)
            preds = out.argmax(dim=1, keepdim=True)
            loss = train_crit(out, labels, batch_ids=ids, preds=preds).mean()
            loss.backward()
            optimizer.step()

    # ---- one full eval pass (populates test loss/prediction per sample) ----
    for inputs, ids, labels in test_loader:
        with wl.guard_testing_context:
            out = model(inputs)
            preds = out.argmax(dim=1, keepdim=True)
            probs = torch.softmax(out, dim=1)[:, 1]
            test_crit(out, labels, batch_ids=ids, preds=preds)
            metrics["precision"].update(preds.view(-1), labels)
            metrics["recall"].update(preds.view(-1), labels)
            metrics["f1"].update(preds.view(-1), labels)
            metrics["ap"].update(probs, labels)

    wl.drain_signals()

    # ---- dump the ledger dataframe the UI reads and inspect it ----
    wl.write_dataframe(path=log_dir, format="csv")
    csvs = glob.glob(os.path.join(log_dir, "*dataframe*.csv"))
    assert csvs, f"no dataframe CSV written to {log_dir}"
    with open(csvs[0], "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
        rows = [ln for ln in fh.read().splitlines() if ln.strip()]

    cols = set(c.strip().strip('"') for c in header)
    n_rows = len(rows)
    expected_rows = len(train_ds) + len(test_ds)

    print("\n=== Ledger dataframe ===")
    print(f"rows: {n_rows} (expected ~{expected_rows})")
    print(f"columns ({len(cols)}): {sorted(cols)}")

    # ---- assertions ----
    problems = []
    if n_rows < expected_rows * 0.9:
        problems.append(f"expected ~{expected_rows} sample rows, got {n_rows}")

    missing_features = [f for f in FEATURE_NAMES if f not in cols]
    if missing_features:
        problems.append(f"missing feature columns: {missing_features}")

    for required in ("target", "prediction"):
        if required not in cols:
            problems.append(f"missing '{required}' column")

    if not any("loss" in c.lower() for c in cols):
        problems.append("no per-sample loss column found")

    # gRPC server must be reachable so Weights Studio can connect live.
    # It starts on a background thread, so poll for a few seconds.
    import time as _time
    grpc_up = False
    for _ in range(20):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        if sock.connect_ex(("127.0.0.1", grpc_port)) == 0:
            sock.close()
            grpc_up = True
            break
        sock.close()
        _time.sleep(0.5)
    if not grpc_up:
        problems.append(f"gRPC server not listening on {grpc_port}")
    else:
        print(f"gRPC server listening on 127.0.0.1:{grpc_port}  ✓")

        # Real gRPC round-trip: the input the model was fed (the feature vector)
        # must reach the UI as raw_data of type 'vector' carrying the values.
        try:
            import grpc
            from weightslab.proto import experiment_service_pb2 as pb2
            from weightslab.proto import experiment_service_pb2_grpc as pb2_grpc

            channel = grpc.insecure_channel(f"127.0.0.1:{grpc_port}")
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = pb2_grpc.ExperimentServiceStub(channel)
            resp = stub.GetDataSamples(pb2.DataSamplesRequest(
                start_index=0, records_cnt=5, include_raw_data=True,
                resize_width=0, resize_height=0), timeout=20)

            records = list(resp.data_records)
            raw = None
            for rec in records:
                for st in rec.data_stats:
                    if st.name == "raw_data":
                        raw = st
                        break
                if raw is not None:
                    break

            if raw is None:
                problems.append("gRPC GetDataSamples returned no raw_data stat")
            elif raw.type != "vector":
                problems.append(f"raw_data.type is '{raw.type}', expected 'vector'")
            elif len(raw.value) != NUM_FEATURES:
                problems.append(
                    f"raw_data carries {len(raw.value)} values, expected {NUM_FEATURES}")
            else:
                print(f"gRPC GetDataSamples: raw_data type='vector', "
                      f"{len(raw.value)} feature values reached the UI  ✓")
                print(f"  sample values: {[round(v, 3) for v in raw.value[:6]]} …")
            channel.close()
        except Exception as e:
            problems.append(f"gRPC GetDataSamples failed: {e}")

    if problems:
        print("\n VERIFY FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\n VERIFY PASSED: WeightsLab is fully wired for tabular mode "
          "(per-sample rows + feature columns + target/prediction/loss + live gRPC).")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard-exit so background serving threads don't keep the process alive.
    os._exit(code)
