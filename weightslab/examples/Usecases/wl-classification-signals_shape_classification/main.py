"""Per-sample signals on MNIST — readable, zero save_signals, with train + eval.

Parameters live in ``config.yaml`` (loaded at startup). A handful of knobs can
also be overridden by environment variables for scripted stress runs.

Per-step user code is just the watched loss. Everything else is a @wl.signal
(defined in ``utils/signals.py``):
  entropy    from ctx.logits when the loss fires
  loss_norm  reactive, from the logged loss
  hardness   reactive, from loss + entropy
  loss_shape reactive, classifies each sample's loss trajectory (live signal,
             not an end-of-run function). Reads history, so it's throttled by
             shape_every (1 = every step, full coverage; higher = cheaper).

Universal loss: the watched crit runs on the test split each epoch too, so test
samples get a loss trajectory and a shape as well.

Env overrides (else the config.yaml value is used):
  WL_STRESS_EPOCHS       (int)  number of training epochs.
  WL_STRESS_OUT          (str)  output dir; wiped and recreated on start. Holds
                         data/, wl_logs/, metrics.jsonl and report.csv.
  WL_SHAPE_EVERY         (int)  throttle for the sig/loss_shape signal, which
                         reads history and is therefore costly. 1 = compute every
                         step (full coverage); higher = every N steps (cheaper).
  WL_QUERY_CACHE_MAXSIZE (int)  backend tuning knob (read in
                         weightslab.backend.logger). Sets the LRU maxsize of the
                         per-sample query cache the ledger uses when serving
                         signals; larger values cache more distinct per-sample
                         queries at the cost of memory.
"""
import os, time, gc, json, shutil
import yaml
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

import weightslab as wl

from collections import Counter

from weightslab import guard_training_context, guard_testing_context

from utils.model import SmallCNN
from utils.data import MNISTIdx
from utils.criterions import SHAPES
from utils.signals import register_signals


# =============================================================================
# Configuration (config.yaml + env overrides, see module docstring)
# =============================================================================
def load_config():
    """Load config.yaml next to this file and apply defaults for missing keys."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
    else:
        cfg = {}

    cfg.setdefault("experiment_name", "signals-mnist")
    cfg.setdefault("device", "auto")
    cfg.setdefault("epochs", 10)
    cfg.setdefault("out", "/tmp/wl_stress")
    cfg.setdefault("batch_size", 64)
    cfg.setdefault("loss_signal_name", "loss_sample")
    cfg.setdefault("shape_every", 1)
    cfg.setdefault("serving_grpc", False)
    cfg.setdefault("serving_cli", False)
    cfg.setdefault("ledger_flush_max_rows", 8192)
    cfg.setdefault("ledger_enable_h5_persistence", False)
    cfg.setdefault("experiment_dump_to_train_steps_ratio", 10_000_000)
    cfg.setdefault("query_cache_maxsize", 2048)
    cfg.setdefault("optimizer", {}).setdefault("lr", 0.01)
    data = cfg.setdefault("data", {})
    data.setdefault("train_loader", {}).setdefault("batch_size", cfg["batch_size"])
    cfg["data"]["train_loader"].setdefault("shuffle", True)
    data.setdefault("test_loader", {}).setdefault("batch_size", 256)
    cfg["data"]["test_loader"].setdefault("shuffle", False)

    # Env overrides for scripted stress runs (env wins over the yaml value).
    cfg["epochs"] = int(os.environ.get("WL_STRESS_EPOCHS", cfg["epochs"]))
    cfg["out"] = os.environ.get("WL_STRESS_OUT", cfg["out"])
    cfg["shape_every"] = int(os.environ.get("WL_SHAPE_EVERY", cfg["shape_every"]))
    cfg["min_step"] = int(os.environ.get("WL_MIN_STEP", cfg["min_step"]))
    cfg["query_cache_maxsize"] = int(
        os.environ.get("WL_QUERY_CACHE_MAXSIZE", cfg["query_cache_maxsize"]))
    return cfg


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


# =============================================================================
# Helpers
# =============================================================================
def rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / (1024 * 1024)
    except Exception:
        return -1.0


def log(msg):
    print(msg, flush=True)


def vanilla_baseline(dev, batch, lr, sync):
    """Plain-PyTorch step time (ms) for the WeightsLab overhead comparison."""
    vb = SmallCNN().to(dev); vo = optim.Adam(vb.parameters(), lr=lr); vc = nn.CrossEntropyLoss()
    xb = torch.randn(batch, 1, 28, 28, device=dev); yb = torch.randint(0, 10, (batch,), device=dev)
    for _ in range(10):
        vo.zero_grad(); vc(vb(xb), yb).backward(); vo.step()
    sync(); t0 = time.perf_counter()
    for _ in range(50):
        vo.zero_grad(); vc(vb(xb), yb).backward(); vo.step()
    sync(); vanilla_ms = 1000 * (time.perf_counter() - t0) / 50
    del vb, vo; gc.collect()
    return vanilla_ms


# =============================================================================
# Train / Test loops (classification, using watcher-wrapped loaders)
# =============================================================================
def train(loader, model, opt, crit, dev, sync):
    """One training epoch. Returns the list of per-step wall times (ms).

    The only per-step call is the watched loss: it logs loss_sample and fires
    the @wl.signal chain. No save_signals.
    """
    ep_ms = []
    for img, ids, lab in loader:
        img, lab = img.to(dev), lab.to(dev)
        sync(); ts = time.perf_counter()
        with guard_training_context:
            opt.zero_grad()
            logits = model(img)
            crit(logits, lab, batch_ids=ids, preds=logits.argmax(1, keepdim=True)).mean().backward()
            opt.step()
        sync(); ep_ms.append(1000 * (time.perf_counter() - ts))
    return ep_ms


def test(test_loader, model, crit, dev):
    """Universal loss: run the watched crit over the whole test split (one pass).

    Test samples get a loss trajectory and a shape too.
    """
    with torch.no_grad():
        for tb in test_loader:
            ti, tid, tl = tb[0].to(dev), tb[1], tb[2].to(dev)
            with guard_testing_context:
                tlg = model(ti)
                crit(tlg, tl, batch_ids=tid, preds=tlg.argmax(1, keepdim=True))


# =============================================================================
# Main
# =============================================================================
def main():
    # --- 1) Config: load, resolve device, prepare output dir ---
    cfg = load_config()
    LOSS = cfg["loss_signal_name"]
    OUT = cfg["out"]
    EPOCHS = cfg["epochs"]
    SHAPE_EVERY = cfg["shape_every"]
    SHAPE_MIN_STEP = cfg["min_step"]
    BATCH = int(cfg["data"]["train_loader"]["batch_size"])
    LR = float(cfg["optimizer"]["lr"])
    # Backend reads this from the environment when it builds the query cache.
    os.environ["WL_QUERY_CACHE_MAXSIZE"] = str(cfg["query_cache_maxsize"])

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT + "/wl_logs", exist_ok=True)
    dev = resolve_device(cfg["device"])
    torch.manual_seed(0)
    metrics = open(OUT + "/metrics.jsonl", "w")
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)

    # --- 2) Vanilla baseline (plain PyTorch) for the overhead comparison ---
    vanilla_ms = vanilla_baseline(dev, BATCH, LR, sync)
    log(f"[run] vanilla baseline = {vanilla_ms:.2f} ms/step (dev {dev})")

    # --- 3) WeightsLab setup: both splits tracked, watched loss ---
    hp = {"experiment_name": cfg["experiment_name"], "device": str(dev),
          "root_log_dir": OUT + "/wl_logs",
          "serving_grpc": cfg["serving_grpc"], "serving_cli": cfg["serving_cli"],
          "ledger_flush_max_rows": cfg["ledger_flush_max_rows"],
          "ledger_enable_h5_persistence": cfg["ledger_enable_h5_persistence"],
          "experiment_dump_to_train_steps_ratio": cfg["experiment_dump_to_train_steps_ratio"],
          "data": {"train_loader": {"batch_size": BATCH,
                                    "shuffle": cfg["data"]["train_loader"]["shuffle"]}}}
    wl.watch_or_edit(hp, flag="hyperparameters", defaults=hp)
    train_ds = MNISTIdx(OUT + "/data", train=True, base=0)
    test_ds = MNISTIdx(OUT + "/data", train=False, base=1_000_000)
    model = wl.watch_or_edit(SmallCNN().to(dev), flag="model", device=dev)
    opt = wl.watch_or_edit(optim.Adam(model.parameters(), lr=LR), flag="optimizer")
    loader = wl.watch_or_edit(train_ds, flag="data", loader_name="train_loader",
                              batch_size=BATCH,
                              shuffle=cfg["data"]["train_loader"]["shuffle"],
                              is_training=True, preload_labels=True)
    test_loader = wl.watch_or_edit(test_ds, flag="data", loader_name="test_loader",
                                   batch_size=int(cfg["data"]["test_loader"]["batch_size"]),
                                   shuffle=cfg["data"]["test_loader"]["shuffle"],
                                   is_training=False, preload_labels=True)
    crit = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                            flag="loss", signal_name=LOSS, per_sample=True, log=True)

    # --- 4) Per-sample signals (the @wl.signal chain, see utils/signals.py) ---
    register_signals(LOSS, SHAPE_EVERY, SHAPE_MIN_STEP)

    # --- 5) Serve + launch training ---
    wl.serve(serving_grpc=cfg["serving_grpc"], serving_cli=cfg["serving_cli"])
    wl.start_training(timeout=5)  # Let WeightsLab Initialize
    log(f"[run] tracked train={len(train_ds)} test={len(test_ds)} | shape_every={SHAPE_EVERY}")

    # --- 6) Train / eval loop ---
    step_times, gstep, t_run = [], 0, time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        ep_ms = train(loader, model, opt, crit, dev, sync)
        gstep += len(ep_ms)
        test(test_loader, model, crit, dev)
        wl_ms = float(np.mean(ep_ms)); step_times += ep_ms
        rec = {"epoch": ep, "gstep": gstep, "wl_ms": round(wl_ms, 2), "vanilla_ms": round(vanilla_ms, 2),
               "rss_gb": round(rss_gb(), 2), "elapsed_s": round(time.perf_counter() - t_run, 1)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush()
        log(f"[run] ep {ep:3d}/{EPOCHS} | WL {wl_ms:6.2f} ms vs vanilla {vanilla_ms:.2f} "
            f"= +{100*(wl_ms/vanilla_ms-1):.0f}% | RSS {rec['rss_gb']:.2f} GB | {rec['elapsed_s']:.0f}s")

    # --- 7) Report + summary ---
    path = wl.write_dataframe(OUT + "/report.csv", format="csv", columns="signals")
    df = pd.read_csv(path)
    sc = [c for c in df.columns if c.endswith("sig/loss_shape")][0]
    dist = Counter(SHAPES[int(v)] for v in df[sc].dropna() if v >= 0)
    med = float(np.median(step_times))
    metrics.close()
    log("\n[run] ===== SUMMARY =====")
    log(f"[run] vanilla {vanilla_ms:.2f} ms | WL median {med:.2f} ms/step (+{100*(med/vanilla_ms-1):.0f}%)")
    log(f"[run] report {len(df)} rows | loss_shape covered {df[sc].notna().sum()}/{len(df)}")
    log(f"[run] shape distribution: {dict(dist)}")
    log(f"[run] report -> {path}")


if __name__ == "__main__":
    main()
