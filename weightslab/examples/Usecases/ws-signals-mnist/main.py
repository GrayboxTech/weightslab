"""Per-sample signals on MNIST — readable, zero save_signals, with train + eval.

Per-step user code is just the watched loss. Everything else is a @wl.signal:
  entropy    from ctx.logits when the loss fires
  loss_norm  reactive, from the logged loss
  hardness   reactive, from loss + entropy

loss_shape comes from WeightsLab's built-in classifier, applied on report write
(wl.write_dataframe(loss_shape_signal=...)) — no classifier code lives here.

Universal loss: the watched crit runs on the test split each epoch too, so test
samples get a loss trajectory and a shape as well.

Env: WL_STRESS_EPOCHS, WL_STRESS_OUT.
"""
import os, time, gc, json, shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import weightslab as wl
from weightslab.components.global_monitoring import guard_training_context, guard_testing_context

LOSS = "loss_sample"
OUT = os.environ.get("WL_STRESS_OUT", "/tmp/wl_stress")
EPOCHS = int(os.environ.get("WL_STRESS_EPOCHS", "10"))
BATCH = 64


def rss_gb():
    try:
        for ln in open("/proc/self/status"):
            if ln.startswith("VmRSS:"):
                return int(ln.split()[1]) / (1024 * 1024)
    except Exception:
        return -1.0


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.net(x)


class MNISTIdx(Dataset):
    """Yields (image, uid, label). uid namespaced by split so train/test don't
    collide in the shared ledger; fast_get_label skips decode at ledger init."""
    def __init__(self, root, train, base):
        self.m = datasets.MNIST(root, train=train, download=True, transform=None)
        self.t = transforms.ToTensor(); self.base = base

    def __len__(self):
        return len(self.m)

    def __getitem__(self, i):
        img, lab = self.m[i]
        return self.t(img), self.base + i, lab

    def fast_get_label(self, i):
        return int(self.m.targets[i])


def log(msg):
    print(msg, flush=True)


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT + "/wl_logs", exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    metrics = open(OUT + "/metrics.jsonl", "w")

    # vanilla baseline (plain PyTorch) for the overhead comparison
    vb = SmallCNN().to(dev); vo = optim.Adam(vb.parameters(), lr=0.01); vc = nn.CrossEntropyLoss()
    xb = torch.randn(BATCH, 1, 28, 28, device=dev); yb = torch.randint(0, 10, (BATCH,), device=dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for _ in range(10):
        vo.zero_grad(); vc(vb(xb), yb).backward(); vo.step()
    sync(); t0 = time.perf_counter()
    for _ in range(50):
        vo.zero_grad(); vc(vb(xb), yb).backward(); vo.step()
    sync(); vanilla_ms = 1000 * (time.perf_counter() - t0) / 50
    del vb, vo; gc.collect()
    log(f"[run] vanilla baseline = {vanilla_ms:.2f} ms/step (dev {dev})")

    # WeightsLab setup: both splits tracked, watched loss, signals
    hp = {"experiment_name": "signals-mnist", "device": str(dev), "root_log_dir": OUT + "/wl_logs",
          "serving_grpc": False, "serving_cli": False, "ledger_flush_max_rows": 8192,
          "ledger_enable_h5_persistence": False, "experiment_dump_to_train_steps_ratio": 10_000_000,
          "data": {"train_loader": {"batch_size": BATCH, "shuffle": True}}}
    wl.watch_or_edit(hp, flag="hyperparameters", defaults=hp)
    train_ds = MNISTIdx(OUT + "/data", train=True, base=0)
    test_ds = MNISTIdx(OUT + "/data", train=False, base=1_000_000)
    model = wl.watch_or_edit(SmallCNN().to(dev), flag="model", device=dev)
    opt = wl.watch_or_edit(optim.Adam(model.parameters(), lr=0.01), flag="optimizer")
    loader = wl.watch_or_edit(train_ds, flag="data", loader_name="train_loader",
                              batch_size=BATCH, shuffle=True, is_training=True, preload_labels=True)
    test_loader = wl.watch_or_edit(test_ds, flag="data", loader_name="test_loader",
                                   batch_size=256, shuffle=False, is_training=False, preload_labels=True)
    crit = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                            flag="loss", signal_name=LOSS, per_sample=True, log=True)

    @wl.signal(name="sig/entropy", subscribe_to=LOSS, batched=True)
    def entropy(b):
        p = torch.softmax(b.logits, 1)
        return (-(p * (p + 1e-12).log()).sum(1)).detach().cpu().numpy()

    @wl.signal(name="sig/loss_norm", inputs=[LOSS], batched=True)
    def loss_norm(b):
        return b.inputs[LOSS] / (float(np.mean(b.inputs[LOSS])) + 1e-8)

    @wl.signal(name="sig/hardness", inputs=[LOSS, "sig/entropy"], batched=True)
    def hardness(b):
        return b.inputs[LOSS] * b.inputs["sig/entropy"]

    wl.serve(serving_grpc=False, serving_cli=False)
    wl.start_training(timeout=0)
    log(f"[run] tracked train={len(train_ds)} test={len(test_ds)}")

    def test_eval():
        """Universal loss: watched crit over the test split each epoch."""
        with torch.no_grad():
            for tb in test_loader:
                ti, tid, tl = tb[0].to(dev), tb[1], tb[2].to(dev)
                with guard_testing_context:
                    tlg = model(ti)
                    crit(tlg, tl, batch_ids=tid, preds=tlg.argmax(1, keepdim=True))

    step_times, gstep, t_run = [], 0, time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        ep_ms = []
        for img, ids, lab in loader:
            img, lab = img.to(dev), lab.to(dev)
            sync(); ts = time.perf_counter()
            with guard_training_context:
                opt.zero_grad()
                logits = model(img)
                # only per-step call: the watched loss logs loss_sample and fires
                # the @wl.signal chain. No save_signals.
                crit(logits, lab, batch_ids=ids, preds=logits.argmax(1, keepdim=True)).mean().backward()
                opt.step()
            sync(); ep_ms.append(1000 * (time.perf_counter() - ts))
            gstep += 1
        test_eval()
        wl_ms = float(np.mean(ep_ms)); step_times += ep_ms
        rec = {"epoch": ep, "gstep": gstep, "wl_ms": round(wl_ms, 2), "vanilla_ms": round(vanilla_ms, 2),
               "rss_gb": round(rss_gb(), 2), "elapsed_s": round(time.perf_counter() - t_run, 1)}
        metrics.write(json.dumps(rec) + "\n"); metrics.flush()
        log(f"[run] ep {ep:3d}/{EPOCHS} | WL {wl_ms:6.2f} ms vs vanilla {vanilla_ms:.2f} "
            f"= +{100*(wl_ms/vanilla_ms-1):.0f}% | RSS {rec['rss_gb']:.2f} GB | {rec['elapsed_s']:.0f}s")

    import pandas as pd
    # loss_shape comes from WeightsLab's built-in classifier, run on report write.
    path = wl.write_dataframe(OUT + "/report.csv", format="csv",
                              columns=["signals", "tags"], loss_shape_signal=LOSS)
    df = pd.read_csv(path)
    sc = [c for c in df.columns if c.endswith("loss_shape")][0]
    med = float(np.median(step_times))
    metrics.close()
    log("\n[run] ===== SUMMARY =====")
    log(f"[run] vanilla {vanilla_ms:.2f} ms | WL median {med:.2f} ms/step (+{100*(med/vanilla_ms-1):.0f}%)")
    log(f"[run] report {len(df)} rows | loss_shape covered {df[sc].notna().sum()}/{len(df)}")
    log(f"[run] shape distribution: {df[sc].value_counts(dropna=True).to_dict()}")
    log(f"[run] report -> {path}")


if __name__ == "__main__":
    main()
