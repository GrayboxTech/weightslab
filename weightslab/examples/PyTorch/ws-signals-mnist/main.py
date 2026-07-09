"""Per-sample signals on MNIST — the idiomatic WeightsLab way, zero save_signals.

The whole per-step user code is the watched loss. Everything else is a
``@wl.signal``:

* the watched loss (``crit``) logs a per-sample ``loss_sample`` trajectory;
* ``sig/entropy`` is computed from ``ctx.logits`` when the loss fires (a
  logit-derived signal — no manual push);
* ``sig/loss_norm`` / ``sig/hardness`` are reactive, derived from logged signals;
* trajectory shapes are classified once at the end from the loss history.

Run:  python main.py --outdir ./out
"""
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import weightslab as wl
from weightslab.components.global_monitoring import guard_training_context

LOSS = "loss_sample"


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
    """Yields (image, uid, label). fast_get_label skips image decode at ledger init."""
    def __init__(self, root, n):
        self.m = datasets.MNIST(root, train=True, download=True, transform=None)
        self.t = transforms.ToTensor(); self.n = n

    def __len__(self):
        return min(len(self.m), self.n)

    def __getitem__(self, i):
        img, lab = self.m[i]
        return self.t(img), i, lab

    def fast_get_label(self, i):
        return int(self.m.targets[i])


def classify_shape(values):
    """Per-sample loss trajectory -> shape label (None if < 5 points)."""
    y = np.asarray(values, dtype=float)
    if y.size < 5:
        return None
    n = y.size
    rng = max(float(y.max() - y.min()), 1e-8)
    drop = (float(y[0]) - float(y[-1])) / (abs(float(y[0])) + 1e-8)
    cv = float(y.std()) / (abs(float(y.mean())) + 1e-8)
    argmin = int(np.argmin(y))
    rebound = (float(y[-1]) - float(y.min())) / rng
    tail = y[int(0.6 * n):]
    tail_flat = float(tail.std()) / (abs(float(tail.mean())) + 1e-8) < 0.1
    if 0.2 * n < argmin < 0.8 * n and rebound > 0.3:
        return "U_Shape"
    if drop > 0.4:
        return "monotonic"
    if drop > 0.15 and tail_flat:
        return "plateaued"
    if float(np.diff(y).max()) / rng > 0.5:
        return "Spiked"
    if cv > 0.5:
        return "high_variance"
    return "Flat_high"


def finalize_shapes():
    """End sweep: classify each sample's loss trajectory into a categorical tag."""
    series = defaultdict(list)
    for sid, step, val, _ in wl.query_signal_history(LOSS):
        series[sid].append((step, val))
    by_label = defaultdict(list)
    for sid, pts in series.items():
        label = classify_shape([v for _, v in sorted(pts)])
        if label is not None:
            by_label[label].append(sid)
    for label, sids in by_label.items():
        wl.set_categorical_tag(sids, "loss_shape", label)
    return {k: len(v) for k, v in by_label.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./out")
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    hp = {"experiment_name": "ws-signals-mnist", "device": str(dev),
          "root_log_dir": args.outdir + "/wl_logs", "serving_grpc": False, "serving_cli": False,
          "ledger_flush_max_rows": 8192,
          "data": {"train_loader": {"batch_size": 64, "shuffle": True}}}
    wl.watch_or_edit(hp, flag="hyperparameters", defaults=hp)

    ds = MNISTIdx(args.outdir + "/data", args.subset)
    model = wl.watch_or_edit(SmallCNN().to(dev), flag="model", device=dev)
    opt = wl.watch_or_edit(optim.Adam(model.parameters(), lr=0.01), flag="optimizer")
    loader = wl.watch_or_edit(ds, flag="data", loader_name="train_loader",
                              batch_size=64, shuffle=True, is_training=True, preload_labels=True)
    crit = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                            flag="loss", signal_name=LOSS, per_sample=True, log=True)

    # entropy from the model output when the loss fires — a @wl.signal, no push.
    @wl.signal(name="sig/entropy", subscribe_to=LOSS, batched=True)
    def entropy(b):
        p = torch.softmax(b.logits, 1)
        return (-(p * (p + 1e-12).log()).sum(1)).detach().cpu().numpy()

    # reactive derived signals (off logged signals; no model output needed)
    @wl.signal(name="sig/loss_norm", inputs=[LOSS], batched=True)
    def loss_norm(b):
        return b.inputs[LOSS] / (float(np.mean(b.inputs[LOSS])) + 1e-8)

    @wl.signal(name="sig/hardness", inputs=[LOSS, "sig/entropy"], batched=True)
    def hardness(b):
        return b.inputs[LOSS] * b.inputs["sig/entropy"]

    wl.serve(serving_grpc=False, serving_cli=False)
    wl.start_training(timeout=0)

    for epoch in range(1, args.epochs + 1):
        for img, ids, lab in loader:
            img, lab = img.to(dev), lab.to(dev)
            with guard_training_context:
                opt.zero_grad()
                logits = model(img)
                # the only per-step call: the watched loss logs loss_sample and
                # fires the @wl.signal chain. No save_signals.
                crit(logits, lab, batch_ids=ids, preds=logits.argmax(1, keepdim=True)).mean().backward()
                opt.step()
        print(f"epoch {epoch}/{args.epochs}")

    print("trajectory-shape distribution:", finalize_shapes())
    path = wl.write_dataframe(args.outdir + "/per_sample_signals.csv",
                              format="csv", columns=["signals", "tags"])
    print("per-sample report:", path)


if __name__ == "__main__":
    main()
