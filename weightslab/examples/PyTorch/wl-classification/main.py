import itertools
import math
import os
import random
import ssl
import time
import logging
import tempfile

# Windows SSL fix: some Windows cert stores contain malformed ASN1 certs that
# crash ssl.create_default_context(). Fall back to unverified only when broken.
try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import yaml
import tqdm
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import weightslab as wl


# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# =============================================================================
# Custom MNIST Dataset with Filepath Metadata
# =============================================================================
class MNISTCustomDataset(Dataset):
    """
    Custom MNIST dataset that includes filepath metadata for each image.

    Returns tuples of (image, label, filepath) where filepath is stored
    as metadata that can be tracked by WeightsLab.
    """

    def __init__(self, root, train=True, download=False, transform=None, max_samples=None):
        """
        Args:
            root (str): Root directory where MNIST data is stored
            train (bool): If True, use training data; else use test data
            download (bool): If True, download the data if not present
            transform (callable, optional): Optional transform to be applied on images
        """

        # Load the standard MNIST dataset
        try:
            self.mnist = datasets.MNIST(
                root=root,
                train=train,
                download=download,
                transform=None # We'll apply transform manually to track filepath
            )
        except RuntimeError as e:
            logger.error(f"Error loading MNIST dataset: {e}")
            self.mnist = datasets.MNIST(
                root=root,
                train=train,
                download=True,
                transform=None # We'll apply transform manually to track filepath
            )
        self.transform = transform
        self.train = train
        self.root = root
        self.max_samples = max_samples

        # Build filepath mapping for each sample
        self._build_filepath_mapping()

    def _build_filepath_mapping(self):
        """Build a mapping of sample index to filepath."""

        self.filepaths = {}

        # For each index, construct a meaningful filepath
        # MNIST doesn't have original individual files, so we create virtual paths
        for idx in range(len(self.mnist)):
            if self.max_samples != None and idx >= self.max_samples:
                break
            label = self.mnist.targets[idx].item() if hasattr(self.mnist.targets[idx], 'item') else self.mnist.targets[idx]
            split = 'train' if self.train else 'test'

            # Create a virtual filepath that identifies the image
            virtual_path = os.path.join(
                'MNIST',
                'processed',
                split,
                f'class_{label}',
                f'sample_{idx:05d}.pt'
            )
            self.filepaths[idx] = virtual_path

    def __len__(self):
        if self.max_samples != None:
            return min(len(self.mnist), self.max_samples)
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, idx, label)
        """

        image, label = self.mnist[idx]

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)

        return image, idx, label


# =============================================================================
# Model and recipe
# =============================================================================
class CNN(nn.Module):
    """2 conv + 2 fc layers (1.2M parameters), raw logits out, optional dropout.

    Raw logits, not a softmax: `nn.CrossEntropyLoss` applies its own log-softmax.
    A softmax here would be applied twice, which still trains but squashes the
    per-sample loss values -- and those values are what you sort and filter on
    in the Studio.
    """

    def __init__(self, dropout=False):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25 if dropout else 0.0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.relu3 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5 if dropout else 0.0)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        x = self.drop1(self.pool(self.relu2(self.conv2(self.relu1(self.conv1(x))))))
        return self.relu3(self.fc1(self.flatten(x)))

    def forward(self, x):
        return self.fc2(self.drop2(self.features(x)))


def gpu_augment(x, rot_deg=10.0, shift=0.1, scale=(0.9, 1.1)):
    """Random affine per image (rotation, shift, scale), on the batch, on the GPU."""
    b = x.size(0)
    ang = (torch.rand(b, device=x.device) * 2 - 1) * rot_deg * math.pi / 180
    s = torch.empty(b, device=x.device).uniform_(*scale)
    tx = (torch.rand(b, device=x.device) * 2 - 1) * shift * 2
    ty = (torch.rand(b, device=x.device) * 2 - 1) * shift * 2
    cos, sin = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def lr_at(step, total_steps, base_lr, sched):
    """const, or warmcos: linear warmup over 15% of the steps, then cosine decay to ~0."""
    if sched == "const":
        return base_lr
    warm = max(1, int(0.15 * total_steps))
    if step <= warm:
        return base_lr * (0.04 + 0.96 * step / warm)
    p = (step - warm) / max(1, total_steps - warm)
    return 1e-6 + base_lr * 0.5 * (1 + math.cos(math.pi * p))


def get_val_ids(targets, path):
    """Fixed validation split: 500 train images per class, never trained on.

    Fixed RNG seed, so the split is identical on every machine and every rerun.
    Delete the file to draw a new one.
    """
    if path and os.path.exists(path):
        return sorted(int(l) for l in open(path) if l.strip())
    y = targets.numpy() if hasattr(targets, "numpy") else np.asarray(targets)
    rng = np.random.RandomState(2026)
    ids = []
    for c in range(10):
        ids += rng.choice(np.where(y == c)[0], 500, replace=False).tolist()
    ids = sorted(int(i) for i in ids)
    if path:
        open(path, "w").write("\n".join(map(str, ids)) + "\n")
    return ids


class SubsetWithIds(Dataset):
    """A view over `base` restricted to `ids`, in the order given.

    WeightsLab numbers each split's samples contiguously, so a sample id is a
    position within the split, not an MNIST index. `ids` is the lookup back:
    the n-th sample of this loader is `ids[n]` in MNIST, which is also the n-th
    line of the matching train_ids.txt / val_ids.txt.
    """

    def __init__(self, base, ids):
        self.base = base
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.base[self.ids[i]]


# -----------------------------------------------------------------------------
# Train / Test functions
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, device, step, recipe):
    """Single training step using the tracked dataloader + watched loss.

    `recipe` carries the run's knobs: augmentation and the LR schedule.
    """

    with wl.guard_training_context:
        (inputs, ids, labels) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Augment on the GPU, on the batch
        if recipe["augment"]:
            inputs = gpu_augment(inputs)

        # LR schedule is applied per step, from the model's age
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step, recipe["total_steps"], recipe["lr"], recipe["schedule"])

        # Infer
        optimizer.zero_grad()
        preds_raw = model(inputs)

        # Preds
        if preds_raw.ndim == 1:
            preds = (preds_raw > 0.0).long()
        else:
            preds = preds_raw.argmax(dim=1, keepdim=True)

        # Loss is a watched object => pass metadata for logging/stats.
        # per_sample=True on the criterion keeps one loss value per digit per
        # visit, so each sample has a loss history you can inspect in the Studio.
        loss_batch_mlt = criterion_mlt(
            preds_raw.float(),
            labels.long(),
            batch_ids=ids,
            preds=preds
        )
        total_loss = loss_batch_mlt.mean() # Final scalar loss

        # Model
        total_loss.backward()
        optimizer.step()

    return total_loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device, test_loader_len, split="test"):
    """Full evaluation pass over one evaluation loader.

    The metric is reset first: torchmetrics accumulates across `update` calls,
    so without this each evaluation would report a running average over every
    previous evaluation instead of the current one -- and the val and test
    loaders would pollute each other's number.
    """
    losses = torch.tensor(0.0, device=device)
    metric_mlt.reset()

    for (inputs, ids, labels) in loader:
        with wl.guard_testing_context:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Infer
            outputs = model(inputs)

            # Preds
            if outputs.ndim == 1:
                preds = (outputs > 0.0).long()
            else:
                preds = outputs.argmax(dim=1, keepdim=True)

            # Compute signals
            loss_batch = criterion_mlt(
                outputs,
                labels,
                batch_ids=ids,
                preds=preds,
            )
            losses += torch.mean(loss_batch)
            metric_mlt.update(outputs, labels)

            # Per-sample accuracy: 1.0 if correct, else 0.0
            preds_flat = preds.view(-1)
            acc_per_sample = (preds_flat == labels.view(-1)).float()
            acc_reversed_per_sample = (preds_flat != labels.view(-1)).float()

            # Log per-sample metric alongside signals; persists via the storer
            signals = {
                f"{split}_metric/Accuracy_per_sample": acc_per_sample,
                f"{split}_metric/Inverse_Accuracy_per_sample": acc_reversed_per_sample,
            }
            wl.save_signals(
                preds_raw=outputs,
                targets=labels,
                batch_ids=ids,
                signals=signals,
                preds=preds,
            )

    loss = losses / test_loader_len
    metric = metric_mlt.compute() * 100

    return loss.detach().cpu().item(), metric.detach().cpu().item()


def _set_dropout(model, enabled: bool) -> None:
    """Toggle dropout between phases without rebuilding the model.

    Rebuilding would create a NEW watched model (new hash, new checkpoint
    lineage) and lose the step-0 weights we are about to restore, so the rates
    are set in place on the existing modules instead.
    """
    for name, p in (("drop1", 0.25), ("drop2", 0.5)):
        mod = getattr(model, name, None)
        if mod is not None:
            mod.p = p if enabled else 0.0


def _reload_initial_weights(wl, model, optimizer, init_state, log_dir) -> dict:
    """Put the model back to its step-0 random initialisation.

    Prefers WeightsLab's own step-0 checkpoint (what "reload at model age 0"
    means from the Studio); falls back to the in-memory snapshot taken before
    phase A. Either way the result is verified against that snapshot, so the
    number reported is a measurement and not a claim.
    """
    source = "in-memory snapshot"
    try:
        from weightslab.backend.ledgers import get_checkpoint_manager
        cm = get_checkpoint_manager()
        exp_hash = getattr(cm, "current_exp_hash", None) or cm.get_latest_hash()
        cm.load_checkpoint(exp_hash=exp_hash, load_model=False, load_weights=True,
                           load_config=False, load_data=False, target_step=0, force=True)
        source = "step-0 checkpoint"
    except Exception as exc:
        print(f"[reset] step-0 checkpoint reload unavailable ({exc}); using the snapshot", flush=True)

    def _drift():
        cur = model.state_dict()
        return max((cur[k].float() - v.float()).abs().max().item()
                   for k, v in init_state.items() if k in cur and torch.is_tensor(cur[k]))

    worst = _drift()
    if worst > 0:
        model.load_state_dict(init_state, strict=False)   # snapshot is authoritative
        after = _drift()
        source += f" + snapshot (checkpoint differed by {worst:.3g})"
        worst = after

    # Adam moments must go too: keeping them would carry phase A's gradient
    # history into a run that is supposed to start from scratch.
    optimizer.state = type(optimizer.state)()
    return {"source": source, "max_abs_diff": worst}


def _fmt_si(n: float, unit: str) -> str:
    """1199882 -> '1.2M'. Model cards quote these rounded, so match that."""
    for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}{suf}{unit}"
    return f"{n:.0f}{unit}"


def _card(title, acc, params, n_data) -> None:
    """The four-field summary, one per experiment.

    Parameters and FLOPs are identical for both runs by construction -- same
    architecture, same input size. That is the point: only NumberTrainingData
    moves, so any accuracy difference is attributable to the data, not capacity.
    """
    print(f"  {title}")
    print(f"    Accuracy:           {acc:.2f}%")
    print(f"    Parameters:         {_fmt_si(params, '')}")
    print(f"    NumberTrainingData: {n_data:,}")



def _summarise(results, n_train, goldset_size) -> None:
    """Print the comparison the experiment exists to produce."""
    a, b = results["signal"], results["goldset"]
    macs = results["flops_fwd_per_sample"]
    params = results["model_params"]
    fpv = macs * 3          # fwd + bwd, per sample-visit, in MACs

    print("\n" + "=" * 74)
    print(" MODEL CARDS")
    print("=" * 74)
    _card("full trainset  (phase A: signal run)",
          a["best"].get("test_acc", float("nan")), params, n_train)
    print()
    _card("goldset        (phase B: retrained from step-0 weights)",
          b["best"].get("test_acc", float("nan")), params, goldset_size)

    print("\n" + "=" * 74)
    print(" TRAINING COST (what actually differs)")
    print("=" * 74)
    print(f"{'':24s}{'full trainset':>18s}{'goldset':>18s}")
    rows = [
        ("training digits", f"{n_train:,}", f"{goldset_size:,}"),
        ("steps x batch", f"{a['steps']:,} x {a['batch_size']}", f"{b['steps']:,} x {b['batch_size']}"),
        ("sample visits", f"{a['sample_visits']:,}", f"{b['sample_visits']:,}"),
        ("epochs over own set", f"{a['sample_visits'] / max(1, n_train):.1f}",
                                f"{b['sample_visits'] / max(1, goldset_size):.1f}"),
        ("train TFLOPs", f"{a['sample_visits'] * fpv * 2 / 1e12:.2f}",
                         f"{b['sample_visits'] * fpv * 2 / 1e12:.2f}"),
        ("TFLOPs to best val", f"{a['best'].get('tflops', 0) * 2:.2f}",
                               f"{b['best'].get('tflops', 0) * 2:.2f}"),
        ("train seconds", f"{a['train_seconds']:.1f}", f"{b['train_seconds']:.1f}"),
        ("eval seconds", f"{a['eval_seconds']:.1f}", f"{b['eval_seconds']:.1f}"),
        ("wall seconds", f"{a['wall_seconds']:.1f}", f"{b['wall_seconds']:.1f}"),
        ("best val acc %", f"{a['best'].get('val_acc', float('nan')):.2f}",
                           f"{b['best'].get('val_acc', float('nan')):.2f}"),
        ("test @ best val %", f"{a['best'].get('test_acc', float('nan')):.2f}",
                              f"{b['best'].get('test_acc', float('nan')):.2f}"),
        ("step of best val", f"{a['best'].get('step', 0):,}", f"{b['best'].get('step', 0):,}"),
    ]
    for label, av, bv in rows:
        print(f"{label:24s}{av:>18s}{bv:>18s}")

    ep_a = n_train * fpv * 2 / 1e12
    ep_b = goldset_size * fpv * 2 / 1e12
    saved = f"{(1 - ep_b / ep_a) * 100:.1f}% less per epoch" if ep_a > 0 else "n/a"
    print("\n one epoch over its own training set:")
    print(f"   full trainset: {ep_a:.3f} TFLOPs over {n_train:,} digits")
    print(f"   goldset:       {ep_b:.3f} TFLOPs over {goldset_size:,} digits   ({saved})")
    print(" Same architecture, same per-sample FLOPs -- training cost scales purely")
    print(" with sample-visits, so the saving is in the data, not the model.")
    print("=" * 74)


# ============================================================================
# Experiment phases: signal run -> goldset -> retrain from the initial weights
# ============================================================================
def model_flops_per_sample(model, device) -> int:
    """Forward MACs for one sample, counted from the conv/linear layers.

    The model is identical in both phases, so this is a constant; what differs
    between the phases is only how many sample-visits each one pays for.
    """
    macs = {"n": 0}
    hooks = []

    def conv_hook(mod, inp, out):
        macs["n"] += out.numel() * mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]

    def lin_hook(mod, inp, out):
        macs["n"] += out.numel() * mod.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 28, 28, device=device))
    model.train(was_training)
    for h in hooks:
        h.remove()
    if macs["n"] == 0:
        raise RuntimeError(
            "FLOPs count came back 0 -- no Conv2d/Linear was visited. Pass the "
            "UNWRAPPED model (the object built before wl.watch_or_edit): a watched "
            "model is a ModelInterface whose .modules() yields only itself.")
    return int(macs["n"])


def build_goldset(wl, steps_per_pass, labels_by_id, pool_ids, gcfg):
    """Per class, the digits with the highest mean loss over passes 2..N, minus
    the noisy ones (median loss over all passes above the threshold).

    Reads the per-sample loss history WeightsLab recorded during the signal run
    -- no second inference pass, no extra bookkeeping of our own.
    """
    wl.drain_signals()
    rows = wl.query_signal_history("train-loss-CE")
    hist = pd.DataFrame(rows, columns=["sample_id", "step", "loss", "run"][:len(rows[0])]) if rows else pd.DataFrame()
    if hist.empty:
        raise SystemExit("[goldset] no train-loss-CE history recorded; cannot build the goldset")
    hist["sample_id"] = hist["sample_id"].astype(int)
    hist["loss"] = hist["loss"].astype(float)
    # "Pass n" is the n-th time THIS digit was seen, taken from its own visit
    # order rather than from step arithmetic: the recorded step numbering is not
    # guaranteed to start at 1, and a uniform shuffle is what makes the two
    # equivalent in the first place.
    hist = hist.sort_values(["sample_id", "step"], kind="stable")
    hist["pass"] = hist.groupby("sample_id").cumcount()
    per_pass = hist.groupby(["sample_id", "pass"])["loss"].last().unstack()

    noisy = per_pass.median(axis=1, skipna=True) > float(gcfg.get("noisy_median_loss", 1.0))
    first = 1 if gcfg.get("skip_first_pass", True) else 0
    n_passes = int(per_pass.shape[1])
    if n_passes <= first:
        raise SystemExit(
            f"[goldset] only {n_passes} pass(es) recorded; the rule scores passes "
            f"{first + 1}..N, so it needs at least {first + 2}. Per-sample loss over "
            f"fewer passes measures batch placement, not difficulty.")
    score = per_pass.iloc[:, first:].mean(axis=1, skipna=True)

    pool = set(int(i) for i in pool_ids)
    cand = pd.DataFrame({"score": score, "noisy": noisy.reindex(score.index, fill_value=False)})
    cand = cand[cand.index.map(lambda i: int(i) in pool) & ~cand["noisy"] & cand["score"].notna()]
    cand["label"] = [labels_by_id[int(i)] for i in cand.index]

    k = int(gcfg.get("per_class", 360))
    gold, per_class_got = [], {}
    for c in sorted(cand["label"].unique()):
        g = cand[cand["label"] == c].sort_values("score", ascending=False, kind="stable")
        picked = [int(i) for i in g.head(k).index]
        gold += picked
        per_class_got[int(c)] = len(picked)

    # A goldset that is empty, or short of its per-class quota, means the rule or
    # the history is wrong -- stop rather than retrain on a malformed subset.
    if not gold:
        raise SystemExit("[goldset] selection is empty; refusing to continue")
    short = {c: n for c, n in per_class_got.items() if n < k}
    if short:
        raise SystemExit(f"[goldset] classes below the {k}/class quota: {short}")
    if len(set(gold)) != len(gold):
        raise SystemExit("[goldset] duplicate ids in the selection")

    return sorted(gold), {"noisy": int(noisy.sum()), "scored": int(len(cand)),
                          "passes": int(per_pass.shape[1]), "per_class": per_class_got}


def run_phase(name, total_steps, recipe, eval_every, ctx):
    """Train one phase, evaluating validation on a cadence and TEST ONLY when
    validation reaches a new maximum.

    Test accuracy is therefore never used to steer anything -- it is read at
    exactly the points where validation says the model just got better, which
    is the only honest moment to look at it.
    """
    wl, model, optimizer = ctx["wl"], ctx["model"], ctx["optimizer"]
    train_loader, val_loader, test_loader = ctx["train_loader"], ctx["val_loader"], ctx["test_loader"]
    device = ctx["device"]

    flops_per_visit = ctx["flops_fwd_per_sample"] * 3  # fwd + bwd ~= 3x fwd
    bs = recipe["batch_size"]

    best_val, best = -1.0, {}
    train_seconds = eval_seconds = 0.0
    visits = 0
    history = []
    t_phase = time.perf_counter()

    for step in range(1, total_steps + 1):
        t0 = time.perf_counter()
        loss = train(train_loader, model, optimizer, ctx["train_criterion"], device, step, recipe)
        train_seconds += time.perf_counter() - t0
        visits += bs

        if step % eval_every == 0 or step == total_steps:
            t0 = time.perf_counter()
            val_loss, val_acc = test(val_loader, model, ctx["val_criterion"], ctx["val_metric"],
                                     device, ctx["val_loader_len"], split="val")
            row = {"phase": name, "step": step, "train_loss": round(loss, 5),
                   "val_acc": round(val_acc, 4), "test_acc": None,
                   "visits": visits, "tflops": round(visits * flops_per_visit / 1e12, 4),
                   "train_s": round(train_seconds, 1)}
            signals = {"val/accuracy": val_acc}

            if val_acc > best_val:          # new best validation -> read test once
                best_val = val_acc
                _, test_acc = test(test_loader, model, ctx["test_criterion"], ctx["test_metric"],
                                   device, ctx["test_loader_len"], split="test")
                row["test_acc"] = round(test_acc, 4)
                signals["test/accuracy"] = test_acc
                best = {"step": step, "val_acc": val_acc, "test_acc": test_acc,
                        "visits": visits, "tflops": row["tflops"],
                        "train_s": round(train_seconds, 1)}
            eval_seconds += time.perf_counter() - t0

            wl.save_model_signals(signals)
            history.append(row)
            mark = "  <- new best val, test read" if row["test_acc"] is not None else ""
            print(f"[{name}] step {step:5d}/{total_steps}  loss {loss:.4f}  "
                  f"val {val_acc:6.2f}%  test {row['test_acc'] if row['test_acc'] is not None else '   -  '}"
                  f"  {row['tflops']:.2f} TFLOPs{mark}", flush=True)

    return {
        "phase": name,
        "steps": total_steps,
        "batch_size": bs,
        "sample_visits": visits,
        "train_seconds": round(train_seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "wall_seconds": round(time.perf_counter() - t_phase, 1),
        "flops_fwd_per_sample": ctx["flops_fwd_per_sample"],
        "train_tflops": round(visits * flops_per_visit / 1e12, 4),
        "best": best,
        "history": history,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    # Load hyperparameters (from YAML if present)
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters = parameters or {}

    # ---- sensible defaults / normalization ----
    parameters.setdefault("experiment_name", "mnist_cnn")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000000)
    parameters.setdefault("eval_full_to_steps_ratio", 50)
    parameters.setdefault("seed", 0)
    parameters.setdefault("augment", False)
    parameters.setdefault("dropout", False)
    parameters.setdefault("optimizer", {}).setdefault("schedule", "const")
    parameters["optimizer"].setdefault("weight_decay", 0.0)

    # Deterministic seeding: a rerun reproduces the same shuffle, and therefore
    # the same per-digit visit order.
    seed = int(parameters["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Experiment name
    exp_name = parameters["experiment_name"]

    # Hyperparameters (must use 'hyperparameters' flag for trainer services / UI)
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        poll_interval=1.0,
    )

    # Device selection
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    # Logging dir
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    # Parameters
    verbose = parameters.get('verbose', True)
    log_dir = parameters["root_log_dir"]
    tqdm_display = parameters.get("tqdm_display", True)
    eval_full_to_train_steps_ratio = parameters.get("eval_full_to_train_steps_ratio", 50)
    write_export_ratio = parameters.get("write_export_ratio", 100)
    enable_h5_persistence = parameters.get("enable_h5_persistence", True)

    # Model
    _model = CNN(dropout=bool(parameters["dropout"])).to(device)
    model = wl.watch_or_edit(_model, flag="model", device=device)

    # Optimizer
    opt_cfg = parameters["optimizer"]
    lr = opt_cfg.get("lr", 0.001)
    _optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=float(opt_cfg["weight_decay"]))
    optimizer = wl.watch_or_edit(
        _optimizer,
        flag="optimizer",
    )

    # Data (MNIST train/val/test)
    # Use data_root from config if provided, otherwise fall back to log_dir/data
    if parameters.get("data_root"):
        should_download = False
        if not os.path.exists(parameters["data_root"]):
            print(f"Warning: data_root {parameters['data_root']} does not exist. Will attempt to download to this location.")
            should_download = True
        data_root = parameters["data_root"]
    else:
        data_root = os.path.join(parameters["root_log_dir"], "data")
        should_download = True
        print(f"Downloading data to {data_root}")
    os.makedirs(data_root, exist_ok=True)

    # Read data config for all loaders
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    val_cfg = parameters.get("data", {}).get("val_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    _full_train = MNISTCustomDataset(
        root=data_root,
        train=True,
        download=should_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        max_samples=train_cfg.get("max_samples", None)
    )
    _test_dataset = MNISTCustomDataset(
        root=data_root,
        train=False,
        download=should_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        max_samples=test_cfg.get("max_samples", None)
    )

    # Split the 60,000 MNIST train images into the training pool and a held-out
    # validation set BEFORE anything is registered: the two loaders then own
    # disjoint images, so nothing has to be discarded afterwards and no image is
    # registered twice.
    #
    # WeightsLab numbers each split's samples contiguously from where the
    # previous one ended, so a sample id here is a position in the split, not an
    # MNIST index. Both index lists are written next to this config, in loader
    # order, so the mapping back is a lookup:
    #
    #     mnist_index = train_ids[wl_sample_id]                 (train split)
    #     mnist_index = val_ids[wl_sample_id - len(train_ids)]  (val split)
    val_ids_path = parameters.get("val_ids")
    if val_ids_path and not os.path.isabs(val_ids_path):
        val_ids_path = os.path.join(os.path.dirname(__file__), val_ids_path)
    val_ids = get_val_ids(_full_train.mnist.targets, val_ids_path)
    held_out = set(val_ids)
    train_ids = [i for i in range(len(_full_train)) if i not in held_out]
    assert not (set(train_ids) & held_out), "train ids overlap the validation split"

    if val_ids_path:  # same order the loader serves them in
        with open(os.path.join(os.path.dirname(val_ids_path), "train_ids.txt"), "w") as fh:
            fh.write("\n".join(map(str, train_ids)) + "\n")

    _train_dataset = SubsetWithIds(_full_train, train_ids)
    _val_dataset = SubsetWithIds(_full_train, val_ids)
    n_train = len(_train_dataset)

    # Create tracked loaders for train, val and test
    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=train_cfg.get("shuffle", True),
        is_training=True,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=False,
        enable_h5_persistence=enable_h5_persistence
    )
    val_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        loader_name="val_loader",
        batch_size=val_cfg.get("batch_size", 500),
        shuffle=val_cfg.get("shuffle", False),
        is_training=False,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=False,
        enable_h5_persistence=enable_h5_persistence
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 16),
        shuffle=test_cfg.get("shuffle", False),
        is_training=False,
        compute_hash=False,
        preload_labels=True,
        preload_metadata=False,
        enable_h5_persistence=enable_h5_persistence
    )

    # 8 passes over 55,000 digits at batch 64 = 8 x 860 = 6,880 steps.
    batch_size = train_cfg.get("batch_size", 16)
    steps_per_pass = -(-n_train // batch_size)
    if parameters.get("epochs"):
        parameters["training_steps_to_do"] = int(parameters["epochs"]) * steps_per_pass
    total_steps = int(parameters["training_steps_to_do"])
    recipe = {
        "augment": bool(parameters["augment"]),
        "lr": float(lr),
        "schedule": opt_cfg["schedule"],
        "total_steps": total_steps,
    }

    # Losses & metrics (watched objects – they log themselves)
    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="train-loss-CE", per_sample=True, log=True)
    test_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="test-loss-CE", log=True)

    val_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="val-loss-CE", log=True)

    metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric", signal_name="metric-ACC", log=True)
    val_metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric", signal_name="metric-ACC-val", log=True)

    # Start WeightsLab services (gRPC only, no CLI)
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", False)
    )

    # ---- constants shared by both phases -------------------------------------
    val_loader_len = len(val_loader)
    test_loader_len = len(test_loader)
    flops_fwd = model_flops_per_sample(_model, device)   # unwrapped: see the docstring
    labels_by_id = {i: int(_full_train.mnist.targets[m]) for i, m in enumerate(train_ids)}
    gcfg = parameters.get("goldset", {}) or {}

    ctx = dict(wl=wl, model=model, optimizer=optimizer, device=device,
               train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
               train_criterion=train_criterion, val_criterion=val_criterion,
               test_criterion=test_criterion, val_metric=val_metric, test_metric=metric,
               val_loader_len=val_loader_len, test_loader_len=test_loader_len,
               flops_fwd_per_sample=flops_fwd)

    p_sig = parameters["phases"]["signal"]
    p_gold = parameters["phases"]["goldset"]
    sig_steps = int(p_sig["epochs"]) * (-(-n_train // int(p_sig["batch_size"])))

    print("=" * 72)
    print(" TWO-PHASE EXPERIMENT (one process, one experiment directory)")
    print(f" splits: train={n_train}  val={len(_val_dataset)}  test={len(_test_dataset)}")
    print(f" model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"{flops_fwd / 1e6:.2f} MFLOPs forward per sample ({flops_fwd * 3 / 1e6:.2f} incl. backward)")
    print(f" phase A (signal):  {sig_steps} steps x batch {p_sig['batch_size']} "
          f"= {p_sig['epochs']} passes over {n_train} digits")
    print(f" phase B (goldset): {p_gold['training_steps_to_do']} steps x batch {p_gold['batch_size']}")
    print(f" logs: {log_dir}")
    print("=" * 72 + "\n")

    # Hand control to WeightsLab before the first guarded step: without this the
    # training guard never opens and the loop stalls with the process alive.
    wl.start_training(timeout=int(parameters.get("start_training_timeout", 1)))

    results = {"splits": {"train": n_train, "val": len(_val_dataset), "test": len(_test_dataset)},
               "model_params": int(sum(p.numel() for p in model.parameters())),
               "flops_fwd_per_sample": flops_fwd}
    t_all = time.perf_counter()

    # ---- the initial random weights, kept so phase B can start from them ------
    # Tensors only: a watched model's state_dict also carries scalar bookkeeping
    # (the age counter), which has nothing to clone or compare.
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if torch.is_tensor(v)}

    # ---- PHASE A: signal run over the whole training pool ---------------------
    recipe_sig = {"augment": bool(p_sig["augment"]), "lr": float(p_sig["lr"]),
                  "schedule": p_sig["schedule"], "total_steps": sig_steps,
                  "batch_size": int(p_sig["batch_size"])}
    train_loader.set_batch_size(int(p_sig["batch_size"]))
    _set_dropout(model, bool(p_sig["dropout"]))
    for g in optimizer.param_groups:
        g["lr"] = float(p_sig["lr"])
    results["signal"] = run_phase("signal", sig_steps, recipe_sig, int(p_sig["eval_every"]), ctx)

    # ---- build the goldset from what phase A recorded -------------------------
    t0 = time.perf_counter()
    steps_per_pass = -(-n_train // int(p_sig["batch_size"]))
    goldset, ginfo = build_goldset(wl, steps_per_pass, labels_by_id, range(n_train), gcfg)
    wl.tag_samples(goldset, "goldset")
    results["goldset_build"] = {**ginfo, "size": len(goldset),
                                "seconds": round(time.perf_counter() - t0, 1)}
    mnist_ids = sorted(int(train_ids[i]) for i in goldset)
    with open(os.path.join(log_dir, "goldset_ids_mnist.txt"), "w") as fh:
        fh.write("\n".join(map(str, mnist_ids)) + "\n")
    print(f"\n[goldset] {len(goldset)} digits ({len(goldset) / n_train:.2%} of the pool) "
          f"from {ginfo['passes']} passes; {ginfo['noisy']} noisy excluded; "
          f"tagged 'goldset'; MNIST ids -> goldset_ids_mnist.txt\n", flush=True)

    # ---- reload the INITIAL weights (model age 0), then train on the goldset ---
    reload_info = _reload_initial_weights(wl, model, optimizer, init_state, log_dir)
    results["reload"] = reload_info
    print(f"[reset] restored the step-0 weights ({reload_info['source']}); "
          f"max|w - w0| = {reload_info['max_abs_diff']}\n", flush=True)

    keep = set(int(i) for i in goldset)
    wl.discard_samples([i for i in range(n_train) if i not in keep])

    recipe_gold = {"augment": bool(p_gold["augment"]), "lr": float(p_gold["lr"]),
                   "schedule": p_gold["schedule"], "total_steps": int(p_gold["training_steps_to_do"]),
                   "batch_size": int(p_gold["batch_size"])}
    train_loader.set_batch_size(int(p_gold["batch_size"]))
    _set_dropout(model, bool(p_gold["dropout"]))
    results["goldset"] = run_phase("goldset", int(p_gold["training_steps_to_do"]),
                                   recipe_gold, int(p_gold["eval_every"]), ctx)

    # ---- report ---------------------------------------------------------------
    results["wall_seconds_total"] = round(time.perf_counter() - t_all, 1)
    _summarise(results, n_train, len(goldset))
    with open(os.path.join(log_dir, "experiment_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\n results -> {os.path.join(log_dir, 'experiment_results.json')}")

    wl.write_history()
    wl.write_dataframe()

    # Keep the main thread alive so the Studio stays attached. Set
    # keep_serving: false to exit once the results are written -- a batch sweep
    # runs these back to back and must not block on the last one.
    if parameters.get("keep_serving", True):
        wl.keep_serving()
    else:
        print(" keep_serving: false -> exiting", flush=True)
