"""Fashion-MNIST classification with per-step MODEL signals.

Same shape as ``examples/PyTorch/wl-classification`` (plain PyTorch loop,
watched model/optimizer/loaders/loss/metric, guard contexts) with one addition:
this run also plots its own training dynamics.

    metrics/global/grad_norm                whole-model gradient L2 norm
    metrics/global/weights_norm             whole-model parameter L2 norm
    metrics/layer/<layer_id>/grad_norm      per-layer parameter gradients
    metrics/layer/<layer_id>/weights_norm   per-layer parameters
    metrics/layer/<layer_id>/activation_mean
    metrics/layer/<layer_id>/activation_std
    metrics/layer/<layer_id>/activation_max
    metrics/layer/<layer_id>/activation_min

All of it comes from ONE argument -- ``track_model_signals=True`` on the model's
``watch_or_edit`` (see MODEL below). No hooks, no per-step bookkeeping, and no
call anywhere in the training loop: gradients are read by post-accumulate hooks
the moment they are final, activations by forward hooks, and the whole set is
flushed once per step just before ``optimizer.step()`` consumes it.

Why these signals are NOT ``wl.save_signals``: everything that verb records is
keyed by a sample, and a gradient norm does not belong to any sample -- it
belongs to the step. ``wl.save_model_signals`` is the step-keyed write path (use
it directly for any dynamics value of your own, e.g. a gradient-to-weight
ratio); ``wl.track_model_signals`` is the collector that fills it in for you.

What the curves are for -- Fashion-MNIST makes each failure legible:

    grad_norm collapsing toward 0 in the EARLY layers while the late ones stay
    healthy is vanishing gradient; the run keeps "training" and stops learning.

    grad_norm spiking by orders of magnitude is the exploding case -- pair it
    with the loss curve to see which came first.

    activation_std -> 0 on a layer is that layer going constant (dead ReLUs,
    saturated BatchNorm): it is still consuming compute and contributing
    nothing. activation_min stuck at exactly 0.0 across a whole ReLU is the
    same story from the other side.

    weights_norm climbing without bound while the loss flattens is the model
    growing weights instead of learning structure -- the moment to add decay.

Run::

    python main.py                       # reads config.yaml next to this file
    WEIGHTSLAB_ROOT_LOG_DIR=<dir> python main.py
"""

import itertools
import logging
import os
import ssl
import tempfile
import time

# Windows SSL fix: some Windows cert stores contain malformed ASN1 certs that
# crash ssl.create_default_context(). Fall back to unverified only when broken.
try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import Dataset
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms

import weightslab as wl

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Fashion-MNIST's own label order (torchvision docs). Carried as per-sample
# metadata so the grid shows "Pullover" instead of "2" -- which is the
# difference between spotting a shirt/coat/pullover confusion and not.
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


# =============================================================================
# Dataset
# =============================================================================
class FashionMNISTDataset(Dataset):
    """Fashion-MNIST yielding ``(image, sample_id, label)``.

    ``sample_id`` is offset per split (``id_base``) so train and test ids never
    collide in the shared ledger -- without it, test sample 0 would overwrite
    train sample 0's signal history.
    """

    def __init__(self, root, train=True, download=True, transform=None,
                 max_samples=None, id_base=0):
        try:
            self.data = datasets.FashionMNIST(root=root, train=train,
                                             download=download, transform=None)
        except RuntimeError as exc:
            logger.error(f"Error loading Fashion-MNIST: {exc}")
            self.data = datasets.FashionMNIST(root=root, train=train,
                                             download=True, transform=None)
        self.transform = transform
        self.train = train
        self.max_samples = max_samples
        self.id_base = id_base

    def __len__(self):
        if self.max_samples is not None:
            return min(len(self.data), self.max_samples)
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.id_base + idx, label

    def fast_get_label(self, idx):
        """Lets the ledger read labels at init without decoding every image."""
        return int(self.data.targets[idx])

    def get_metadata(self, idx):
        """Per-sample metadata surfaced in the grid / metadata panel."""
        label = int(self.data.targets[idx])
        return {
            "class_name": CLASS_NAMES[label],
            "split": "train" if self.train else "test",
        }


# =============================================================================
# Model
# =============================================================================
class FashionCNN(nn.Module):
    """Three conv blocks + a two-layer head.

    Deliberately deeper than the task needs: per-layer signals only tell you
    something once there are enough layers for the early ones to behave
    differently from the late ones. Every module is a named attribute (no
    ``nn.Sequential``) so each gets its own layer id and therefore its own
    curve -- a Sequential block would collapse to one.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)                      # 28 -> 14

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)                      # 14 -> 7

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        # Logits, not softmax: the watched CrossEntropyLoss applies its own.
        return self.fc2(self.relu4(self.fc1(self.flatten(x))))


def print_layer_legend(model):
    """Map each layer id to its module, once, at startup.

    ``metrics/layer/7/grad_norm`` says nothing on its own. These are the same
    ids the model panel and every architecture op (freeze/reset) use, so this
    legend is what lets you read a curve and act on the layer behind it.
    """
    from weightslab.components.model_signals import _iter_layers

    rows = _iter_layers(model)
    if not rows:
        print(" (no layer ids resolved -- per-layer curves will be positional)")
        return
    print(f" {'layer_id':>9}  {'module':<14}  shape")
    for layer_id, module in rows:
        weight = getattr(module, "weight", None)
        shape = tuple(weight.shape) if weight is not None else "-"
        print(f" {layer_id:>9}  {type(module).__name__:<14}  {shape}")


# -----------------------------------------------------------------------------
# Train / test
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion, device):
    """One training step. Nothing here logs model signals -- the hooks do."""
    with wl.guard_training_context:
        inputs, ids, labels = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        preds = logits.argmax(dim=1, keepdim=True)

        loss_per_sample = criterion(
            logits.float(), labels.long(), batch_ids=ids, preds=preds,
        )
        total_loss = loss_per_sample.mean()
        total_loss.backward()
        optimizer.step()

    return total_loss.detach().cpu().item()


def test(loader, model, criterion, metric, device, num_batches):
    """Full pass over the test split.

    No model signals come out of this: the tracker only collects inside
    ``guard_training_context``, so an eval pass can't contaminate a gradient or
    activation curve with values the optimizer never saw.
    """
    losses = torch.tensor(0.0, device=device)

    for inputs, ids, labels in loader:
        with wl.guard_testing_context, torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            preds = logits.argmax(dim=1, keepdim=True)

            losses += criterion(
                logits, labels, batch_ids=ids, preds=preds,
            ).mean()
            metric.update(logits, labels)

            correct = (preds.view(-1) == labels.view(-1)).float()
            wl.save_signals(
                signals={
                    "test_metric/accuracy_per_sample": correct,
                    "test_metric/error_per_sample": 1.0 - correct,
                },
                batch_ids=ids,
                preds_raw=logits,
                targets=labels,
                preds=preds,
            )

    return (losses / num_batches).detach().cpu().item(), (metric.compute() * 100).detach().cpu().item()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    start_time = time.time()

    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}

    parameters.setdefault("experiment_name", "fashion_mnist_model_signals")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 3000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 250)
    parameters.setdefault("model_signals_every_n_steps", 1)

    # Hyperparameters first: everything below reads from the watched dict, so a
    # value edited in the UI is picked up without restarting.
    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]

    # `weightslab start <dir>` exports WEIGHTSLAB_ROOT_LOG_DIR -- honor it, so
    # this run lands in the directory the dashboard is actually watching. A
    # temp dir only when nothing said otherwise.
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR") or tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]

    verbose = parameters.get("verbose", True)
    tqdm_display = parameters.get("tqdm_display", True)
    eval_ratio = parameters.get("eval_full_to_train_steps_ratio", 250)
    enable_h5 = parameters.get("enable_h5_persistence", True)

    # ---- MODEL -------------------------------------------------------------
    # `track_model_signals=True` is the whole feature: it installs the hooks
    # that produce every metrics/* curve. Pass a list to narrow the set, e.g.
    # track_model_signals=["grad_norm", "activation_std"].
    #
    # `model_signals_every_n_steps` samples every Nth step. The activation
    # hooks are the only per-step cost worth thinking about; on a big model
    # raise this to 10-50 and the overhead disappears while the curves stay
    # perfectly readable.
    model = wl.watch_or_edit(
        FashionCNN(num_classes=len(CLASS_NAMES)).to(device),
        flag="model",
        device=device,
        track_model_signals=True,
        model_signals_every_n_steps=parameters.get("model_signals_every_n_steps", 1),
    )

    # Build the optimizer from the WATCHED model's parameters, not the raw one.
    lr = parameters.get("optimizer", {}).get("lr", 0.001)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=lr), flag="optimizer")

    # ---- DATA --------------------------------------------------------------
    if parameters.get("data_root"):
        data_root = parameters["data_root"]
        should_download = not os.path.exists(data_root)
    else:
        data_root = os.path.join(log_dir, "data")
        should_download = True
    os.makedirs(data_root, exist_ok=True)

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    to_tensor = transforms.Compose([transforms.ToTensor()])

    train_dataset = FashionMNISTDataset(
        root=data_root, train=True, download=should_download, transform=to_tensor,
        max_samples=train_cfg.get("max_samples"), id_base=0,
    )
    test_dataset = FashionMNISTDataset(
        root=data_root, train=False, download=should_download, transform=to_tensor,
        max_samples=test_cfg.get("max_samples"), id_base=1_000_000,
    )

    train_loader = wl.watch_or_edit(
        train_dataset, flag="data", loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=train_cfg.get("shuffle", True),
        is_training=True, compute_hash=False,
        preload_labels=True, preload_metadata=True,
        enable_h5_persistence=enable_h5,
    )
    test_loader = wl.watch_or_edit(
        test_dataset, flag="data", loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 256),
        shuffle=test_cfg.get("shuffle", False),
        is_training=False, compute_hash=False,
        preload_labels=True, preload_metadata=True,
        enable_h5_persistence=enable_h5,
    )

    # ---- LOSS / METRIC -----------------------------------------------------
    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="train-loss-CE", log=True, per_sample=True)
    test_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="test-loss-CE", log=True, per_sample=True)
    metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=len(CLASS_NAMES)).to(device),
        flag="metric", signal_name="metric-ACC", log=True)

    wl.serve(serving_grpc=parameters.get("serving_grpc", True))

    print("=" * 72)
    print(" FASHION-MNIST + PER-STEP MODEL SIGNALS")
    print(f" train={len(train_dataset)}  test={len(test_dataset)}  device={device}")
    print(f" eval every {eval_ratio} steps | model signals every "
          f"{parameters.get('model_signals_every_n_steps', 1)} step(s)")
    print(f" logs -> {log_dir}")
    print("-" * 72)
    print(" LAYER LEGEND (these ids name the metrics/layer/<id>/* curves)")
    print_layer_legend(model)
    print("=" * 72 + "\n")

    # Training runs until YOU stop it -- from the studio's pause button, the CLI,
    # or Ctrl+C. itertools.count() rather than range(training_steps_to_do): a
    # predefined step budget ends the process mid-experiment, which is the
    # opposite of how WeightsLab is used (inspect the curves, edit the data or
    # the architecture, keep going). `training_steps_to_do` remains a live
    # hyperparameter for the UI's own "run N more steps" control; it is not a
    # ceiling on this loop.
    if tqdm_display:
        train_range = tqdm.tqdm(
            itertools.count(),
            desc="Training",
            bar_format="{desc}: {n} steps [{elapsed}, {rate_fmt}] {bar} | {postfix}",
            ncols=140, position=0, leave=True,
        )
    else:
        train_range = itertools.count()

    wl.start_training(timeout=3)

    train_loss = None
    test_loss, test_metric = None, None
    test_batches = len(test_loader)

    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step

        train_loss = train(train_loader, model, optimizer, train_criterion, device)

        if age > 0 and age % eval_ratio == 0:
            test_loss, test_metric = test(
                test_loader, model, test_criterion, metric, device, test_batches)

        if tqdm_display:
            parts = [f"train_loss={train_loss:.4f}"]
            if test_loss is not None:
                parts.append(f"test_loss={test_loss:.4f}")
            if test_metric is not None:
                parts.append(f"test_acc={test_metric:.1f}%")
            train_range.set_postfix_str(" | ".join(parts))
        elif verbose:
            import sys
            msg = f"Step {train_step} (age {age}): loss={train_loss:.4f}"
            if test_loss is not None:
                msg += f" | test={test_loss:.4f} ({test_metric:.1f}%)"
            sys.stdout.write(f"\r{msg:<100}")
            sys.stdout.flush()

    print("\n" + "=" * 72)
    print(f" Done in {time.time() - start_time:.1f}s | logs -> {log_dir}")
    print("=" * 72)

    # Flush async signals before reading anything back, then dump both the
    # step-keyed signal history (where every metrics/* curve lives) and the
    # per-sample grid.
    wl.write_history()
    wl.write_dataframe()

    wl.keep_serving()
