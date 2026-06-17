import itertools
import os
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
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import weightslab as wl
from weightslab.backend import ledgers
from weightslab.baseline_models.pytorch.models import FashionCNN as CNN
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)
from signal_maintenance import SignalMaintenanceThread


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
                transform=None  # We'll apply transform manually to track filepath
            )
        except RuntimeError as e:
            logger.error(f"Error loading MNIST dataset: {e}")
            self.mnist = datasets.MNIST(
                root=root,
                train=train,
                download=True,
                transform=None  # We'll apply transform manually to track filepath
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
            if self.max_samples is not None and idx >= self.max_samples:
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
        if self.max_samples is not None:
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


# -----------------------------------------------------------------------------
# Per-sample metric modules
# -----------------------------------------------------------------------------
# Each is a tiny nn.Module whose forward(outputs, labels) returns one value per
# sample ([B]). Wrapped with wl.watch_or_edit(flag="metric", per_sample=True),
# WL auto-converts every call into a per-sample signal keyed by batch_ids — so
# the train loop just *calls* them, no manual wl.save_signals(). Same pattern as
# the 3D-detection example's PerSampleBevIoU.
class _ProbMetric(nn.Module):
    """Base: softmax once, then a per-sample reduction in forward()."""
    @staticmethod
    def _probs(outputs):
        return torch.softmax(outputs.float(), dim=1)


class Confidence(_ProbMetric):
    def forward(self, outputs, labels):
        return self._probs(outputs).max(dim=1).values


class TrueClassProb(_ProbMetric):
    def forward(self, outputs, labels):
        p = self._probs(outputs)
        return p.gather(1, labels.long().view(-1, 1)).squeeze(1)


class PredMargin(_ProbMetric):
    def forward(self, outputs, labels):
        top2 = self._probs(outputs).topk(2, dim=1).values
        return top2[:, 0] - top2[:, 1]


class PredEntropy(_ProbMetric):
    def forward(self, outputs, labels):
        p = self._probs(outputs)
        return -(p * (p + 1e-9).log()).sum(dim=1)


class IsCorrect(_ProbMetric):
    def forward(self, outputs, labels):
        return (outputs.argmax(dim=1) == labels.view(-1)).float()


class CorrectClassRank(_ProbMetric):
    def forward(self, outputs, labels):
        p = self._probs(outputs)
        true_p = p.gather(1, labels.long().view(-1, 1))
        return (p > true_p).sum(dim=1).float() + 1.0


# name -> metric class; registered once in main and called every train step.
STRESS_METRICS = {
    "train/confidence": Confidence,
    "train/true_class_prob": TrueClassProb,
    "train/pred_margin": PredMargin,
    "train/pred_entropy": PredEntropy,
    "train/is_correct": IsCorrect,
    "train/correct_class_rank": CorrectClassRank,
}


# --- Off-thread signal compute (the SignalMaintenanceThread prototype) ---------
# Raw (UNwrapped) metric modules — compute only; no watch_or_edit, no inline save.
_METRICS = {name: cls() for name, cls in STRESS_METRICS.items()}


def compute_all_signals(t):
    """Runs on the maintenance thread. t = {'preds_raw':…, 'labels':…} (detached).
    Returns one value/sample for each of the 6 base metrics (need raw logits).
    'decision_shape' is NO LONGER computed here — it is a framework-driven
    @wl.signal over the loss trajectory (see below)."""
    return {name: mod(t["preds_raw"], t["labels"]) for name, mod in _METRICS.items()}


# -----------------------------------------------------------------------------
# Categorical trajectory shape — a real @wl.signal, computed by the FRAMEWORK
# executor whenever a sample's loss updates. It classifies the *shape* of the
# per-sample loss curve handed to it via ctx.value.history.
# -----------------------------------------------------------------------------
SHAPE_LABELS = {0: "low", 1: "hilo", 2: "U", 3: "hi"}
SHAPE_TAU = 2.0     # CE threshold between low(~1.46) and high(~2.46) clusters
SHAPE_DELTA = 0.3   # min end-drop / mid-dip to count a transition


def classify(history, tau=SHAPE_TAU, delta=SHAPE_DELTA):
    """Reduce a per-sample loss trajectory to {0:low,1:hilo,2:U,3:hi}."""
    n = len(history)
    if n == 0:
        return 0
    k = max(1, n // 3)
    early = sum(history[:k]) / k
    late = sum(history[-k:]) / k
    mid = sum(history[k:n - k]) / (n - 2 * k) if n - 2 * k >= 1 else sum(history) / n
    overall = sum(history) / n
    if early > tau and late <= tau and (early - late) >= delta:
        return 1   # hilo: high -> low (being learned)
    if early > tau and late > tau and mid <= tau and (min(early, late) - mid) >= delta:
        return 2   # U: high -> low -> high (forgotten / unstable)
    return 3 if overall > tau else 0   # hi (stubborn) / low (easy)


# Overhead A/B/C knobs:
#   WL_SHAPE=0        -> signal not registered at all (baseline)
#   WL_SHAPE_HEAVY=N  -> N iterations of a pure-Python CPU burn per classify call
#                        (simulates a heavy signal; holds the GIL, so it contends
#                         with training even though it runs on the executor thread)
SHAPE_HEAVY = int(os.environ.get("WL_SHAPE_HEAVY", "0"))


def decision_shape(ctx):
    # Fired by the framework executor on every loss update for this sample.
    hist = ctx.value.history
    if SHAPE_HEAVY:
        s = 0.0
        for _ in range(SHAPE_HEAVY):
            s = (s + 1.234) ** 0.5      # pure-Python busy work (holds the GIL)
    return classify(hist)


if os.environ.get("WL_SHAPE", "1") == "1":
    wl.signal(name="train/decision_shape", subscribe_to="train-loss-CE")(decision_shape)


# -----------------------------------------------------------------------------
# Train / Test functions
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, sig_worker, device):
    """Single training step. ALL per-sample signal compute is off-loaded to the
    maintenance thread via one cheap submit()."""

    with guard_training_context:
        (inputs, ids, labels) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Infer
        optimizer.zero_grad()
        preds_raw = model(inputs)
        preds = preds_raw.argmax(dim=1, keepdim=True)

        # Watched loss (needed for backward + saves train-loss-CE).
        loss_batch_mlt = criterion_mlt(
            preds_raw.float(), labels.long(), batch_ids=ids, preds=preds)
        total_loss = loss_batch_mlt.mean()

        # Off-thread: detach + enqueue. The maintenance thread computes the 6
        # metrics + the derived decision_shape and persists them. O(1) here.
        sig_worker.submit(ids, preds_raw=preds_raw, labels=labels)

        # Model
        total_loss.backward()
        optimizer.step()

    return total_loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device, test_loader_len):
    """Full evaluation pass over the test loader."""
    losses = torch.tensor(0.0, device=device)

    for (inputs, ids, labels) in loader:
        with guard_testing_context:
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
                "test_metric/Accuracy_per_sample": acc_per_sample,
                "test_metric/Inverse_Accuracy_per_sample": acc_reversed_per_sample,
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

    # Experiment name
    exp_name = parameters["experiment_name"]

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
    enable_h5_persistence = parameters.get("enable_h5_persistence", True)
    training_steps_to_do = parameters.get("training_steps_to_do", 1000)

    # Hyperparameters (must use 'hyperparameters' flag for trainer services / UI)
    hp = ledgers.get_hyperparams()
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        defaults=parameters,
        poll_interval=1.0,
    )

    # Model
    _model = CNN().to(device)
    model = wl.watch_or_edit(_model, flag="model", device=device)

    # Optimizer
    lr = parameters.get("optimizer", {}).get("lr", 0.01)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
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
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    _train_dataset = MNISTCustomDataset(
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

    # Create tracked loaders for train, test, and test
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

    # Losses & metrics (watched objects – they log themselves)
    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="train-loss-CE", log=True)
    test_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="test-loss-CE", log=True)

    metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=10).to(device),
        flag="metric", signal_name="metric-ACC", log=True)

    # Dedicated maintenance thread owns ALL per-sample signal compute off the
    # training hot path (6 metrics + derived decision_shape). The train loop only
    # submits detached tensors; this thread computes + persists via save_signals.
    sig_worker = SignalMaintenanceThread(compute_all_signals, origin="train").start()

    # Start WeightsLab services (gRPC only, no CLI)
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", False),
        serving_cli=parameters.get("serving_cli", False),
    )

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"🔄 Evaluation every {eval_full_to_train_steps_ratio} steps")
    print(f"� Dataset splits: train={len(_train_dataset)}, test={len(_test_dataset)}")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    # Setup clean progress bar with custom format
    if tqdm_display:
        train_range = tqdm.tqdm(
            range(training_steps_to_do) if training_steps_to_do is not None else itertools.count(),
            desc="Training",
            bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {bar} | {postfix}",
            ncols=140,
            position=0,
            leave=True
        )
    else:
        train_range = range(training_steps_to_do) if training_steps_to_do is not None else itertools.count()

    test_loader_len = len(test_loader)  # Store length before wrapping with tqdm

    train_loss = None
    test_loss, test_metric = None, None
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step  # Get model age in steps (not necessarily equal to train_step if model was reloaded or has seen more data than training steps)

        # Train one step
        train_loss = train(train_loader, model, optimizer, train_criterion, sig_worker, device)
        if train_step % 200 == 0:
            avg = sig_worker.compute_ms_sum / max(sig_worker.processed, 1)
            print(f"[sigworker] step~{sig_worker.processed} backlog={sig_worker.backlog} "
                  f"compute_ms avg={avg:.2f} max={sig_worker.compute_ms_max:.2f} last={sig_worker.compute_ms_last:.2f}",
                  flush=True)
            try:
                from weightslab.src import get_signal_executor
                h = get_signal_executor().health()
                print(f"[executor] backlog={h['backlog']} processed={h['processed']} "
                      f"dropped={h['dropped']} errors={h['errors']} compute_ms={h['compute_ms']:.2f}",
                      flush=True)
            except Exception as _e:
                print(f"[executor] health err: {_e}", flush=True)

        # Periodic test evaluation
        if age > 0 and age % eval_full_to_train_steps_ratio == 0:
            # Test (no nested progress bar)
            test_loss, test_metric = test(
                test_loader,
                model,
                test_criterion,
                metric,
                device,
                test_loader_len
            )

        # Verbose
        if verbose and not tqdm_display:
            import sys
            # Build compact progress message
            msg = f"Step {train_step} (Age {age}): Loss={train_loss:.4f}"
            if test_loss is not None:
                msg += f" | Test={test_loss:.4f} ({test_metric:.1f}%)"

            # Clear line completely and print (pad to 100 chars to overwrite previous content)
            sys.stdout.write(f"\r{msg:<100}")
            sys.stdout.flush()
        elif tqdm_display:
            # Build compact postfix string
            postfix_parts = [f"train_loss={train_loss:.4f}"]
            if test_loss is not None:
                postfix_parts.append(f"test_loss={test_loss:.4f}")
            if test_metric is not None:
                postfix_parts.append(f"test_acc={test_metric:.1f}%")

            train_range.set_postfix_str(" | ".join(postfix_parts))

    print("\n" + "=" * 60)
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
