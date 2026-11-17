"""Train to convergence, pause from code, then project with YOUR OWN algorithm.

The flow this example shows:

1. Fashion-MNIST trains with WeightsLab's built-in projection turned OFF
   (``projection=False``) -- the projection here is going to be yours.
2. The script watches test accuracy and, once it stops improving, pauses
   training itself with ``wl.pause_training()``. Nothing is lost: the process,
   the model and the Studio all stay up; only the next step waits for Play.
3. In Weights Studio's notebook (the button left of the logo), the notebook
   ``own-projection`` is ready: it collects the converged model's features,
   runs a t-SNE written from scratch in plain torch -- not WeightsLab's -- and
   hands the coordinates over with ``wl.save_projection_coords``.
4. Data Exploration -> Projection shows your t-SNE (and the PCA of the last
   cell) in the picker: lasso a cluster, and the grid loads those images.

Running a notebook cell pauses training first, so a cell always sees one state
of the model even if you run it before convergence.

Run::

    python main.py          # then open the studio and press Play
"""

import itertools
import logging
import os
import ssl
import tempfile

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
from projection_notebook import write_projection_notebook

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


# =============================================================================
# Dataset / model (same as wl-fashion-mnist-umap)
# =============================================================================
class FashionMNISTDataset(Dataset):
    """Fashion-MNIST yielding ``(image, sample_id, label)``; ids offset per split."""

    def __init__(self, root, train=True, download=True, transform=None,
                 max_samples=None, id_base=0):
        self.data = datasets.FashionMNIST(root=root, train=train,
                                         download=download, transform=None)
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
        return int(self.data.targets[idx])

    def get_metadata(self, idx):
        label = int(self.data.targets[idx])
        return {"class_name": CLASS_NAMES[label],
                "split": "train" if self.train else "test"}


class FashionCNN(nn.Module):
    """Two conv blocks, a 128-D representation (``embed``), the classifier."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def embed(self, x):
        """The representation the notebook projects: fc1 after its ReLU."""
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        return self.relu3(self.fc1(self.flatten(x)))

    def forward(self, x):
        return self.fc2(self.embed(x))


# =============================================================================
# Convergence
# =============================================================================
class Plateau:
    """Converged = test accuracy has not improved by ``min_delta`` points for
    ``patience`` evaluations in a row."""

    def __init__(self, patience=3, min_delta=0.25):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = None
        self.stale = 0

    def update(self, accuracy: float) -> bool:
        if self.best is None or accuracy > self.best + self.min_delta:
            self.best, self.stale = accuracy, 0
        else:
            self.stale += 1
        return self.stale >= self.patience


# =============================================================================
# Train / test
# =============================================================================
def train(loader, model, optimizer, criterion, device):
    with wl.guard_training_context:      # blocks here while training is paused
        inputs, ids, labels = next(loader)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        preds = logits.argmax(dim=1, keepdim=True)
        loss = criterion(logits.float(), labels.long(), batch_ids=ids, preds=preds)
        loss.mean().backward()
        optimizer.step()
    return loss.mean().detach().cpu().item()


def test(loader, model, criterion, metric, device):
    total, batches = 0.0, 0
    metric.reset()
    for inputs, ids, labels in loader:
        with wl.guard_testing_context, torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1, keepdim=True)
            total += criterion(logits, labels, batch_ids=ids, preds=preds).mean().item()
            metric.update(logits, labels)
            batches += 1
    return total / max(1, batches), float(metric.compute() * 100)


def announce_pause(step, accuracy, why, notebook_path):
    print("\n" + "=" * 72)
    print(f" {why} at step {step} (test accuracy {accuracy:.1f}%).")
    print(" Training PAUSED from code (wl.pause_training()). In Weights Studio:")
    print(f"   1. open the notebook -> '{notebook_path.stem}' is ready;")
    print("   2. run its cells top to bottom: your own t-SNE of this model;")
    print("   3. Data Exploration -> Projection, pick 'tsne' (or 'pca').")
    print(" Press Play in the header to resume training afterwards.")
    print("=" * 72 + "\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters.setdefault("experiment_name", "fashion_mnist_custom_projection")
    parameters.setdefault("device", "auto")

    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR") or tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]

    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]
    conv = parameters.get("convergence", {})
    eval_every = int(conv.get("eval_every_n_steps", 250))
    plateau = Plateau(conv.get("patience", 3), conv.get("min_delta", 0.25))
    pause_after = conv.get("pause_after_steps")      # safety net for the demo
    tqdm_display = parameters.get("tqdm_display", True)
    enable_h5 = parameters.get("ledger_enable_h5_persistence", True)

    # The built-in projection is OFF: the projection in this run is yours.
    model = wl.watch_or_edit(FashionCNN(num_classes=len(CLASS_NAMES)).to(device),
                             flag="model", device=device, projection=False)
    lr = parameters.get("optimizer", {}).get("lr", 0.001)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=lr), flag="optimizer")

    data_root = parameters.get("data_root") or os.path.join(log_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})
    to_tensor = transforms.Compose([transforms.ToTensor()])
    train_dataset = FashionMNISTDataset(root=data_root, train=True, transform=to_tensor,
                                        max_samples=train_cfg.get("max_samples"), id_base=0)
    test_dataset = FashionMNISTDataset(root=data_root, train=False, transform=to_tensor,
                                       max_samples=test_cfg.get("max_samples"),
                                       id_base=1_000_000)
    train_loader = wl.watch_or_edit(
        train_dataset, flag="data", loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 64), shuffle=train_cfg.get("shuffle", True),
        is_training=True, compute_hash=False, preload_labels=True, preload_metadata=True,
        enable_h5_persistence=enable_h5)
    test_loader = wl.watch_or_edit(
        test_dataset, flag="data", loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 256), shuffle=test_cfg.get("shuffle", False),
        is_training=False, compute_hash=False, preload_labels=True, preload_metadata=True,
        enable_h5_persistence=enable_h5)

    train_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="train-loss-CE", log=True, per_sample=True)
    test_criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none"),
        flag="loss", signal_name="test-loss-CE", log=True, per_sample=True)
    metric = wl.watch_or_edit(
        Accuracy(task="multiclass", num_classes=len(CLASS_NAMES)).to(device),
        flag="metric", signal_name="metric-ACC", log=True)

    # Before serving, so the Studio's notebook opens on it.
    notebook_path = write_projection_notebook(log_dir)
    wl.serve(serving_grpc=parameters.get("serving_grpc", True))

    print("=" * 72)
    print(" FASHION-MNIST -> CONVERGE -> PAUSE -> YOUR OWN PROJECTION")
    print(f" train={len(train_dataset)}  test={len(test_dataset)}  device={device}")
    print(f" pauses itself when test accuracy plateaus ({plateau.patience} evals of "
          f"{eval_every} steps without +{plateau.min_delta} pt)")
    print(f" notebook -> {notebook_path}")
    print(f" logs     -> {log_dir}")
    print(" Studio: press Play to start training.")
    print("=" * 72 + "\n")

    if parameters.get("auto_start_training", False):
        wl.start_training()

    steps = itertools.count()
    if tqdm_display:
        steps = tqdm.tqdm(steps, desc="Training", ncols=120,
                          bar_format="{desc}: {n} steps [{elapsed}, {rate_fmt}] | {postfix}")
    paused_once = False
    test_acc = None
    for _ in steps:
        train_loss = train(train_loader, model, optimizer, train_criterion, device)
        age = model.get_age()
        if age > 0 and age % eval_every == 0:
            _, test_acc = test(test_loader, model, test_criterion, metric, device)
            if not paused_once:
                converged = plateau.update(test_acc)
                capped = pause_after is not None and age >= int(pause_after)
                if converged or capped:
                    paused_once = True
                    announce_pause(age, test_acc,
                                   "Converged" if converged else "Step budget reached",
                                   notebook_path)
                    # The next train() blocks in its guard until Play.
                    wl.pause_training()
        if tqdm_display:
            parts = [f"loss={train_loss:.4f}"]
            if test_acc is not None:
                parts.append(f"test_acc={test_acc:.1f}%")
            steps.set_postfix_str(" | ".join(parts))

    wl.keep_serving()
