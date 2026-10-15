"""Fashion-MNIST classification with the live 3-D UMAP projection (beta).

A plain PyTorch training loop -- watched model/optimizer/loaders/loss/metric,
guard contexts -- and the built-in parametric UMAP running alongside it. There
is no projection code here: wrapping the model with
``wl.watch_or_edit(..., flag="model")`` attaches it, because it is on by
default. The ``projection=`` dict below only tunes it (see config.yaml).

What you get in Weights Studio, while it trains:

    Data Exploration header -> Projection button opens the 3-D board beside
    the data grid. One point per image, coloured by class. Early on the cloud
    is one tangle; as the representation forms, ten clusters pull apart --
    and the classes that stay mixed (Shirt / T-shirt / Pullover / Coat) are
    exactly the ones the model confuses. Lasso a mixed region to load those
    images into the grid.

    The projection reads the INPUT of the last Linear layer (``fc2``): the
    128-D representation the classifier decides from, never the logits.

Columns ``signals//umap_x|y|z`` appear per sample, so the grid can sort and
filter by them too. The encoder is saved with every checkpoint and restored
with it, so a restart continues the same layout.

To go further -- reload a checkpoint, discard the lowest-loss samples, lay the
rest out with a PCA of your own and compare it with this UMAP -- see the staged
version, ``examples/PyTorch/wl-fashion-mnist-umap``.

Run::

    python main.py                         # then open the studio and press Play
    WEIGHTSLAB_PROJECTION=0 python main.py # the same run, projection removed
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
    collide in the shared ledger.
    """

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
        """Lets the ledger read labels at init without decoding every image."""
        return int(self.data.targets[idx])

    def get_metadata(self, idx):
        label = int(self.data.targets[idx])
        return {"class_name": CLASS_NAMES[label],
                "split": "train" if self.train else "test"}


# =============================================================================
# Model
# =============================================================================
class FashionCNN(nn.Module):
    """Two conv blocks, then a 128-D representation and the classifier.

    ``fc2`` is the last Linear, so the projection hooks ITS input -- the
    128-D output of ``fc1`` after the ReLU. ``embed()`` returns the same
    tensor, for anyone who wants it outside the projection.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.input_shape = (1, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)                      # 28 -> 14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)                      # 14 -> 7
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def embed(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        return self.relu3(self.fc1(self.flatten(x)))

    def forward(self, x):
        # Logits, not softmax: the watched CrossEntropyLoss applies its own.
        return self.fc2(self.embed(x))


# =============================================================================
# Train / test
# =============================================================================
def train(loader, model, optimizer, criterion, device):
    """One training step. The projection rides on the criterion call: the
    forward hook captured the features, the criterion supplies the ids."""
    with wl.guard_training_context:
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
    """Full pass over the test split. Test images are PLACED in the cloud by
    the current encoder but never fitted on, so the layout cannot learn from
    data the model is only scored on."""
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


def projection_status():
    tracker = wl.projection.get_tracker()
    if tracker is None:
        return "projection off"
    stats = tracker.stats()
    if stats["disabled_reason"]:
        return f"projection stopped: {stats['disabled_reason']}"
    return f"umap fits={stats['fits']} placed={stats['samples_written']}"


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    start_time = time.time()

    parameters = {}
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    parameters.setdefault("experiment_name", "fashion_mnist_live_umap")
    parameters.setdefault("device", "auto")
    parameters.setdefault("eval_full_to_train_steps_ratio", 300)

    # `weightslab start <dir>` exports WEIGHTSLAB_ROOT_LOG_DIR -- honor it, so
    # this run lands in the directory the dashboard is watching.
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR") or tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]

    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = parameters["device"]
    eval_ratio = parameters.get("eval_full_to_train_steps_ratio", 300)
    tqdm_display = parameters.get("tqdm_display", True)
    enable_h5 = parameters.get("ledger_enable_h5_persistence", True)

    # ---- MODEL -------------------------------------------------------------
    # This is where the projection attaches. Nothing else is needed; the dict
    # only tunes it (every_n_steps, graph_size, ...). `projection=False` would
    # turn it off for this model, `projection="fc1"` would hook another layer.
    model = wl.watch_or_edit(
        FashionCNN(num_classes=len(CLASS_NAMES)).to(device),
        flag="model", device=device,
        projection=parameters.get("projection") or {},
    )
    lr = parameters.get("optimizer", {}).get("lr", 0.001)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=lr), flag="optimizer")

    # ---- DATA --------------------------------------------------------------
    data_root = parameters.get("data_root") or os.path.join(log_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})
    to_tensor = transforms.Compose([transforms.ToTensor()])

    train_dataset = FashionMNISTDataset(
        root=data_root, train=True, transform=to_tensor,
        max_samples=train_cfg.get("max_samples"), id_base=0)
    test_dataset = FashionMNISTDataset(
        root=data_root, train=False, transform=to_tensor,
        max_samples=test_cfg.get("max_samples"), id_base=1_000_000)

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
    print(" FASHION-MNIST + LIVE 3-D UMAP PROJECTION (beta)")
    print(f" train={len(train_dataset)}  test={len(test_dataset)}  device={device}")
    print(f" {projection_status()} | eval every {eval_ratio} steps")
    print(f" logs -> {log_dir}")
    print(" Studio: press Play, then Data Exploration -> Projection.")
    print("=" * 72 + "\n")

    # Uncomment to start without pressing Play in the studio.
    # wl.start_training()

    # Training runs until YOU stop it (studio pause button, CLI, Ctrl+C).
    steps = itertools.count()
    if tqdm_display:
        steps = tqdm.tqdm(steps, desc="Training", ncols=140,
                          bar_format="{desc}: {n} steps [{elapsed}, {rate_fmt}] | {postfix}")
    test_loss = test_acc = None
    for _ in steps:
        train_loss = train(train_loader, model, optimizer, train_criterion, device)
        age = model.get_age()
        if age > 0 and age % eval_ratio == 0:
            test_loss, test_acc = test(test_loader, model, test_criterion, metric, device)
        if tqdm_display:
            parts = [f"loss={train_loss:.4f}"]
            if test_acc is not None:
                parts.append(f"test_acc={test_acc:.1f}%")
            parts.append(projection_status())
            steps.set_postfix_str(" | ".join(parts))

    wl.keep_serving()
