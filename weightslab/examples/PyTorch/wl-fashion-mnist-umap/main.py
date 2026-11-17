"""Fashion-MNIST classification, explored with projections: the built-in
parametric UMAP first, then a PCA of your own to compare it with.

A plain PyTorch loop, driven in stages so you can look at the Projection board
between them (Weights Studio -> Data Exploration -> Projection):

1. **Train** ``steps_initial`` steps (default 1000) with the live 3-D parametric
   UMAP running beside the model -- there is no projection code in the loop,
   wrapping the model attaches it -- and keep a checkpoint of those weights.
2. **Reload** that checkpoint: weights, optimizer, the UMAP encoder and every
   sample's loss as they were at step 1000. (If you let the run go on, this is
   the way back to the weights the next stages are about.)
3. **Discard** the ``discard_fraction`` (default 30 %) of training samples with
   the lowest loss: the ones the model already gets right, which teach it little.
4. **PCA, 2-D**, written in plain torch below, of what is left -- stored as the
   projection ``pca2d_step<N>``. WeightsLab collects the features;
   ``wl.project_dataset(model, loader, method=pca_coordinates)`` hands them to your
   code.
5. **Compare** the UMAP and the PCA: in the board's picker, and here, with a
   side-by-side picture and a number (how often a point's nearest neighbours in
   the picture share its class).
6. **Retrain** ``steps_retrain`` steps on what is left; the live UMAP follows
   the model, the PCA stays the snapshot you took.

Run::

    python main.py                       # then open the studio (weightslab start)
    python main.py --steps 200 100       # a quick pass: 200 initial + 100 retrain
    python main.py --wait                # stop for Enter before retraining, to look
"""

import argparse
import logging
import os
import ssl
import tempfile

# Windows SSL fix: some Windows cert stores contain malformed ASN1 certs that
# crash ssl.create_default_context(). Fall back to unverified only when broken.
try:
    ssl.create_default_context()
except ssl.SSLError:
    ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import yaml
from torch.utils.data import Dataset
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms

import weightslab as wl
from weightslab.backend import ledgers

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
LOSS_COLUMN = "signals//train-loss-CE"
TRAIN_SPLIT = "train_loader"


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

    ``fc2`` is the last Linear, so the projection hooks ITS input -- the 128-D
    output of ``fc1`` after the ReLU, the representation the classifier decides
    from (never the logits).
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

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        # Logits, not softmax: the watched CrossEntropyLoss applies its own.
        return self.fc2(self.relu3(self.fc1(self.flatten(x))))


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
    """Full pass over the test split. Test images are PLACED in the cloud by the
    current encoder but never fitted on, so the layout cannot learn from data
    the model is only scored on."""
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
# The stages' building blocks
# =============================================================================
def sample_table(origin=TRAIN_SPLIT):
    """The per-sample table of one split: one row per sample, ``sample_id`` as
    the index, with the signals (``signals//...``), ``target``, ``discarded``."""
    wl.drain_signals()                                    # flush pending writes
    df = ledgers.get_dataframe().get_df_view()
    if "annotation_id" in (df.index.names or []):
        df = df.xs(0, level="annotation_id")              # the sample-level rows
    if "origin" in df.columns:
        df = df[df["origin"] == origin]
    # The ledger hands ids back as strings and targets as objects; this example's
    # ids and classes are numbers, and everything below indexes by them.
    df = df.copy()
    df.index = pd.Index([int(i) for i in df.index], name="sample_id")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    return df


def lowest_loss_ids(table, fraction, column=LOSS_COLUMN):
    """Sample ids of the *fraction* of rows with the lowest loss.

    Rows with no loss yet are never picked: an unseen sample is not an easy one.
    Returns plain ints, ready for ``wl.discard_samples``.
    """
    if column not in table.columns or fraction <= 0:
        return []
    loss = pd.to_numeric(table[column], errors="coerce").dropna()
    count = int(len(loss) * min(float(fraction), 1.0))
    return [int(i) for i in loss.nsmallest(count).index] if count else []


def save_checkpoint():
    """Write a checkpoint of the model as it is now; returns its step."""
    manager = ledgers.get_checkpoint_manager()
    if manager.save_model_checkpoint(force_dump_pending=True, update_manifest=True) is None:
        raise RuntimeError("could not write a checkpoint (has training started?)")
    return int(ledgers.get_model().get_age())


def reload_checkpoint(step):
    """Roll the experiment back to the checkpoint written at *step*: weights,
    optimizer, the UMAP encoder, and every sample's loss / seen-count as they
    were then. Returns the model's age afterwards."""
    manager = ledgers.get_checkpoint_manager()
    if not manager.load_state(manager.get_current_experiment_hash(), target_step=step):
        raise RuntimeError(f"could not reload the checkpoint of step {step}")
    return int(ledgers.get_model().get_age())


def pca_coordinates(features, dim=2):
    """PCA by SVD, in plain torch: ``(N, F)`` features -> ``(N, dim)``
    coordinates, and the share of the variance those ``dim`` axes keep."""
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    x = x - x.mean(0)
    _, singular, axes = torch.linalg.svd(x, full_matrices=False)
    variance = singular ** 2
    return (x @ axes[:dim].T).numpy(), float(variance[:dim].sum() / variance.sum())


def project_with_pca(model, loader, prefix, dim=2):
    """Store a PCA of the model's representation as the projection *prefix*.

    WeightsLab runs the loader through the model and collects the features; the
    algorithm is the function below -- anything returning ``(N, 2|3)`` works.
    Returns ``(stats, features, variance_kept)``; ``features[n]`` belongs to the
    sample ``stats["sample_ids"][n]``.
    """
    kept = {}

    def pca(features):
        coords, kept["variance"] = pca_coordinates(features, dim)
        kept["features"] = features
        return coords

    stats = wl.project_dataset(model, loader, method=pca, prefix=prefix, verbose=False)
    return stats, kept["features"], kept["variance"]


def knn_purity(coords, labels, k=10):
    """Mean share of each point's *k* nearest neighbours (in ``coords``) that
    have its class: 1.0 = every neighbourhood is one class, ~0.1 = ten classes
    thrown together. Reads how well a picture keeps the classes apart."""
    points = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(labels))
    k = min(int(k), len(points) - 1)
    if k < 1:
        return float("nan")
    same = 0.0
    for start in range(0, len(points), 2048):             # rows at a time: N x N never exists
        block = torch.cdist(points[start:start + 2048], points)
        block[torch.arange(block.shape[0]), torch.arange(start, start + block.shape[0])] = float("inf")
        neighbours = block.topk(k, largest=False).indices
        same += (labels[neighbours] == labels[start:start + block.shape[0], None]).float().sum().item()
    return same / (len(points) * k)


def layout(table, prefix):
    """``(ids, coords)`` of the kept samples *prefix* has placed (NaN rows and
    discarded samples left out)."""
    columns = [f"signals//{prefix}_{axis}" for axis in "xyz"
               if f"signals//{prefix}_{axis}" in table.columns]
    if not columns:
        return [], np.empty((0, 0))
    rows = table[~table["discarded"].astype(bool)]
    coords = rows[columns].apply(pd.to_numeric, errors="coerce")
    placed = coords.notna().all(axis=1)
    return list(rows.index[placed]), coords[placed].to_numpy(dtype=np.float32)


def compare(table, prefixes, k=10):
    """k-NN class purity of each projection in *prefixes*, over the kept samples
    that ALL of them place -- the same points in every picture."""
    layouts = {p: layout(table, p) for p in prefixes}
    common = set.intersection(*(set(ids) for ids, _ in layouts.values()))
    targets = table["target"]
    scores = {}
    for prefix, (ids, coords) in layouts.items():
        keep = [n for n, i in enumerate(ids) if i in common]
        scores[prefix] = knn_purity(coords[keep], targets.loc[[ids[n] for n in keep]].to_numpy(), k)
    return scores, len(common)


def plot_side_by_side(table, prefixes):
    """A figure with the projections of *prefixes* next to each other, coloured
    by class (3-D ones as 3-D axes). The caller shows or saves it."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6 * len(prefixes), 5.5))
    for slot, prefix in enumerate(prefixes, 1):
        ids, coords = layout(table, prefix)
        three_d = coords.shape[1] == 3
        ax = fig.add_subplot(1, len(prefixes), slot, projection="3d" if three_d else None)
        classes = table["target"].loc[ids].to_numpy()
        for c, name in enumerate(CLASS_NAMES):
            m = classes == c
            ax.scatter(*(coords[m, a] for a in range(coords.shape[1])), s=2, label=name)
        ax.set_title(prefix)
        ax.set_xticks([]); ax.set_yticks([])
        if three_d:
            ax.set_zticks([])
        if slot == 1:
            ax.legend(markerscale=5, fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def save_side_by_side(table, prefixes, path):
    """:func:`plot_side_by_side`, written to *path* without opening a window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plot_side_by_side(table, prefixes)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# =============================================================================
# The run
# =============================================================================
def load_parameters(config_path=None):
    config_path = config_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "config.yaml")
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def run(parameters, steps=None, wait=False, keep_serving=False):
    """Run the staged exploration described at the top of this file.

    *parameters* is the config dict (``config.yaml``); *steps* overrides
    ``(steps_initial, steps_retrain)``. Returns a summary dict -- checkpoint
    step, how many samples were discarded, the purity of each projection.
    """
    explore = {"steps_initial": 1000, "steps_retrain": 1000, "discard_fraction": 0.3,
               "knn_k": 10}
    explore.update(parameters.get("exploration") or {})
    if steps:
        explore["steps_initial"], explore["steps_retrain"] = steps
    parameters.setdefault("experiment_name", "fashion_mnist_umap_vs_pca")
    parameters.setdefault("device", "auto")
    parameters["training_steps_to_do"] = None

    # `weightslab start <dir>` exports WEIGHTSLAB_ROOT_LOG_DIR -- honor it, so
    # this run lands in the directory the dashboard is watching.
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = os.environ.get("WEIGHTSLAB_ROOT_LOG_DIR") or tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]
    wl.watch_or_edit(parameters, flag="hyperparameters", poll_interval=1.0)

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(parameters["device"])
    eval_every = parameters.get("eval_full_to_train_steps_ratio", 250)
    enable_h5 = parameters.get("ledger_enable_h5_persistence", True)

    # ---- MODEL: this is where the live UMAP attaches ------------------------
    model = wl.watch_or_edit(
        FashionCNN(num_classes=len(CLASS_NAMES)).to(device), flag="model", device=device,
        projection=parameters.get("projection") or {})
    lr = parameters.get("optimizer", {}).get("lr", 0.001)
    optimizer = wl.watch_or_edit(optim.Adam(model.parameters(), lr=lr), flag="optimizer")

    # ---- DATA ---------------------------------------------------------------
    data_root = parameters.get("data_root") or os.path.join(log_dir, "data")
    os.makedirs(data_root, exist_ok=True)
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})
    to_tensor = transforms.ToTensor()
    train_dataset = FashionMNISTDataset(root=data_root, train=True, transform=to_tensor,
                                        max_samples=train_cfg.get("max_samples"), id_base=0)
    test_dataset = FashionMNISTDataset(root=data_root, train=False, transform=to_tensor,
                                       max_samples=test_cfg.get("max_samples"),
                                       id_base=1_000_000)
    train_loader = wl.watch_or_edit(
        train_dataset, flag="data", loader_name=TRAIN_SPLIT,
        batch_size=train_cfg.get("batch_size", 64), shuffle=train_cfg.get("shuffle", True),
        is_training=True, compute_hash=False, preload_labels=True, preload_metadata=True,
        enable_h5_persistence=enable_h5)
    test_loader = wl.watch_or_edit(
        test_dataset, flag="data", loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 256), shuffle=test_cfg.get("shuffle", False),
        is_training=False, compute_hash=False, preload_labels=True, preload_metadata=True,
        enable_h5_persistence=enable_h5)

    # ---- LOSS / METRIC ------------------------------------------------------
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
    wl.start_training(timeout=parameters.get("start_timeout", 3))

    def train_to(age):
        """Train until the model's age reaches *age*; evaluate every so often."""
        bar = tqdm.tqdm(total=max(0, age - model.get_age()), desc="Training", ncols=120,
                        disable=not parameters.get("tqdm_display", True))
        while model.get_age() < age:
            loss = train(train_loader, model, optimizer, train_criterion, device)
            bar.update(1)
            if model.get_age() % eval_every == 0:
                _, accuracy = test(test_loader, model, test_criterion, metric, device)
                bar.set_postfix_str(f"loss={loss:.3f} test_acc={accuracy:.1f}% {projection_status()}")
        bar.close()

    def banner(text):
        print(f"\n[{text}]")

    summary = {"log_dir": log_dir}
    print("=" * 72)
    print(" FASHION-MNIST: LIVE UMAP vs YOUR OWN PCA")
    print(f" train={len(train_dataset)}  test={len(test_dataset)}  device={device}")
    print(f" logs -> {log_dir}")
    print(" Studio: weightslab start, then Data Exploration -> Projection.")
    print("=" * 72)

    # 1 -- train with the live UMAP ---------------------------------------------
    banner(f"1. Train {explore['steps_initial']} steps with the live parametric UMAP")
    train_to(explore["steps_initial"])
    summary["checkpoint_step"] = save_checkpoint()
    print(f"   checkpoint written at step {summary['checkpoint_step']}; {projection_status()}")

    # 2 -- reload those weights -------------------------------------------------
    banner(f"2. Reload the step-{summary['checkpoint_step']} checkpoint")
    summary["reloaded_age"] = reload_checkpoint(summary["checkpoint_step"])
    print(f"   model age after reload: {summary['reloaded_age']}")

    # 3 -- discard what teaches little -----------------------------------------
    banner(f"3. Discard the {explore['discard_fraction']:.0%} lowest-loss training samples")
    table = sample_table()
    easy = lowest_loss_ids(table, explore["discard_fraction"])
    if easy:
        wl.discard_samples(easy, discarded=True)
    summary["discarded"] = len(easy)
    print(f"   discarded {len(easy)} of {len(table)}")
    per_class = table.loc[table.index.isin(easy), "target"].astype(int).value_counts().sort_index()
    print("   by class: " + ", ".join(f"{CLASS_NAMES[c]} {n}" for c, n in per_class.items()))

    # 4 -- your own PCA ----------------------------------------------------------
    pca_prefix = f"pca2d_step{summary['reloaded_age']}"
    banner(f"4. PCA (2-D) of what is left -> '{pca_prefix}'")
    stats, features, variance = project_with_pca(model, train_loader, pca_prefix, dim=2)
    summary["pca_samples"] = stats["samples"]
    print(f"   {stats['samples']} samples x {stats['feature_dim']} features; "
          f"2 axes keep {variance:.0%} of the variance")
    save_checkpoint()                                    # the PCA is kept with this checkpoint

    # 5 -- compare ----------------------------------------------------------------
    banner("5. Compare the UMAP and the PCA")
    table = sample_table()
    scores, shared = compare(table, ["umap", pca_prefix], explore["knn_k"])
    summary["purity"] = scores
    summary["purity_features"] = knn_purity(
        features, table["target"].loc[[int(i) for i in stats["sample_ids"]]].to_numpy(),
        explore["knn_k"])
    print(f"   neighbours sharing a point's class (k={explore['knn_k']}), over {shared} kept samples:")
    print(f"     parametric UMAP (3-D) : {scores['umap']:.3f}")
    print(f"     your PCA (2-D)        : {scores[pca_prefix]:.3f}")
    print(f"     the 128-D features    : {summary['purity_features']:.3f}   (what both pictures summarize)")
    figure = save_side_by_side(table, ["umap", pca_prefix], os.path.join(log_dir, "umap_vs_pca.png"))
    print(f"   side-by-side picture -> {figure}")
    print("   In the studio: Data Exploration -> Projection, pick 'umap' then the PCA.")
    if wait:
        input("   Look at both in the studio, then press Enter to retrain... ")

    # 6 -- retrain on what is left ---------------------------------------------------
    if explore["steps_retrain"] > 0:
        target = summary["reloaded_age"] + explore["steps_retrain"]
        banner(f"6. Retrain to step {target} on the remaining samples")
        train_to(target)
        again = f"pca2d_step{model.get_age()}"
        project_with_pca(model, train_loader, again, dim=2)
        scores, shared = compare(sample_table(), ["umap", again], explore["knn_k"])
        summary["purity_after"] = scores
        summary["final_age"] = int(model.get_age())
        print(f"   step {model.get_age()}: UMAP {scores['umap']:.3f}  |  fresh PCA {scores[again]:.3f}"
              f"  (the live UMAP followed the model; '{pca_prefix}' stayed a snapshot)")

    print("\nDone.")
    if keep_serving:
        wl.keep_serving()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", nargs=2, type=int, metavar=("INITIAL", "RETRAIN"),
                        help="training steps before the discard and after it "
                             "(default: the config's exploration section, 1000 and 1000)")
    parser.add_argument("--wait", action="store_true",
                        help="stop for Enter after the comparison, to look in the studio")
    parser.add_argument("--keep-serving", action="store_true",
                        help="stay up at the end so you can keep using the studio")
    args = parser.parse_args(argv)
    run(load_parameters(), steps=args.steps, wait=args.wait, keep_serving=args.keep_serving)


if __name__ == "__main__":
    main()
