import os
import time
import logging
import tempfile

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
    
    def __init__(self, root, train=True, download=False, transform=None):
        """
        Args:
            root (str): Root directory where MNIST data is stored
            train (bool): If True, use training data; else use test data
            download (bool): If True, download the data if not present
            transform (callable, optional): Optional transform to be applied on images
        """
        # Load the standard MNIST dataset
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll apply transform manually to track filepath
        )
        self.transform = transform
        self.train = train
        self.root = root
        
        # Build filepath mapping for each sample
        self._build_filepath_mapping()
    
    def _build_filepath_mapping(self):
        """Build a mapping of sample index to filepath."""
        self.filepaths = {}
        
        # For each index, construct a meaningful filepath
        # MNIST doesn't have original individual files, so we create virtual paths
        for idx in range(len(self.mnist)):
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
        return len(self.mnist)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, label, filepath)
        """
        image, label = self.mnist[idx]
        
        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Get the filepath for this sample
        metadata = {'filepath': self.filepaths.get(idx, 'unknown')}
        
        return image, idx, label, metadata


# -----------------------------------------------------------------------------
# Train / Test functions
# -----------------------------------------------------------------------------
def train(loader, model, optimizer, criterion_mlt, device):
    """Single training step using the tracked dataloader + watched loss."""
    with guard_training_context:    
        (inputs, ids, labels, _) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Infer
        optimizer.zero_grad()
        preds_raw = model(inputs)

        # Preds
        if preds_raw.ndim == 1:
            preds = (preds_raw > 0.0).long()
        else:
            preds = preds_raw.argmax(dim=1, keepdim=True)

        # Loss is a watched object => pass metadata for logging/stats
        loss_batch_mlt = criterion_mlt(
            preds_raw.float(),
            labels.long(),
            batch_ids=ids,
            preds=preds
        )
        total_loss = loss_batch_mlt.mean()  # Final scalar loss

        # Model
        total_loss.backward()
        optimizer.step()

    return total_loss.detach().cpu().item()


def test(loader, model, criterion_mlt, metric_mlt, device, test_loader_len):
    """Full evaluation pass over the test loader."""
    losses = torch.tensor(0.0, device=device)

    for (inputs, ids, labels, _) in loader:
        with guard_testing_context, torch.no_grad():
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
    parameters.setdefault("training_steps_to_do", 1000)
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

    verbose = parameters.get('verbose', True)
    log_dir = parameters["root_log_dir"]
    tqdm_display = parameters.get("tqdm_display", True)
    eval_every = parameters.get("eval_every", 50)
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

    _train_dataset = MNISTCustomDataset(
        root=data_root,
        train=True,
        download=should_download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
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
    )

    # Read data config for all loaders
    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    train_bs = train_cfg.get("batch_size", 16)
    test_bs = test_cfg.get("batch_size", 16)

    train_shuffle = train_cfg.get("shuffle", True)
    test_shuffle = test_cfg.get("shuffle", False)

    # Create tracked loaders for train, test, and test
    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_bs,
        shuffle=train_shuffle,
        is_training=True,
        compute_hash=True,
        preload_labels=False,
        preload_metadata=False,
        enable_h5_persistence=enable_h5_persistence
    )
    test_loader = wl.watch_or_edit(
        _test_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_bs,
        shuffle=test_shuffle,
        is_training=False,
        compute_hash=True,
        preload_labels=False,
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

    # Start WeightsLab services (gRPC only, no CLI)
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", False),
        serving_cli=parameters.get("serving_cli", False),
        serving_ui=parameters.get("serving_ui", False),
    )

    print("=" * 60)
    print("🚀 STARTING TRAINING")
    print(f"🔄 Evaluation every {eval_every} steps")
    print(f"� Dataset splits: train={len(_train_dataset)}, test={len(_test_dataset)}")
    print(f"💾 Logs will be saved to: {log_dir}")
    print("=" * 60 + "\n")

    # Setup clean progress bar with custom format
    if tqdm_display:
        train_range = tqdm.tqdm(
            range(training_steps_to_do),
            desc="Training",
            bar_format="{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}] {bar} | {postfix}",
            ncols=140,
            position=0,
            leave=True
        )
    else:
        train_range = range(training_steps_to_do)

    test_loader_len = len(test_loader)  # Store length before wrapping with tqdm

    train_loss = None
    test_loss, test_metric = None, None
    for train_step in train_range:
        # Train one step
        train_loss = train(train_loader, model, optimizer, train_criterion, device)

        # Periodic test evaluation
        if train_step > 0 and train_step % eval_every == 0:
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
            msg = f"Step {train_step}: Loss={train_loss:.4f}"
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