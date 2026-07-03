import os
import time
import numpy as np
import tqdm
import yaml
import torch
import logging
import tempfile
import itertools

import weightslab as wl

from torch import optim


from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

from utils.data import BDD100kSegDataset, seg_collate
from utils.model import SmallUNet
from utils.criterions import (
    PerSampleDice, PerInstanceDice,
    PerSampleBCE, PerInstanceBCE,
)

# Setup loggers
logging.basicConfig(level=logging.ERROR)
logging.getLogger("PIL").setLevel(logging.INFO)


# =============================================================================
# Train / Test loops (segmentation, using watcher-wrapped loaders)
# =============================================================================

def _instance_batch_idx(labels):
    """Flat instance→sample map (sample-major) matching the PerInstance* ordering."""
    return torch.tensor(
        [s for s, insts in enumerate(labels) for _ in insts],
        dtype=torch.long,
    )


def _run_instance_signals(sig, outputs, labels, ids, preds, return_metric=False):
    """Compute + log/save the per-sample AND per-instance Dice (metric) and BCE (loss)."""
    bce_sample = sig["bce_sample"](outputs, labels, batch_ids=ids, preds=preds)
    dice_sample = sig["dice_sample"](outputs, labels, batch_ids=ids) # Register processed predictions one time only

    sig["dice_instance"](outputs, labels, batch_ids=ids) # Register processed predictions one time only
    sig["bce_instance"](outputs, labels, batch_ids=ids)

    avg_loss = 0.5 * dice_sample + 0.5 * bce_sample
    wl.save_signals({"combined_bce_dice_per_sample": avg_loss}, ids) # Save the per-sample aggregate loss for backward step
    if return_metric:
        return avg_loss, dice_sample
    return avg_loss


def _user_custom_signals(preds, labels):
    """Example of user-defined custom signals to save additional info to WL."""
    return {
        "preds_classes_per_sample": [
            preds[i].unique() for i in range(preds.shape[0])
        ],
        "target_classes_per_sample": [
            torch.unique(torch.cat(labels[i])) for i in range(len(labels))
        ],
        "tp_classes_per_sample": [
            torch.tensor([c for c in torch.unique(torch.cat(labels[i])) if c in preds[i].unique()])
            for i in range(len(labels))
        ],
        "fp_classes_per_sample": [
            torch.tensor([c for c in preds[i].unique() if c not in torch.unique(torch.cat(labels[i]))])
            for i in range(len(labels))
        ],
        "fn_classes_per_sample": [
            torch.tensor([c for c in torch.unique(torch.cat(labels[i])) if c not in preds[i].unique()])
            for i in range(len(labels))
        ],
    }


def train(loader, model, optimizer, sig, device):
    """
    Single training step using the tracked dataloader + watched loss.

    loader yields (inputs, ids, labels, metadata) because of DataSampleTrackingWrapper.
    `labels` is per sample a LIST of instance masks (see utils/data.seg_collate).
    """
    with guard_training_context:
        (inputs, ids, labels, _) = next(loader)
        inputs = inputs.to(device)
        labels = [[m.to(device) for m in insts] for insts in labels] # per-sample list of instances

        optimizer.zero_grad()
        outputs = model(inputs) # [B,C,H,W]
        preds = outputs.argmax(dim=1) # [B,H,W]

        # Per-instance + per-sample Dice/BCE (tracked & saved at annotation level).
        loss_per_sample = _run_instance_signals(sig, outputs, labels, ids, preds=preds)

        # Backward loss: per-sample CrossEntropy over the merged semantic mask.
        loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()

        # I want to see in the UI the per-sample classes predicted by the model and what classes are missing compared to the target (for error analysis)
        wl.save_signals(
            _user_custom_signals(preds, labels),
            ids
        ) # Save the per-sample predictions for visualization

    return float(loss.detach().cpu().item())


def test(loader, model, sig, device, test_loader_len):
    """Full evaluation pass over the val loader."""
    losses = 0.0
    dices = 0.0
    with guard_testing_context, torch.no_grad():
        for inputs, ids, labels, _ in loader:
            inputs = inputs.to(device)
            labels = [[m.to(device) for m in insts] for insts in labels] # per-sample list of instances

            outputs = model(inputs)
            preds = outputs.argmax(dim=1) # [B,H,W]

            # Per-instance + per-sample Dice/BCE (tracked & saved at annotation level).
            loss_per_sample, dice_sample = _run_instance_signals(sig, outputs, labels, ids, preds=preds, return_metric=True)

            losses += torch.mean(loss_per_sample) # Average over the batch and accumulate
            dices += torch.mean(dice_sample) # Average over the batch and accumulate

            # I want to see in the UI the per-sample classes predicted by the model
            wl.save_signals(_user_custom_signals(preds, labels), ids) # Save the per-sample predictions for visualization

    loss = float((losses / test_loader_len).detach().cpu().item())
    dice = float((dices / test_loader_len).detach().cpu().item())
    return loss, dice * 100.0 # Return average Dice as percentage


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "bdd_segmentation")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 500)
    parameters.setdefault("eval_full_to_train_steps_ratio", 50)
    parameters.setdefault("number_of_workers", 4)
    parameters.setdefault("num_classes", 6) # adjust to your label set
    parameters.setdefault("class_names", None) # adjust to your label set
    parameters.setdefault("ignore_index", 255) # if you have void pixels
    parameters.setdefault("image_size", 256)
    parameters.setdefault("compute_natural_sort", True)

    # --- 4) Register hyperparameters ---
    exp_name = parameters["experiment_name"]
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )
    num_classes = int(parameters["num_classes"])
    class_names = parameters["class_names"]
    ignore_index = int(parameters["ignore_index"])
    image_size = int(parameters["image_size"])

    # --- 2) Device selection ---
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = parameters["device"]

    # --- 3) Logging directory ---
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified, using temporary directory: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    log_dir = parameters["root_log_dir"]
    max_steps = parameters["training_steps_to_do"]
    eval_full_to_train_steps_ratio = parameters["eval_full_to_train_steps_ratio"]
    write_export_ratio = parameters.get("write_export_ratio", 100)
    verbose = parameters.get("verbose", True)
    tqdm_display = parameters.get("tqdm_display", True)

    # --- 5) Data (BDD100k reduced) ---
    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "BDD_subset")
    )
    data_root = parameters.get("data_root", default_data_root)
    if os.path.exists(data_root):
        print(f"Using data root: {data_root}")
    else:
        data_root = default_data_root
        print(f"Data root not found, using default: {data_root}")

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    _train_dataset = BDD100kSegDataset(
        root=data_root,
        split="train",
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=ignore_index,
        image_size=image_size,
        max_samples=train_cfg.get("max_samples", None) # Optionally limit number of samples for faster testing
    )
    _val_dataset = BDD100kSegDataset(
        root=data_root,
        split="val",
        num_classes=num_classes,
        class_names=class_names,
        ignore_index=ignore_index,
        image_size=image_size,
        max_samples=test_cfg.get("max_samples", None) # Optionally limit number of samples for faster testing
    )

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=train_cfg.get("shuffle", True),
        compute_hash=False,
        is_training=True,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        preload_labels=False,
        collate_fn=seg_collate,
    )
    test_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=test_cfg.get("batch_size", 2),
        shuffle=test_cfg.get("shuffle", False),
        compute_hash=False,
        is_training=False,
        array_autoload_arrays=False,
        array_return_proxies=True,
        array_use_cache=True,
        preload_labels=True,
        collate_fn=seg_collate,
    )

    # --- 6) Model, optimizer, losses, metric ---
    _model = SmallUNet(
        in_channels=3, num_classes=num_classes, image_size=image_size
    ).to(device)
    model = wl.watch_or_edit(
        _model,
        flag="model",
        device=device,
        compute_dependencies=True
    )
    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(
        _optimizer,
        flag="optimizer",
    )

    # --- Per-instance + per-sample Dice (metric) and BCE (loss) signals ---
    # Each instance mask is scored against the model's per-class probability map.
    # per_instance=True auto-saves one value per (sample_id, annotation_id);
    # per_sample=True logs the per-sample aggregate (mean over the sample's instances).
    def _make_seg_signals(split: str, weights: dict = None) -> dict:
        return {
            "dice_sample": wl.watch_or_edit(
                PerSampleDice(), flag="metric",
                name=f"{split}_dice/sample", per_sample=True, log=True,
            ),
            "dice_instance": wl.watch_or_edit(
                PerInstanceDice(), flag="metric",
                name=f"{split}_dice/instance", per_instance=True, log=True,
            ),
            "bce_sample": wl.watch_or_edit(
                PerSampleBCE(weights=weights), flag="loss",
                name=f"{split}_bce/sample", per_sample=True, log=True,
            ),
            "bce_instance": wl.watch_or_edit(
                PerInstanceBCE(weights=weights), flag="loss",
                name=f"{split}_bce/instance", per_instance=True, log=True,
            ),
        }

    # Compute class weights firstg
    def compute_class_weights(dataset, num_classes, ignore_index=255, max_samples=100):
        print("\n" + "=" * 60, flush=True)
        print(f"Computing class weights for {num_classes} classes (max {max_samples} samples)...", flush=True)
        class_counts = np.zeros(num_classes, dtype=np.float64)
        num_samples = min(len(dataset), max_samples)

        for idx in tqdm.tqdm(range(num_samples), desc=" Analyzing Distribution"):
            _, _, label, _ = dataset.get_items(idx, include_labels=True) # Get the label/mask for this sample
            label_np = label.numpy() if hasattr(label, 'numpy') else np.array(label)
            for c in range(num_classes):
                class_counts[c] += (label_np == c).sum()

        class_counts = np.maximum(class_counts, 1) # Avoid div by zero
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts)
        class_weights = class_weights / class_weights.mean() # Normalize

        print("\nClass distribution and weights:", flush=True)
        for c in range(num_classes):
            pct = (class_counts[c] / total_pixels) * 100
            print(f"Class {c}: {pct:6.2f}% -> weight: {class_weights[c]:.3f}", flush=True)
        print("=" * 60 + "\n", flush=True)
        return torch.FloatTensor(class_weights).to(device)

    weights = compute_class_weights(_train_dataset, num_classes)

    train_sig = _make_seg_signals("train", weights=weights)
    test_sig = _make_seg_signals("test", weights=weights)

    # --- 7) Start WeightsLab services ---
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", True),
    )

    print("=" * 60)
    print(" STARTING BDD100k SEGMENTATION TRAINING")
    print(f" Total steps: {max_steps}")
    print(f" Evaluation every {eval_full_to_train_steps_ratio} steps")
    print(f" Logs will be saved to: {log_dir}")
    print(f" Data root: {data_root}")
    print("=" * 60 + "\n")

    # ================
    # Training Loop
    wl.start_training(timeout=3) # This will block and keep the main thread alive while background services run. You can optionally set a timeout (in seconds) to automatically stop after a certain duration.

    # ================
    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loss, test_metric = None, None
    start_time = time.time()
    for train_step in train_range:
        age = model.get_age() if hasattr(model, "get_age") else train_step # Get model age in steps (not necessarily equal to train_step if model was reloaded or has seen more data than training steps)

        # Train
        train_loss = train(train_loader, model, optimizer, train_sig, device)

        # Test
        if age == 0 or age % eval_full_to_train_steps_ratio == 0:
            test_loader_len = len(test_loader) # Store length before wrapping with tqdm
            test_loader_it = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display else test_loader
            test_loss, test_metric = test(test_loader_it, model, test_sig, device, test_loader_len)

        # Periodic history + dataframe export (JSON/CSV snapshots to root_log_dir)
        if age > 0 and age % write_export_ratio == 0:
            wl.write_history()
            wl.write_dataframe()

        # Verbose
        if verbose and not tqdm_display:
            print(
                "Training.. " +
                f"Step {train_step} (Age {age}): " +
                f"| Train Loss: {train_loss:.4f} " +
                (f"| Test Loss: {test_loss:.4f} " if test_loss is not None else '') +
                (f"| Test Acc mlt: {test_metric:.2f}% " if test_metric is not None else '')
            )
        elif tqdm_display:
            train_range.set_description("Step")
            train_range.set_postfix(
                train_loss=f"{train_loss:.4f}",
                test_loss=f"{test_loss:.4f}" if test_loss is not None else "N/A",
                acc=f"{test_metric:.2f}%" if test_metric is not None else "N/A"
            )

    print("\n" + "=" * 60)
    print(f" Training completed in {time.time() - start_time:.2f} seconds")
    print(f" Logs saved to: {log_dir}")
    print("=" * 60)

    # Final export of signal history and data grid to root_log_dir
    wl.write_history()
    wl.write_dataframe()

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
