"""
Face Recognition Training with Triplet Loss
============================================

Toy example using a pretrained ResNet-18 backbone + lightweight embedding head,
trained with online batch-hard triplet loss on the Olivetti Faces dataset.

Dataset options (set in config.yaml -> data.dataset_type):
  "olivetti"  - sklearn Olivetti (40 ids, 400 imgs) - works offline, default
  "lfw"       - LFW People via torchvision (download required)
  "folder"    - any ImageFolder-style directory

Training flow
-------------
1. Load dataset via FaceDataset -> wrap with wl.watch_or_edit (data flag)
2. Build FaceEmbeddingModel (pretrained backbone, frozen by default)
3. Register hyperparameters with wl.watch_or_edit (hyperparameters flag)
4. Open-ended train loop: forward -> batch-hard triplets -> triplet loss -> backprop
5. Periodic evaluation: verification + retrieval + similarity grouping signals
6. wl.serve() to expose metrics; wl.keep_serving() keeps process alive
"""

import logging
import os
import tempfile

import numpy as np
import torch
import yaml

import weightslab as wl

from typing import Any, Dict, List

from face.data import FaceDataset
from face.model import FaceEmbeddingModel
from face.signals import FaceMetrics
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    model: FaceEmbeddingModel,
    loader: torch.utils.data.DataLoader,
    name: str = "test",
) -> Dict[str, float]:
    """Collect embeddings from loader and compute face recognition metrics."""
    print(f"\n{'=' * 55}")
    print(f"Evaluation [{name}]")
    print(f"{'=' * 55}")

    all_embeddings: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_uids: List[str] = []

    for images, uids, labels, _metadata in loader:
        emb = model.get_embeddings(images)  # (B, D)
        all_embeddings.append(emb.numpy())
        if isinstance(labels, torch.Tensor):
            all_labels.append(labels.numpy())
        else:
            all_labels.append(np.array(labels))
        all_uids.extend(uids if isinstance(uids, list) else list(uids))

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = FaceMetrics.compute_all_metrics(
        ids=all_uids,
        embeddings=all_embeddings,
        labels=all_labels,
        name=name,
    )

    print(f"  verification_accuracy : {metrics.get('verification_accuracy', float('nan')):.4f}")
    print(f"  rank1_accuracy        : {metrics.get('rank1_accuracy', float('nan')):.4f}")
    print(f"  FAR                   : {metrics.get('far', float('nan')):.4f}")
    print(f"  FRR                   : {metrics.get('frr', float('nan')):.4f}")
    print(f"  best_threshold        : {metrics.get('best_threshold', float('nan')):.4f}")
    if "num_clusters" in metrics:
        print(f"  num_clusters          : {metrics['num_clusters']:.0f}")
        print(f"  noise_ratio           : {metrics['noise_ratio']:.4f}")
        print(f"  mean_nn1_distance     : {metrics['mean_nn1_distance']:.4f}")

    return metrics


# =============================================================================
# Training loop (open-ended)
# =============================================================================

def train(
    model: FaceEmbeddingModel,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    eval_every: int = 50,
    log_every: int = 10,
    loss_name: str = "triplet",
) -> Dict[str, Any]:
    """Open-ended training loop.

    Runs indefinitely until interrupted (Ctrl+C). Periodic test evaluation is
    performed every eval_every steps when test_loader is provided.
    """
    print("\n" + "=" * 60)
    print("Face Recognition Training  (open-ended while loop)")
    print(f"  Loss       : {loss_name}")
    print(f"  Eval every : {eval_every} steps")
    print("  Max steps  : infinite (stop with Ctrl+C)")
    print("=" * 60)

    data_iter = iter(train_loader)
    losses: List[float] = []
    eval_history: List[Dict[str, Any]] = []
    step = 0

    try:
        while True:
            step += 1

            with guard_training_context:
                # ---- Fetch next batch (cycle loader) ----
                try:
                    images, batch_ids, labels, _metadata = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    images, batch_ids, labels, _metadata = next(data_iter)

                if not isinstance(batch_ids, list):
                    batch_ids = list(batch_ids)

                if isinstance(labels, torch.Tensor):
                    labels_tensor = labels.long()
                else:
                    labels_tensor = torch.tensor(labels, dtype=torch.long)

                # ---- Gradient step ----
                loss_val = model.train_step(
                    images=images,
                    labels=labels_tensor,
                    batch_ids=batch_ids,
                    loss_name=loss_name,
                )
                losses.append(loss_val)

                if step == 1 or step % max(1, log_every) == 0:
                    window = losses[-log_every:] if len(losses) >= log_every else losses
                    print(
                        f"[train] step {step:>7d} "
                        f"| loss={loss_val:.6f} "
                        f"| running_mean={np.mean(window):.6f}"
                    )

                should_eval = (
                    test_loader is not None
                    and (step % eval_every == 0)
                )

            if should_eval:
                with guard_testing_context:
                    print(f"\n[eval@test] step {step}")
                    metrics = evaluate(model=model, loader=test_loader, name="test")
                    eval_history.append({"step": step, "metrics": metrics})

    except KeyboardInterrupt:
        print("\nInterrupted by user. Stopping training loop.")

    summary = {
        "train_steps": step,
        "train_loss_mean": float(np.mean(losses)) if losses else float("nan"),
        "train_loss_last": float(losses[-1]) if losses else float("nan"),
        "num_evals": len(eval_history),
        "stopped_by_user": True,
    }

    print("\nTraining summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # ---- Load config ----
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "face_triplet_training")
    parameters.setdefault("device", "auto")
    parameters.setdefault("eval_every", 50)
    parameters.setdefault("log_every", 10)

    # ---- Device ----
    if parameters.get("device", "auto") == "auto":
        parameters["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    device = parameters["device"]

    # ---- Log dir ----
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = tmp_dir
        print(f"No root_log_dir specified - using temp dir: {parameters['root_log_dir']}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    eval_every = int(parameters.get("eval_every", 50))
    log_every = int(parameters.get("log_every", 10))
    enable_h5 = parameters.get("enable_h5_persistence", True)

    # ---- Hyperparameter tracking ----
    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        defaults=parameters,
        poll_interval=1.0,
    )

    # ---- Datasets ----
    data_cfg = parameters.get("data", {})
    dataset_type = data_cfg.get("dataset_type", "olivetti")
    root_data = data_cfg.get("root_data_dir", parameters.get("root_log_dir", "."))
    image_size = int(data_cfg.get("image_size", 64))
    min_imgs = int(data_cfg.get("min_images_per_class", 2))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))

    train_dataset = FaceDataset(
        root=root_data,
        dataset_type=dataset_type,
        split="train",
        image_size=image_size,
        train_ratio=train_ratio,
        min_images_per_class=min_imgs,
    )
    test_dataset = FaceDataset(
        root=root_data,
        dataset_type=dataset_type,
        split="test",
        image_size=image_size,
        train_ratio=train_ratio,
        min_images_per_class=min_imgs,
    )

    print(
        f"\nDataset : {dataset_type}"
        f" | train={len(train_dataset)}"
        f" | test={len(test_dataset)}"
        f" | classes={train_dataset.num_classes}"
    )

    train_cfg = data_cfg.get("train_loader", {})
    test_cfg = data_cfg.get("test_loader", {})

    # ---- WeightsLAB-tracked data loaders ----
    train_loader = wl.watch_or_edit(
        train_dataset,
        flag="data",
        loader_name="train_loader",
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=train_cfg.get("shuffle", True),
        is_training=True,
        compute_hash=False,
        num_workers=int(train_cfg.get("n_workers", 0)),
        enable_h5_persistence=enable_h5,
    )
    test_loader = wl.watch_or_edit(
        test_dataset,
        flag="data",
        loader_name="test_loader",
        batch_size=int(test_cfg.get("batch_size", 64)),
        shuffle=test_cfg.get("shuffle", False),
        is_training=False,
        compute_hash=False,
        num_workers=int(test_cfg.get("n_workers", 0)),
        enable_h5_persistence=enable_h5,
    )

    # ---- Model ----
    model_cfg = parameters.get("model", {})
    model = FaceEmbeddingModel(
        backbone_name=model_cfg.get("backbone", "resnet18"),
        embedding_dim=int(model_cfg.get("embedding_dim", 128)),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", 256)),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        pretrained=model_cfg.get("pretrained", True),
        margin=float(model_cfg.get("margin", 0.3)),
        device=device,
    )

    # ---- Start WeightsLAB services ----
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", False),
    )

    print("\n" + "=" * 60)
    print("STARTING FACE RECOGNITION TRAINING")
    print(f"  Experiment   : {parameters['experiment_name']}")
    print(f"  Device       : {device}")
    print(f"  Steps        : infinite  |  eval_every={eval_every}")
    print(f"  Loss         : {model_cfg.get('loss', 'triplet')}")
    print(f"  Logs         : {parameters['root_log_dir']}")
    print("=" * 60)

    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        eval_every=eval_every,
        log_every=log_every,
        loss_name=model_cfg.get("loss", "triplet"),
    )

    print("\n" + "=" * 60)
    print("TRAINING STOPPED")
    print(f"Logs saved to: {parameters['root_log_dir']}")
    print("=" * 60)

    wl.keep_serving()
