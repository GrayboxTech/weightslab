import os
import time
import logging
import tempfile
import itertools

import tqdm
import yaml
import torch
import numpy as np

from torch import nn, optim
from torchvision import transforms
from torchmetrics import JaccardIndex
from torch.utils.data import Dataset
from PIL import Image

import weightslab as wl

from weightslab.utils.logger import LoggerQueue as Logger
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
)


# Setup loggers
logging.basicConfig(level=logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "debug"
logging.getLogger("PIL").setLevel(logging.INFO)


# =============================================================================
# Small UNet-ish segmentation model
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=6, image_size=256):
        super().__init__()
        # For WeightsLab
        self.task_type = "segmentation"
        self.num_classes = num_classes
        self.input_shape = (1, in_channels, image_size, image_size)

        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(32 + 32, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        # ⚠️ Important: no `if` on shapes; always interpolate
        u2 = F.interpolate(u2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        logits = self.head(d1)  # [B, C, H, W]
        return logits


# =============================================================================
# BDD100k segmentation dataset
# =============================================================================
class BDD100kSegDataset(Dataset):
    """
    Uses your existing layout:

      data/BDD100k_reduced/
        images_1280x720/
          train/
          val/
        bdd100k_labels_dac_daa_lls_lld_curbs/
          train/
          val/

    Assumes image & label share basename (e.g. 0001.jpg / 0001.png).
    """

    def __init__(
        self,
        root,
        split="train",
        num_classes=6,
        ignore_index=255,
        image_size=256,
        max_samples=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.task_type = "segmentation"
        self.max_samples = max_samples

        # Directories for images and labels
        img_dir = os.path.join(root, "images", split)
        lbl_dir = os.path.join(root, "labels", split)

        # Find files
        image_files = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_files = sorted(set(image_files))[: max_samples] if max_samples is not None else sorted(set(image_files))

        self.images = []
        self.masks = []
        for fname in image_files:
            img_path = os.path.join(img_dir, fname)
            base, _ = os.path.splitext(fname)
            lbl_name = base + ".png"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(lbl_path):
                self.images.append(img_path)
                self.masks.append(lbl_path)

        if len(self.images) == 0:
            raise RuntimeError(f"No image/label pairs found in {img_dir} / {lbl_dir}")

        # This is used by load_raw_image in DataService / trainer_tools
        # so exposing .images is enough.
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size,
                        image_size
                    ),
                    interpolation=Image.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )
        self.mask_resize = transforms.Resize(
            (image_size, image_size), interpolation=Image.NEAREST
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        IMPORTANT: returns (item, target) only.
        DataSampleTrackingWrapper (from watch_or_edit) will wrap this and
        produce (inputs, ids, labels) and _getitem_raw for you.
        """
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = Image.open(img_path).convert("RGB")
        uid = '.'.join(os.path.basename(img_path).split(".")[:-1])
        mask = Image.open(mask_path)

        img_t = self.image_transform(img)
        mask_r = self.mask_resize(mask)
        mask_np = np.array(mask_r, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np)  # [H, W] int64
        
        return img_t, uid, mask_t


# =============================================================================
# Train / Test loops (segmentation, using watcher-wrapped loaders)
# =============================================================================

def train(loader, model, optimizer, criterion_mlt, device):
    """
    Single training step using the tracked dataloader + watched loss.

    loader yields (inputs, ids, labels) because of DataSampleTrackingWrapper.
    """
    with guard_training_context:
        (inputs, ids, labels) = next(loader)
        inputs = inputs.to(device)
        labels = labels.to(device)  # [B,H,W]

        optimizer.zero_grad()
        outputs = model(inputs)  # [B,C,H,W]

        preds = outputs.argmax(dim=1)  # [B,H,W]

        loss_batch = criterion_mlt(
            outputs.float(),
            labels.long(),
            batch_ids=ids,
            preds=preds,
        )  # CrossEntropyLoss(reduction="none") → [B,H,W]
        loss = loss_batch.mean()

        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu().item())

def test(loader, model, criterion_mlt, metric_mlt, device, test_loader_len):
    """Full evaluation pass over the val loader."""
    losses = 0.0
    metric_mlt.reset()

    with guard_testing_context, torch.no_grad():
        for inputs, ids, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)  # [B,H,W]

            loss_batch = criterion_mlt(
                outputs.float(),
                labels.long(),
                batch_ids=ids,
                preds=preds,
            )
            losses += torch.mean(loss_batch)

            metric_mlt.update(preds, labels)

    loss = float((losses / test_loader_len).detach().cpu().item())
    miou = float(metric_mlt.compute().detach().cpu().item() * 100.0)
    return loss, miou


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
    parameters.setdefault("num_classes", 6)      # adjust to your label set
    parameters.setdefault("ignore_index", 255)   # if you have void pixels
    parameters.setdefault("image_size", 256)

    exp_name = parameters["experiment_name"]
    num_classes = int(parameters["num_classes"])
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
    eval_every = parameters["eval_full_to_train_steps_ratio"]
    verbose = parameters.get("verbose", True)
    tqdm_display = parameters.get("tqdm_display", True)

    # --- 4) Register logger + hyperparameters ---
    logger = Logger()
    wl.watch_or_edit(logger, flag="logger", log_dir=log_dir)

    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        defaults=parameters,
        poll_interval=1.0,
    )

    # --- 5) Data (BDD100k reduced) ---
    default_data_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "BDD100k_reduced")
    )
    data_root = parameters.get("data_root", default_data_root)

    train_cfg = parameters.get("data", {}).get("train_loader", {})
    test_cfg = parameters.get("data", {}).get("test_loader", {})

    _train_dataset = BDD100kSegDataset(
        root=data_root,
        split="train",
        num_classes=num_classes,
        ignore_index=ignore_index,
        image_size=image_size,
        max_samples=100
    )
    _val_dataset = BDD100kSegDataset(
        root=data_root,
        split="val",
        num_classes=num_classes,
        ignore_index=ignore_index,
        image_size=image_size,
        max_samples=100
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
        preload_labels=True,
        preload_uids=True,  # use file names as unique IDs for tracking
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
        preload_uids=True,  # use file names as unique IDs for tracking
    )

    # --- 6) Model, optimizer, losses, metric ---
    model = SmallUNet(
        in_channels=3, num_classes=num_classes, image_size=image_size
    ).to(device)
    model = wl.watch_or_edit(
        model,
        flag="model",
        device=device,
        compute_dependencies=False
    )

    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    optimizer = wl.watch_or_edit(
        optim.Adam(model.parameters(), lr=lr),
        flag="optimizer",
    )

    # --- Compute class weights to handle class imbalance ---
    print("\n" + "=" * 60)
    print("Computing class weights to address class imbalance...")
    print("=" * 60)

    def compute_class_weights(dataset, num_classes, max_samples=1000):
        """
        Compute class weights based on inverse pixel frequency.
        Rare classes get higher weights to balance the loss.
        """
        class_counts = np.zeros(num_classes, dtype=np.float64)

        # Sample up to max_samples images to compute statistics
        num_samples = min(len(dataset), max_samples)
        print(f"Analyzing {num_samples} samples from dataset...")

        for idx in range(num_samples):
            try:
                # The dataset returns (img_t, mask_t)
                _, _, label = dataset[idx]
                label_np = label.numpy() if hasattr(label, 'numpy') else np.array(label)

                # Count pixels for each class
                for c in range(num_classes):
                    class_counts[c] += (label_np == c).sum()
            except Exception as e:
                print(f"Warning: Could not process sample {idx}: {e}")
                continue

        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)

        # Compute inverse frequency weights
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts)

        # Normalize so mean weight is 1.0
        class_weights = class_weights / class_weights.mean()

        print("\nClass distribution and weights:")
        print("-" * 60)
        for c in range(num_classes):
            percentage = (class_counts[c] / total_pixels) * 100
            print(f"  Class {c}: {percentage:6.2f}% of pixels → weight: {class_weights[c]:.3f}")
        print("-" * 60)

        return torch.FloatTensor(class_weights)

    # Compute weights from training dataset
    class_weights = compute_class_weights(_train_dataset, num_classes, max_samples=500)
    class_weights = class_weights.to(device)

    print(f"\nApplying class weights: {class_weights.cpu().numpy()}")
    print("=" * 60 + "\n")

    # Create weighted loss functions
    train_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index,
            weight=class_weights  # ← Class weights applied!
        ),
        flag="loss",
        per_sample=True,
        log=True,
        name="train_loss/CE",
    )
    test_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(
            reduction="none",
            ignore_index=ignore_index,
            weight=class_weights  # ← Class weights applied!
        ),
        name="test_loss/CE",
        flag="loss",
        per_sample=True,
        log=True,
    )
    test_metric_mlt = wl.watch_or_edit(
        JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        ).to(device),
        name="test_metric/Jaccard",
        flag="metric",
        log=True,
    )

    # --- 7) Start WeightsLab services ---
    wl.serve(
        serving_grpc=parameters.get("serving_grpc", True),
        serving_cli=parameters.get("serving_cli", True),
    )

    print("=" * 60)
    print("🚀 STARTING BDD100k SEGMENTATION TRAINING")
    print(f"📈 Total steps: {max_steps}")
    print(f"🔄 Evaluation every {eval_every} steps")
    print(f"💾 Logs will be saved to: {log_dir}")
    print(f"📂 Data root: {data_root}")
    print("=" * 60 + "\n")

    # ================
    # 7. Training Loop
    train_range = tqdm.tqdm(itertools.count(), desc="Training") if tqdm_display else itertools.count()
    test_loader_len = len(test_loader)  # Store length before wrapping with tqdm
    test_loss, test_metric = None, None
    start_time = time.time()
    for train_step in train_range:
        # Train
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt, device)

        # Test
        if train_step % eval_every == 0:
            test_loader = tqdm.tqdm(test_loader, desc="Evaluating") if tqdm_display else test_loader
            test_loss, test_metric = test(test_loader, model, test_criterion_mlt, test_metric_mlt, device, test_loader_len)

        # Verbose
        if verbose and not tqdm_display:
            print(
                "Training.. " +
                f"Step {train_step}: " +
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
    print(f"✅ Training completed in {time.time() - start_time:.2f} seconds")
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)

    # Keep the main thread alive to allow background serving threads to run
    wl.keep_serving()
