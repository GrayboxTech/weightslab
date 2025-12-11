import os
import time
import tempfile
import logging

import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np

import weightslab as wl
from torchvision import transforms
from torchmetrics import JaccardIndex
from torch.utils.data import Dataset
from PIL import Image

from weightslab.utils.board import Dash as Logger
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
    pause_controller,
)

logging.basicConfig(level=logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "debug"


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
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.task_type = "segmentation"

        img_dir = os.path.join(root, "images_1280x720", split)
        lbl_dir = os.path.join(root, "labels", split)

        image_files = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        image_files = sorted(set(image_files))

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
                transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
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
        mask = Image.open(mask_path)

        img_t = self.image_transform(img)

        mask_r = self.mask_resize(mask)
        mask_np = np.array(mask_r, dtype=np.int64)
        mask_t = torch.from_numpy(mask_np)  # [H, W] int64

        return img_t, mask_t


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
            model_age=model.get_age(),
            batch_ids=ids,
            preds=preds,
        )  # CrossEntropyLoss(reduction="none") → [B,H,W]
        loss = loss_batch.mean()

        loss.backward()
        optimizer.step()

    return float(loss.detach().cpu().item())


def test(loader, model, criterion_mlt, metric_mlt, device):
    """Full evaluation pass over the val loader."""
    losses = 0.0
    metric_mlt.reset()

    with guard_testing_context, torch.no_grad():
        for (inputs, ids, labels) in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)  # [B,H,W]

            loss_batch = criterion_mlt(
                outputs.float(),
                labels.long(),
                model_age=model.get_age(),
                batch_ids=ids,
                preds=preds,
            )
            losses += torch.mean(loss_batch)

            metric_mlt.update(preds, labels)

    loss = float((losses / len(loader)).detach().cpu().item())
    miou = float(metric_mlt.compute().detach().cpu().item() * 100.0)
    return loss, miou


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "bdd_seg_training_config.yaml")
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
    parameters["is_training"] = True
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
        parameters["root_log_dir"] = os.path.join(tmp_dir, "logs")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    log_dir = parameters["root_log_dir"]
    max_steps = parameters["training_steps_to_do"]
    eval_every = parameters["eval_full_to_train_steps_ratio"]

    # --- 4) Register logger + hyperparameters ---
    logger = Logger()
    wl.watch_or_edit(logger, flag="logger", name=exp_name, log_dir=log_dir)

    wl.watch_or_edit(
        parameters,
        flag="hyperparameters",
        name=exp_name,
        defaults=parameters,
        poll_interval=1.0,
    )

    # --- 5) Data (BDD100k reduced) ---
    # Your layout from earlier:
    #   .../development/merge-main-dev/weightslab  (this script)
    #   .../development/data/BDD100k_reduced      (data)
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
    )
    _val_dataset = BDD100kSegDataset(
        root=data_root,
        split="val",
        num_classes=num_classes,
        ignore_index=ignore_index,
        image_size=image_size,
    )

    train_loader = wl.watch_or_edit(
        _train_dataset,
        flag="data",
        name="train_loader",
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=train_cfg.get("train_shuffle", True),
        is_training=True,
    )
    test_loader = wl.watch_or_edit(
        _val_dataset,
        flag="data",
        name="test_loader",
        batch_size=test_cfg.get("batch_size", 2),
        shuffle=test_cfg.get("test_shuffle", False),
    )

    # --- 6) Model, optimizer, losses, metric ---
    _model = SmallUNet(
        in_channels=3, num_classes=num_classes, image_size=image_size
    )
    model = wl.watch_or_edit(_model, flag="model", name=exp_name, device=device)

    lr = parameters.get("optimizer", {}).get("lr", 1e-3)
    _optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer", name=exp_name)

    train_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index),
        flag="loss",
        name="train_loss/mlt_loss",
        log=True,
    )
    test_criterion_mlt = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index),
        flag="loss",
        name="test_loss/mlt_loss",
        log=True,
    )
    test_metric_mlt = wl.watch_or_edit(
        JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
        ).to(device),
        flag="metric",
        name="test_metric/miou",
        log=True,
    )

    # --- 7) Start WeightsLab services ---
    wl.serve(
        serving_ui=True,
        root_directory=log_dir,
        serving_grpc=True,
        n_workers_grpc=parameters.get("number_of_workers"),
        serving_cli=False,
    )

    print("=" * 60)
    print("🚀 STARTING BDD100k SEGMENTATION TRAINING")
    print(f"📈 Total steps: {max_steps}")
    print(f"🔄 Evaluation every {eval_every} steps")
    print(f"💾 Logs will be saved to: {log_dir}")
    print(f"📂 Data root: {data_root}")
    print("=" * 60 + "\n")

    pause_controller.resume()

    for train_step in tqdm.trange(max_steps, dynamic_ncols=True):
        train_loss = train(train_loader, model, optimizer, train_criterion_mlt, device)

        test_loss, test_metric = None, None
        if train_step % eval_every == 0:
            test_loss, test_metric = test(
                test_loader,
                model,
                test_criterion_mlt,
                test_metric_mlt,
                device,
            )

        if train_step % 10 == 0 or test_loss is not None:
            status = f"[Step {train_step:5d}/{max_steps}] Train Loss: {train_loss:.4f}"
            if test_loss is not None:
                status += (
                    f" | Val Loss: {test_loss:.4f} | mIoU: {test_metric:.2f}%"
                )
            print(status)

    print("\n" + "=" * 60)
    print(f"💾 Logs saved to: {log_dir}")
    print("=" * 60)
