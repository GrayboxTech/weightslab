import os
import sys
import time
import tempfile
import logging
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from types import SimpleNamespace
from PIL import Image

import weightslab as wl
from torchvision import transforms
from torchmetrics import JaccardIndex
from torch.utils.data import Dataset, DataLoader

from weightslab.utils.board import Dash as Logger
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
    pause_controller,
)

from model import SingleLiteNetPlus
from model.caam.CAAM import CAAM
from model.caam.GCN import GCN
from model.common.ConvBatchnormRelu import ConvBatchnormRelu

# 1. SETUP LOGGING & PATHS
logging.basicConfig(level=logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "debug"

# 2. MODEL (NANO ARCHITECTURE)
class PatchedSingleLiteNetPlus(SingleLiteNetPlus):
    """
    Stable version of SingleLiteNetPlus 'Nano' for Weights Studio.
    Fixes the internal CAAM/GCN dimension mismatch and adds UI metadata.
    """
    def __init__(self, enc, caam, dec):
        super().__init__(enc, caam, dec)
        
        # UI Metadata
        self.task_type = "segmentation"
        self.num_classes = 6
        self.seen_samples = 0
        
        # Patch CAAM to work with 6 classes (2x3 grid)
        self.caam = CAAM(
            feat_in=caam.in_channels,
            num_classes=6, 
            bin_size=(2, 3), 
            norm_layer=nn.BatchNorm2d
        )
        self.caam.gcn = GCN(num_node=6, num_channel=caam.in_channels)
        self.caam.fuse = nn.Conv2d(6, 1, kernel_size=1)
        self.conv_caam = ConvBatchnormRelu(caam.in_channels, caam.out_channels)

    def get_age(self):
        return self.seen_samples

    def set_tracking_mode(self, mode):
        pass # Handle data tracking via Loss/Loader wrappers

# 3. DATASET
class BDD100kSegDataset(Dataset):
    def __init__(self, root, split="train", num_classes=6, image_size=256):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        self.task_type = "segmentation"

        img_dir = os.path.join(root, "images_1280x720", split)
        lbl_dir = os.path.join(root, "bdd100k_labels_dac_daa_lls_lld_curbs", split)

        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
        self.images = [os.path.join(img_dir, f) for f in image_files]
        self.masks = [os.path.join(lbl_dir, f.replace(".jpg", ".png").replace(".jpeg", ".png")) for f in image_files]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        return self.transform(img), torch.from_numpy(np.array(self.mask_transform(mask), dtype=np.int64))

# 4. TRAINING & EVAL LOOPS
def train(loader, model, optimizer, criterion_mlt, device, step_idx=0):
    loss_val = 0.0
    with guard_training_context:
        try:
            if step_idx == 0:
                print("Starting first training step...", flush=True)
            
            # WeightsLab Dataloader gives: (inputs, ids, labels)
            (inputs, ids, labels) = next(loader)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Trigger WeightsLab recording (Prediction Map + Loss per sample)
            loss_batch = criterion_mlt(
                outputs.float(), 
                labels.long(), 
                batch_ids=ids, 
                model_age=model.get_age(), 
                preds=outputs.argmax(dim=1).detach()
            )
            
            loss = loss_batch.mean()
            loss.backward()
            optimizer.step()
            
            model.seen_samples += inputs.shape[0]
            loss_val = float(loss.item())
        except Exception as e:
            print(f"\nError in training step {step_idx}: {e}")
            import traceback
            traceback.print_exc()
            
    return loss_val

def test(loader, model, criterion_mlt, metric_mlt, device, max_batches=20):
    losses = 0.0
    metric_mlt.reset()
    pbar = tqdm.tqdm(loader, desc="Evaluating", total=max_batches, leave=False)
    
    with guard_testing_context, torch.no_grad():
        for i, (inputs, ids, labels) in enumerate(pbar):
            if i >= max_batches: break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Record eval loss
            preds = outputs.argmax(dim=1).detach()
            loss_batch = criterion_mlt(
                outputs.float(), 
                labels.long(), 
                batch_ids=ids, 
                model_age=model.get_age(),
                preds=preds
            )
            losses += loss_batch.mean().item()
            
            metric_mlt.update(preds, labels)
            pbar.update(1)

    return float(losses / max_batches), float(metric_mlt.compute().item() * 100.0)


if __name__ == "__main__":
    # --- 1) Load hyperparameters from YAML (if present) ---
    config_path = os.path.join(os.path.dirname(__file__), "aida_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as fh:
            parameters = yaml.safe_load(fh) or {}
    else:
        parameters = {}

    # Defaults
    parameters.setdefault("experiment_name", "single_litenet_aida")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 100)
    parameters["is_training"] = lambda: not pause_controller.is_paused()

    parameters.setdefault("num_classes", 6)
    parameters.setdefault("ignore_index", 255)
    parameters.setdefault("image_size", 192)

    exp_name = parameters["experiment_name"]
    num_classes = int(parameters["num_classes"])
    ignore_index = int(parameters["ignore_index"])
    image_size = int(parameters["image_size"])
    max_steps = int(parameters["training_steps_to_do"])
    eval_every = int(parameters["eval_full_to_train_steps_ratio"])

    # --- 2) Device selection ---
    if parameters.get("device", "auto") == "auto":
        if torch.backends.mps.is_available():
            parameters["device"] = torch.device("mps")
        elif torch.cuda.is_available():
            parameters["device"] = torch.device("cuda")
        else:
            parameters["device"] = torch.device("cpu")
    device = parameters["device"]
    print(f"Using device: {device}")

    # Set batch defaults
    parameters["data"].setdefault("train_loader", {"batch_size": 128, "num_workers": 0})
    parameters["data"].setdefault("test_loader", {"batch_size": 128, "num_workers": 0})
    batch_size = parameters["data"]["train_loader"]["batch_size"]
    num_workers = parameters["data"]["train_loader"].get("num_workers", 0)

    # --- 3) Logging directory ---
    if not parameters.get("root_log_dir"):
        tmp_dir = tempfile.mkdtemp()
        parameters["root_log_dir"] = os.path.join(tmp_dir, f"weightslab_{exp_name}")
    os.makedirs(parameters["root_log_dir"], exist_ok=True)
    log_dir = parameters["root_log_dir"]

    # --- 4) Register logger + hyperparameters ---
    wl.watch_or_edit(Logger(), flag="logger", name=exp_name, log_dir=log_dir)
    wl.watch_or_edit(parameters, flag="hyperparameters", name="main", defaults=parameters)
    wl.watch_or_edit(parameters, flag="hyperparameters", name=None, defaults=parameters)

    # --- 5) Data ---
    data_root = parameters.get("data_root")
    if not data_root:
        print("Error: 'data_root' must be specified in aida_seg_training_config.yaml")
        sys.exit(1)
    
    train_ds = BDD100kSegDataset(data_root, split="train", image_size=image_size)
    val_ds = BDD100kSegDataset(data_root, split="val", image_size=image_size)

    shuffle = parameters["data"]["train_loader"].get("shuffle", False)
    train_loader = wl.watch_or_edit(
        train_ds, flag="data", name="train_loader", 
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, is_training=True
    )
    test_loader = wl.watch_or_edit(
        val_ds, flag="data", name="test_loader", 
        batch_size=batch_size, num_workers=num_workers
    )

    # --- 6) Model (Loaded from Config) ---
    m_cfg = parameters.get("model", {})
    enc_p = m_cfg.get("encoder", {})
    caam_p = m_cfg.get("caam", {})
    dec_p = m_cfg.get("decoder", {})

    enc_h = SimpleNamespace(
        in_channels=3, 
        level_0_ch=int(enc_p.get("level_0_ch", 4)),
        level_1_ch=int(enc_p.get("level_1_ch", 8)),
        level_2_ch=int(enc_p.get("level_2_ch", 16)),
        level_3_ch=int(enc_p.get("level_3_ch", 32)),
        level_4_ch=int(enc_p.get("level_4_ch", 64)),
        out_channels=int(enc_p.get("out_channels", 16)),
        p=int(enc_p.get("p", 1)),
        q=int(enc_p.get("q", 1))
    )
    caam_h = SimpleNamespace(
        in_channels=enc_h.out_channels, 
        num_classes=num_classes, 
        out_channels=int(caam_p.get("out_channels", 8))
    )
    dec_h = SimpleNamespace(
        in_channels=caam_h.out_channels, 
        level_0_ch=dec_p.get("level_0_ch", 4),
        level_1_ch=dec_p.get("level_1_ch", 8),
        out_channels=num_classes
    )

    model_raw = PatchedSingleLiteNetPlus(enc_h, caam_h, dec_h).to(device)
    
    # External Registration
    from weightslab.backend.ledgers import register_model
    register_model(exp_name, model_raw)
    register_model(None, model_raw)
    guard_training_context.model = model_raw
    guard_testing_context.model = model_raw

    # --- 7) Loss & Optimizer ---
    lr = float(parameters.get("optimizer", {}).get("lr", 2e-3))
    optimizer = wl.watch_or_edit(optim.AdamW(model_raw.parameters(), lr=lr), flag="optimizer", name=exp_name)
    
    # Compute class weights
    def compute_class_weights(dataset, num_classes, ignore_index=255, max_samples=500):
        print("\n" + "=" * 60)
        print(f"Computing class weights for {num_classes} classes (max {max_samples} samples)...")
        class_counts = np.zeros(num_classes, dtype=np.float64)
        num_samples = min(len(dataset), max_samples)
        
        for idx in tqdm.tqdm(range(num_samples), desc="📊 Analyzing Distribution"):
            _, label = dataset[idx]
            label_np = label.numpy() if hasattr(label, 'numpy') else np.array(label)
            for c in range(num_classes):
                class_counts[c] += (label_np == c).sum()
        
        class_counts = np.maximum(class_counts, 1) # Avoid div by zero
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts)
        class_weights = class_weights / class_weights.mean() # Normalize
        
        print("\nClass distribution and weights:")
        for c in range(num_classes):
            pct = (class_counts[c] / total_pixels) * 100
            print(f"Class {c}: {pct:6.2f}% -> weight: {class_weights[c]:.3f}")
        print("=" * 60 + "\n")
        return torch.FloatTensor(class_weights).to(device)

    weights = compute_class_weights(train_ds, num_classes)

    criterion = wl.watch_or_edit(
        nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index, weight=weights), 
        flag="loss", name="train_loss/loss", log=True
    )
    metric = wl.watch_or_edit(
        JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device), 
        flag="metric", name="val_metric/miou", log=True
    )

    # --- 8) Run ---
    wl.serve(root_directory=log_dir, serving_ui=False, serving_grpc=True)
    pause_controller.resume()

    print(f"Training {exp_name} on {device}...")
    loader_iter = iter(train_loader)
    pbar = tqdm.trange(max_steps)
    for step in pbar:
        try:
            train_loss = train(loader_iter, model_raw, optimizer, criterion, device, step_idx=step)
            pbar.set_postfix(loss=f"{train_loss:.4f}")
        except StopIteration:
            loader_iter = iter(train_loader)
            train_loss = train(loader_iter, model_raw, optimizer, criterion, device, step_idx=step)

        if step > 0 and step % eval_every == 0:
            val_loss, val_miou = test(test_loader, model_raw, criterion, metric, device)
            print(f"Step {step} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | mIoU: {val_miou:.2f}%")

    print(f"Training complete. Logs: {log_dir}")

