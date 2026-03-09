import os
import yaml
import time
import logging
import tempfile
import itertools
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchmetrics.classification import BinaryAccuracy

import weightslab as wl
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context
)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# =============================================================================
# UNet Multi-Task Model (Classification + Reconstruction + Contrastive)
# =============================================================================
class UNetMulti(nn.Module):
    """
    Standard PyTorch implementation of the VAD UNet.
    WeightsLab will dynamically inject neuron operations during wl.watch_or_edit.
    """
    def __init__(self, in_ch: int = 3, base: int = 4, bottleneck: int = 32, image_size: int = 256):
        super().__init__()
        
        # Metadata for WeightsLab
        self.input_shape = (1, in_ch, image_size, image_size)
        
        C1, C2, C3 = base, base * 2, base * 4

        # ---- Encoder ----
        self.enc1_conv = nn.Conv2d(in_ch, C1, kernel_size=3, padding=1)
        self.enc1_bn   = nn.BatchNorm2d(C1)
        self.enc1_pool = nn.MaxPool2d(2)

        self.enc2_conv = nn.Conv2d(C1, C2, kernel_size=3, padding=1)
        self.enc2_bn   = nn.BatchNorm2d(C2)
        self.enc2_pool = nn.MaxPool2d(2)

        self.enc3_conv = nn.Conv2d(C2, C3, kernel_size=3, padding=1)
        self.enc3_bn   = nn.BatchNorm2d(C3)
        self.enc3_pool = nn.MaxPool2d(2)

        # ---- Mid / Bottleneck ----
        self.mid_conv3  = nn.Conv2d(C3, C3, kernel_size=3, padding=1)
        self.mid_conv5  = nn.Conv2d(C3, C3, kernel_size=5, padding=2)
        self.mid_conv7  = nn.Conv2d(C3, C3, kernel_size=7, padding=3)
        self.mid_bn     = nn.BatchNorm2d(C3 * 3)

        # Bottleneck reduces spatial info and combines channel info
        # 256 -> 128 -> 64 -> 32
        mid_size = image_size // (2**3)
        self.bottleneck = nn.Linear(C3 * 3 * mid_size * mid_size, bottleneck)
        self.bottleneck_up = nn.Linear(bottleneck, C3 * 3 * mid_size * mid_size)

        # ---- Decoder ----
        self.up1_conv = nn.Conv2d(C3, C2, kernel_size=3, padding=1)
        self.up1_bn   = nn.BatchNorm2d(C2)
        
        self.up2_conv = nn.Conv2d(C2, C1, kernel_size=3, padding=1)
        self.up2_bn   = nn.BatchNorm2d(C1)

        # ---- Heads ----
        self.cls_head    = nn.Linear(bottleneck, 1) # anomaly classification
        self.recon_head  = nn.Conv2d(C1, in_ch, kernel_size=1)  # reconstruction
        self.embed_head  = nn.Linear(bottleneck, 64) # contrastive embedding

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc1_bn(self.enc1_conv(x)))
        x = self.enc1_pool(x)
        
        x = F.relu(self.enc2_bn(self.enc2_conv(x)))
        x = self.enc2_pool(x)
        
        x = F.relu(self.enc3_bn(self.enc3_conv(x)))
        x = self.enc3_pool(x)
        
        # Middle
        m3 = self.mid_conv3(x)
        m5 = self.mid_conv5(x)
        m7 = self.mid_conv7(x)
        m = torch.cat([m3, m5, m7], dim=1)
        m = F.relu(self.mid_bn(m))
        
        # Bottleneck
        m_shape = m.shape
        b_in = m.view(m.size(0), -1)
        z = F.relu(self.bottleneck(b_in))
        
        # Classification head
        cls_logits = self.cls_head(z)
        
        # Embedding head for contrastive loss
        embed = self.embed_head(z)
        # Normalize for Cosine Loss
        embed = F.normalize(embed, p=2, dim=1)
        
        # Decoder
        up_z = F.relu(self.bottleneck_up(z))
        up_z = up_z.view(m_shape)
        
        # Reduce channels for decoder path
        d = (up_z[:, :up_z.size(1)//3] + 
             up_z[:, up_z.size(1)//3 : 2*up_z.size(1)//3] + 
             up_z[:, 2*up_z.size(1)//3 :]) / 3
             
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d = F.relu(self.up1_bn(self.up1_conv(d)))
        
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d = F.relu(self.up2_bn(self.up2_conv(d)))
        
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        recon = self.recon_head(d)
        
        return cls_logits, recon, embed

# =============================================================================
# Custom Dataset for VAD Pairs
# =============================================================================
class VADDataset(Dataset):
    """
    Returns a pair of images with their integer indices for WeightsLab tracking.
    """
    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []
        
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
             print(f"Warning: split directory {split_dir} not found.")
             return
            
        for folder in sorted(os.listdir(split_dir)):  # sorted for determinism
            label = 0 if folder == 'good' else 1
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path): continue
            
            for fname in sorted(os.listdir(folder_path)):  # sorted for determinism
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(folder_path, fname), label))
                        
    def __len__(self):
        # Length = number of pairs (each pair uses 2 consecutive samples)
        return len(self.samples) // 2
        
    def __getitem__(self, idx):
        # Pick a pair deterministically
        idx1 = idx * 2
        idx2 = idx * 2 + 1
        
        if idx2 >= len(self.samples):
            idx2 = 0
            
        img1_path, label1 = self.samples[idx1]
        img2_path, label2 = self.samples[idx2]
        
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        if self.transform:
            img1_t = self.transform(img1)
            img2_t = self.transform(img2)
        else:
            img1_t = transforms.ToTensor()(img1)
            img2_t = transforms.ToTensor()(img2)
            
        # "uids" as a Python list tells DataSampleTrackingWrapper.__init__ to expand
        # this physical item into TWO ledger rows (one per image in the pair).
        # Must be a plain list — numpy unicode arrays (dtype <U4) are rejected by
        # PyTorch's default collate_fn and crash the DataLoader.
        group_id = f"{self.split}_pair_{idx1}_{idx2}"
        return (
            [img1_t, img2_t],     # The pair of inputs
            [idx1, idx2],         # Integer indices for WeightsLab dataframe tracking
            [label1, label2],     # Individual labels
            {                     # Metadata — "uids" causes 2 ledger rows per pair
                "group_id": group_id,
                "uids": [str(idx1), str(idx2)],
            }
        )

# =============================================================================
# Training / Evaluation Logic
# =============================================================================
def flatten_lists(arr):
    out = []
    for item in arr:
        if isinstance(item, (list, tuple)):
            out.extend(flatten_lists(item))
        else:
            out.append(item)
    return out

def train_step(loader, model, optimizer, cls_criterion, recon_criterion, contrastive_criterion, device, recon_weight, contrastive_weight):
    with guard_training_context:
        try:
            images, ids, labels, metadata = next(loader)
        except StopIteration:
            return None
            
        # Flatten the pair: [img1_batch, img2_batch] -> [2B, C, H, W]
        inputs_flat = torch.cat([img.float() for img in images], dim=0).to(device)
        labels_flat = torch.cat([torch.tensor(l).float() if not isinstance(l, torch.Tensor) else l.float() for l in labels], dim=0).view(-1, 1).to(device)
        
        # ids is [[idx1_a, idx1_b,...], [idx2_a, idx2_b,...]] — flatten to 2B integer list
        ids_flat = []
        for pair_ids in zip(*ids):
            ids_flat.extend([int(x) for x in pair_ids])
        ids_tensor = torch.tensor(ids_flat)
            
        optimizer.zero_grad()
        cls_logits, recon, embed = model(inputs_flat)
        
        # Classification loss
        cls_loss_batch = cls_criterion(cls_logits, labels_flat)
        cls_loss = cls_loss_batch.mean()
        
        # Reconstruction loss — per-sample mean over spatial dims then log
        recon_loss_batch = recon_criterion(recon, inputs_flat)
        recon_loss_batch = recon_loss_batch.mean(dim=(1, 2, 3))  # [2B]
        recon_loss = recon_loss_batch.mean()
        
        # Contrastive loss (Siamese)
        half = len(embed) // 2
        embed1, embed2 = embed[:half], embed[half:]
        l1, l2 = labels_flat[:half], labels_flat[half:]
        # CosineEmbeddingLoss target: 1 for same class, -1 for different
        y = (l1 == l2).float() * 2 - 1
        y = y.squeeze().to(device)
        
        contrastive_loss_batch = contrastive_criterion(embed1, embed2, y)
        contrastive_loss = contrastive_loss_batch.mean()
        
        total_loss = cls_loss + recon_weight * recon_loss + contrastive_weight * contrastive_loss
        total_loss.backward()
        optimizer.step()
        
        # Compute custom per-sample signals
        recon_error = torch.mean((recon - inputs_flat)**2, dim=(1,2,3)).detach()
        cls_prob = torch.sigmoid(cls_logits).view(-1).detach()
        max_err = recon_error.max().item() + 1e-6
        anomaly_score = 0.5 * cls_prob + 0.5 * (recon_error / max_err)
        
        # Log per-sample signals with integer batch_ids for dataframe tracking
        wl.save_signals(
            batch_ids=ids_tensor,
            signals={
                "train/cls_loss": cls_loss_batch.detach(),
                "train/recon_loss": recon_loss_batch.detach(),
                "recon_error": recon_error,
                "anomaly_score": anomaly_score,
                "labels": labels_flat.view(-1).detach(),
            }
        )
        
        # Group signal for contrastive loss (one entry per pair)
        group_ids = [str(g) for g in metadata["group_id"]]
        wl.save_group_signals(
            signals={"train/contrastive_loss": contrastive_loss_batch.detach()},
            group_ids=group_ids,
            origin="train_loader",
        )
        
    return total_loss.item()

def evaluate_all(loader, model, cls_criterion, recon_criterion, contrastive_criterion, metric, device, recon_weight, contrastive_weight):
    model.eval()
    total_losses = 0
    num_batches = 0
    metric.reset()
    
    with guard_testing_context, torch.no_grad():
        for images, ids, labels, metadata in loader:
            inputs_flat = torch.cat([img.float() for img in images], dim=0).to(device)
            labels_flat = torch.cat([torch.tensor(l).float() if not isinstance(l, torch.Tensor) else l.float() for l in labels], dim=0).view(-1, 1).to(device)
            
            ids_flat = []
            for pair_ids in zip(*ids):
                ids_flat.extend([int(x) for x in pair_ids])
            ids_tensor = torch.tensor(ids_flat)
            
            cls_logits, recon, embed = model(inputs_flat)
            
            # Plain criterion calls — logging handled by wl.save_signals below
            cls_loss_batch = cls_criterion(cls_logits, labels_flat)
            recon_loss_batch = recon_criterion(recon, inputs_flat)
            recon_loss_batch = recon_loss_batch.mean(dim=(1, 2, 3))  # [2B]
            
            half = len(embed) // 2
            embed1, embed2 = embed[:half], embed[half:]
            l1, l2 = labels_flat[:half], labels_flat[half:]
            y = (l1 == l2).float() * 2 - 1
            y = y.squeeze().to(device)
            
            contrastive_loss_batch = contrastive_criterion(embed1, embed2, y)
            
            batch_loss = cls_loss_batch.mean() + recon_weight * recon_loss_batch.mean() + contrastive_weight * contrastive_loss_batch.mean()
            total_losses += batch_loss.item()
            num_batches += 1
            
            metric.update(cls_logits, labels_flat.long())
            
            # Log test signals
            recon_error = torch.mean((recon - inputs_flat)**2, dim=(1,2,3)).detach()
            wl.save_signals(
                batch_ids=ids_tensor,
                signals={
                    "test/cls_loss": cls_loss_batch.detach(),
                    "test/recon_error": recon_error,
                }
            )
            
    final_loss = total_losses / max(1, num_batches)
    return final_loss, metric.compute().item()

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Removed wl.clear_all() to help with resuming/checkpointing persistence
    
    # 1. Hyperparameters & Config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parameters = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            parameters = yaml.safe_load(f) or {}

    parameters.setdefault("experiment_name", "vad_unet_contrastive")
    parameters.setdefault("device", "auto")
    parameters.setdefault("training_steps_to_do", 1000)
    parameters.setdefault("eval_every", 50)
    parameters.setdefault("image_size", 256)
    parameters.setdefault("batch_size", 4)
    parameters.setdefault("lr", 1e-4)
    parameters.setdefault("recon_weight", 10.0)
    parameters.setdefault("contrastive_weight", 1.0)
    parameters.setdefault("root_log_dir", "./logs/vad/2")

    parameters.setdefault("is_training", False)

    if parameters["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(parameters["device"])

    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    # WeightsLab initialization
    wl.watch_or_edit(parameters, flag="hyperparameters", defaults=parameters)

    # 2. Data
    data_root = parameters.get("data_root")
    if not data_root or not os.path.exists(data_root):
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "mvtec_capsule"))
    
    if not os.path.exists(data_root):
        print(f"Warning: Data root {data_root} not found. Creating dummy data structure.")
        os.makedirs(os.path.join(data_root, "train", "good"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "train", "bad"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "test", "good"), exist_ok=True)
        os.makedirs(os.path.join(data_root, "test", "bad"), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((parameters["image_size"], parameters["image_size"])),
        transforms.ToTensor(),
    ])

    _train_ds = VADDataset(data_root, split="train", transform=transform)
    _test_ds  = VADDataset(data_root, split="test", transform=transform)

    train_loader = wl.watch_or_edit(
        _train_ds, flag="data", loader_name="train_loader",
        batch_size=parameters["batch_size"], shuffle=True
    )
    test_loader = wl.watch_or_edit(
        _test_ds, flag="data", loader_name="test_loader",
        batch_size=parameters["batch_size"], shuffle=False
    )

    # 3. Model, Optimizer, etc.
    print("Initialising UNet model...", flush=True)
    _model = UNetMulti(in_ch=3, base=4, bottleneck=32, image_size=parameters["image_size"]).to(device)
    print(f"Watching model with WeightsLab...", flush=True)
    model = wl.watch_or_edit(_model, flag="model", device=device, compute_dependencies=False)

    _optimizer = optim.Adam(model.parameters(), lr=parameters["lr"])
    optimizer = wl.watch_or_edit(_optimizer, flag="optimizer")

    # Plain PyTorch criteria — all logging done via wl.save_signals with batch_ids
    cls_criterion = nn.BCEWithLogitsLoss(reduction="none")
    recon_criterion = nn.MSELoss(reduction="none")
    contrastive_criterion = nn.CosineEmbeddingLoss(margin=0.5, reduction="none").to(device)

    # Only wrap the metric (it's a proper LoggerQueue-compatible object)
    test_metric = wl.watch_or_edit(
        BinaryAccuracy().to(device),
        flag="metric", signal_name="test/accuracy", log=True
    )

    # 4. Serving
    logger.info("Starting WeightsLab servers...")
    wl.serve(serving_grpc=True, serving_cli=True)

    # Give backend a moment to stabilize
    time.sleep(2)

    # 5. Training Loop
    logger.info(f"Starting Contrastive-VAD Training on {device}...")
    train_iter = itertools.cycle(train_loader)
    
    for step in range(parameters["training_steps_to_do"]):
        age = model.get_age() if hasattr(model, "get_age") else step
        
        logger.info(f"Step {age}...")
        loss = train_step(
            train_iter, model, optimizer, 
            cls_criterion, recon_criterion, contrastive_criterion,
            device, parameters["recon_weight"], parameters["contrastive_weight"]
        )
        
        if loss is None:
            logger.info(f"Warning: train_step returned None at step {age}")
            continue
            
        if age % parameters["eval_every"] == 0:
            logger.info(f"Evaluating at step {age}...")
            test_loss, test_acc = evaluate_all(
                test_loader, model, 
                cls_criterion, recon_criterion, contrastive_criterion,
                test_metric, device, parameters["recon_weight"], parameters["contrastive_weight"]
            )
            logger.info(f"Step {age} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    wl.keep_serving()
