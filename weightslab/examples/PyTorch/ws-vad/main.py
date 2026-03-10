import os
import yaml
import time
import logging
import tempfile
import itertools
import random
import numpy as np
from tqdm import tqdm
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
    def __init__(self, in_ch: int = 3, base: int = 4, bottleneck: int = 512, image_size: int = 256):
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

        # NEW: Spatial path for reconstruction (preserves 2D structure)
        self.spatial_bottleneck = nn.Conv2d(C3 * 3, C3, kernel_size=1)
        
        # NEW: Global path for classification (catches peak anomaly signal)
        self.dropout = nn.Dropout(0.3)
        self.bottleneck = nn.Linear(C3 * 3, bottleneck)

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
        
        # 1. Global path (Classification / Embedding)
        # Use Global Adaptive Max Pool to find the single strongest anomaly signal anywhere in the image
        m_pooled = F.adaptive_max_pool2d(m, (1, 1)).view(m.size(0), -1)
        z = F.relu(self.bottleneck(m_pooled))
        z = self.dropout(z)
        
        cls_logits = self.cls_head(z)
        
        embed = self.embed_head(z)
        embed = F.normalize(embed, p=2, dim=1)
        
        # 2. Decoder path (Reconstruction)
        # Use the spatial bottleneck to keep the 2D layout intact
        d = F.relu(self.spatial_bottleneck(m))
        
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
        self.good_samples = []
        self.bad_samples = []
        
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
             print(f"Warning: split directory {split_dir} not found.")
             return
            
        for folder in sorted(os.listdir(split_dir)):  # sorted for determinism
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path): continue
            
            for fname in sorted(os.listdir(folder_path)):  # sorted for determinism
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(folder_path, fname)
                    if folder == 'good':
                        self.good_samples.append((full_path, 0))
                    else:
                        self.bad_samples.append((full_path, 1))
        
        # Combine for an overall reference if needed
        self.all_samples = self.good_samples + self.bad_samples

    def __len__(self):
        # We want to iterate through primarily the good samples to establish a baseline.
        # Length = number of good samples. Each good sample will be paired once per epoch.
        return len(self.good_samples)
        
    def __getitem__(self, idx):
        # Implement balanced pairing: 50% Pos-Pos, 50% Pos-Neg
        # (Assuming we have enough bad samples; if not, they will be recycled)
        n_good = len(self.good_samples)
        n_bad = len(self.bad_samples)
        
        if n_bad == 0:
            # Fallback if no bad samples: just pair consecutive good samples
            idx1 = idx % n_good
            idx2 = (idx + 1) % n_good
            img1_path, label1 = self.good_samples[idx1]
            img2_path, label2 = self.good_samples[idx2]
            uid1, uid2 = f"{self.split}_P{idx}_G{idx1}", f"{self.split}_P{idx}_G{idx2}"
        else:
            # Fully Balanced Pairing Strategy (50/50 Pairs, 50/50 Labels)
            # Cycle through 4 types of pairs based on idx % 4:
            # 0: Good + Good (Positive Contrastive, 100% Good Labels)
            # 1: Bad + Bad   (Positive Contrastive, 100% Bad Labels)
            # 2: Good + Bad  (Negative Contrastive, 50/50 Labels)
            # 3: Bad + Good  (Negative Contrastive, 50/50 Labels)
            
            p_type = idx % 4
            if p_type == 0:
                # GG
                i1, i2 = (idx // 4) % n_good, (idx // 4 + 1) % n_good
                img1_path, label1 = self.good_samples[i1]
                img2_path, label2 = self.good_samples[i2]
                uid1, uid2 = f"{self.split}_P{idx}_G{i1}", f"{self.split}_P{idx}_G{i2}"
            elif p_type == 1:
                # BB
                i1, i2 = (idx // 4) % n_bad, (idx // 4 + 1) % n_bad
                img1_path, label1 = self.bad_samples[i1]
                img2_path, label2 = self.bad_samples[i2]
                uid1, uid2 = f"{self.split}_P{idx}_B{i1}", f"{self.split}_P{idx}_B{i2}"
            elif p_type == 2:
                # GB
                i1, i2 = (idx // 4) % n_good, (idx // 4) % n_bad
                img1_path, label1 = self.good_samples[i1]
                img2_path, label2 = self.bad_samples[i2]
                uid1, uid2 = f"{self.split}_P{idx}_G{i1}", f"{self.split}_P{idx}_B{i2}"
            else:
                # BG
                i1, i2 = (idx // 4) % n_bad, (idx // 4) % n_good
                img1_path, label1 = self.bad_samples[i1]
                img2_path, label2 = self.good_samples[i2]
                uid1, uid2 = f"{self.split}_P{idx}_B{i1}", f"{self.split}_P{idx}_G{i2}"
            
            idx1, idx2 = i1, i2
            
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        
        if self.transform:
            img1_t = self.transform(img1)
            img2_t = self.transform(img2)
        else:
            img1_t = transforms.ToTensor()(img1)
            img2_t = transforms.ToTensor()(img2)
            
        group_id = f"{self.split}_pair_{uid1}_{uid2}"
        return (
            [img1_t, img2_t],     # The pair of inputs
            [idx1, idx2],         # Relative indices
            [label1, label2],     # Individual labels
            {                     # Metadata — "uids" causes 2 ledger rows per pair
                "group_id": group_id,
                "uids": [uid1, uid2],
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

def train_step(loader, model, optimizer, cls_criterion, contrastive_criterion, device, recon_weight, contrastive_weight):
    total_loss = None
    with guard_training_context:
        try:
            images, ids, labels, metadata = next(loader)
        except StopIteration:
            return None
            
        try:
            # Flatten the pair: [img1_batch, img2_batch] -> [2B, C, H, W]
            inputs_flat = torch.cat([img.float() for img in images], dim=0).to(device)
            labels_flat = torch.cat([torch.tensor(l).float() if not isinstance(l, torch.Tensor) else l.float() for l in labels], dim=0).view(-1, 1).to(device)
            
            # Flatten our global UIDs matching the blocked torch.cat order
            uids_flat = list(metadata["uids"][0]) + list(metadata["uids"][1])
            group_ids_list = list(metadata["group_id"])
            group_ids_flat = group_ids_list + group_ids_list
                
            optimizer.zero_grad()
            cls_logits, recon, embed = model(inputs_flat)
            
            # Classification loss
            cls_loss_batch = cls_criterion(cls_logits, labels_flat)
            cls_loss = cls_loss_batch.mean()
            

            # Contrastive loss (Siamese)
            half = len(embed) // 2
            embed1, embed2 = embed[:half], embed[half:]
            l1, l2 = labels_flat[:half], labels_flat[half:]
            # CosineEmbeddingLoss target: 1 for same class, -1 for different
            y = (l1 == l2).float() * 2 - 1
            y = y.squeeze().to(device)
            
            contrastive_loss_batch = contrastive_criterion(embed1, embed2, y)
            contrastive_loss = contrastive_loss_batch.mean()
            
            # Reconstruction loss (MSE)
            recon_loss_batch = F.mse_loss(recon, inputs_flat, reduction="none")
            # Flatten to (batch, -1) then mean over spatial/channel dims
            recon_loss_batch_per_sample = recon_loss_batch.view(recon_loss_batch.size(0), -1).mean(dim=1)
            recon_loss = recon_loss_batch_per_sample.mean()
            
            total_loss = cls_loss + recon_weight * recon_loss + contrastive_weight * contrastive_loss
            total_loss.backward()
            optimizer.step()
            
            # Compute per-sample accuracy
            cls_prob = torch.sigmoid(cls_logits).view(-1).detach()
            train_acc_batch = ((cls_prob > 0.5) == labels_flat.view(-1)).float()
            train_acc = train_acc_batch.mean().item()
            anomaly_score = cls_prob
            
            wl.save_signals(
                batch_ids=uids_flat,
                targets=labels_flat.view(-1).detach(),
                preds=cls_prob,
                preds_raw=cls_logits.view(-1).detach(),
                signals={
                    "train/cls_loss": cls_loss_batch.detach(),
                    "train/recon_loss": recon_loss_batch_per_sample.detach(),
                    "train/accuracy": train_acc_batch.detach(),
                    "pred_label": (cls_prob > 0.5).int(),
                    "anomaly_score": anomaly_score,
                }
            )
            
            # Group signal for contrastive loss (one entry per pair)
            wl.save_group_signals(
                signals={"train/contrastive_loss": contrastive_loss_batch.detach()},
                group_ids=group_ids_list,
                origin="train_loader",
            )
        except Exception as e:
            print(f"ERROR inside train_step: {repr(e)}")
            import traceback
            traceback.print_exc()
            raise e
        
    return total_loss.item() if total_loss is not None else None, train_acc if total_loss is not None else 0.0

def evaluate_all(loader, model, cls_criterion, contrastive_criterion, metric, device, recon_weight, contrastive_weight):
    model.eval()
    total_losses = 0
    num_batches = 0
    metric.reset()
    
    with guard_testing_context, torch.no_grad():
        for images, ids, labels, metadata in loader:
            inputs_flat = torch.cat([img.float() for img in images], dim=0).to(device)
            labels_flat = torch.cat([torch.tensor(l).float() if not isinstance(l, torch.Tensor) else l.float() for l in labels], dim=0).view(-1, 1).to(device)
            
            uids_flat = list(metadata["uids"][0]) + list(metadata["uids"][1])
            group_ids_list = list(metadata["group_id"])
            group_ids_flat = group_ids_list + group_ids_list
            
            cls_logits, recon, embed = model(inputs_flat)
            
            # Plain criterion calls — logging handled by wl.save_signals below
            cls_loss_batch = cls_criterion(cls_logits, labels_flat)

            
            half = len(embed) // 2
            embed1, embed2 = embed[:half], embed[half:]
            l1, l2 = labels_flat[:half], labels_flat[half:]
            y = (l1 == l2).float() * 2 - 1
            y = y.squeeze().to(device)
            
            contrastive_loss_batch = contrastive_criterion(embed1, embed2, y)
            
            # Reconstruction loss (MSE)
            recon_loss_batch = F.mse_loss(recon, inputs_flat, reduction="none")
            recon_loss_batch_per_sample = recon_loss_batch.view(recon_loss_batch.size(0), -1).mean(dim=1)
            
            batch_loss = cls_loss_batch.mean() + recon_weight * recon_loss_batch_per_sample.mean() + contrastive_weight * contrastive_loss_batch.mean()
            total_losses += batch_loss.item()
            num_batches += 1
            
            metric.update(cls_logits, labels_flat.long())
            
            wl.save_signals(
                batch_ids=uids_flat,
                targets=labels_flat.view(-1).detach(),
                preds=torch.sigmoid(cls_logits).view(-1).detach(),
                preds_raw=cls_logits.view(-1).detach(),
                signals={
                    "test/cls_loss": cls_loss_batch.detach(),
                    "test/recon_loss": recon_loss_batch_per_sample.detach(),
                    "pred_label": (torch.sigmoid(cls_logits).view(-1) > 0.5).int(),
                }
            )

            wl.save_group_signals(
                signals={"test/contrastive_loss": contrastive_loss_batch.detach()},
                group_ids=group_ids_list,
                origin="test_loader",
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
    parameters.setdefault("device", "mps")
    # Read training_steps instead of training_steps_to_do to align with config.yaml
    training_steps = parameters.get("training_steps", 1000)
    eval_every = parameters.get("eval_every", 200)
    parameters.setdefault("image_size", 256)
    parameters.setdefault("batch_size", 4)
    parameters.setdefault("lr", 1e-4)

    parameters.setdefault("contrastive_weight", 1.0)
    parameters.setdefault("root_log_dir", "./logs/vad/4")

    parameters.setdefault("is_training", True)

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

    contrastive_criterion = nn.CosineEmbeddingLoss(margin=1.0, reduction="none").to(device)

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
    
    pbar = tqdm(range(training_steps), desc="Training")
    for step in pbar:
        age = model.get_age() if hasattr(model, "get_age") else step
        
        loss_info = train_step(
            train_loader, model, optimizer, 
            cls_criterion, contrastive_criterion,
            device, parameters["recon_weight"], parameters["contrastive_weight"]
        )
        
        if loss_info is None:
            logger.info(f"Warning: train_step returned None at step {age}")
            continue
            
        loss, acc = loss_info
        pbar.set_postfix({"Loss": f"{loss:.4f}", "Acc": f"{acc:.4f}"})
            
        if age % eval_every == 0 and age > 0:
            logger.info(f"Evaluating at step {age}...")
            test_loss, test_acc = evaluate_all(
                test_loader, model, 
                cls_criterion, contrastive_criterion,
                test_metric, device, 
                parameters["recon_weight"], parameters["contrastive_weight"]
            )
            logger.info(f"Step {age} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    wl.keep_serving()
