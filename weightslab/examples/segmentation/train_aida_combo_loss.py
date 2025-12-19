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

# Import backported loss components
from model.losses.FocalLoss import FocalLossSeg
from model.losses.TverskyLoss import TverskyLoss

from weightslab.backend.ledgers import get_logger, get_model

# 1. SETUP LOGGING & PATHS
logging.basicConfig(level=logging.ERROR)
os.environ["GRPC_VERBOSITY"] = "debug"

# 2. MODEL (NANO ARCHITECTURE) - Same as train_aida.py
class PatchedSingleLiteNetPlus(SingleLiteNetPlus):
    def __init__(self, enc, caam, dec):
        super().__init__(enc, caam, dec)
        self.task_type = "segmentation"
        self.num_classes = 6
        self.seen_samples = 0
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
        pass

# 3. COMPOSITE LOSS FOR WEIGHTSLAB
class WeightsLabSingleLoss(nn.Module):
    """
    Combines Focal and Tversky loss while keeping 'Reduction: None' 
    so WeightsLab can track per-sample loss in the UI.
    """
    def __init__(self, num_classes, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # We set ignore_index=None internally to prevent FocalLossSeg 
        # from flattening the tensor into a 1D pixel list.
        # We will apply the mask manually to stay sample-wise.
        self.focal = FocalLossSeg(
            mode="multiclass", 
            alpha=0.25, 
            gamma=2.0, 
            ignore_index=None, 
            reduction="none"
        )
        self.tversky = TverskyLoss(
            mode="multiclass",
            alpha=0.7,
            beta=0.3,
            gamma=1.0,
            from_logits=True,
            ignore_index=ignore_index
        )

    def forward(self, outputs, labels):
        bs = labels.size(0)
        
        # --- 1. Focal Loss (Sample-wise) ---
        # With ignore_index=None, focal returns (N, H, W) in multiclass mode
        f_loss_raw = self.focal(outputs, labels) 
        
        # Manual masking for ignored pixels
        mask = (labels != self.ignore_index).float()
        f_loss_masked = f_loss_raw * mask
        
        # Mean per sample (BS,)
        # sum over spatial / count non-ignored pixels
        spatial_sum = f_loss_masked.view(bs, -1).sum(dim=1)
        pixel_count = mask.view(bs, -1).sum(dim=1).clamp(min=1)
        f_loss = spatial_sum / pixel_count
        
        # --- 2. Tversky Loss (Sample-wise) ---
        # We calculate Tversky score per sample and per class
        num_classes = outputs.size(1)
        y_pred = outputs.log_softmax(dim=1).exp() # Probabilities
        y_true = labels.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
        
        # One-hot encoding, zeroing out ignored pixels
        y_true_valid = torch.clamp(y_true, 0, num_classes - 1)
        y_true_oh = nn.functional.one_hot(y_true_valid, num_classes).permute(0, 2, 1).float()
        
        # Zero out the one-hot entries for ignored pixels
        pixel_mask = (y_true != self.ignore_index).unsqueeze(1).float() # (BS, 1, Pixels)
        y_true_oh = y_true_oh * pixel_mask
        
        # Compute Tversky per sample, per class (BS, C)
        dims = (2,) # Aggregate only spatial dimensions
        t_score = self.tversky.compute_score(y_pred, y_true_oh, smooth=self.tversky.smooth, eps=self.tversky.eps, dims=dims)
        t_loss_raw = 1.0 - t_score 
        
        # Mask out classes that don't exist in the sample ground truth
        class_mask = y_true_oh.sum(dims) > 0 # (BS, C)
        t_loss = (t_loss_raw * class_mask).sum(dim=1) / class_mask.sum(dim=1).clamp(min=1)
        
        # --- 3. Combine ---
        total_loss = f_loss + t_loss
        
        # UI SIGNALING: Record sub-components
        try:
            logger = get_logger()
            m = get_model()
            step = int(m.get_age())
            if logger and hasattr(logger, 'add_scalars'):
                logger.add_scalars("train_loss/focal", {"train_loss/focal": f_loss.mean().item()}, global_step=step)
                logger.add_scalars("train_loss/tversky", {"train_loss/tversky": t_loss.mean().item()}, global_step=step)
        except Exception:
            pass
        
        return total_loss

# 4. DATASET - Same as train_aida.py
class BDD100kSegDataset(Dataset):
    def __init__(self, root, split="train", num_classes=6, image_size=256):
        super().__init__()
        self.root = root
        self.split = split
        self.num_classes = num_classes
        img_dir = os.path.join(root, "images_1280x720", split)
        lbl_dir = os.path.join(root, "bdd100k_labels_dac_daa_lls_lld_curbs", split)
        image_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])
        self.images = [os.path.join(img_dir, f) for f in image_files]
        self.masks = [os.path.join(lbl_dir, f.replace(".jpg", ".png").replace(".jpeg", ".png")) for f in image_files]
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=Image.NEAREST)])

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        return self.transform(img), torch.from_numpy(np.array(self.mask_transform(mask), dtype=np.int64))

# 5. TRAINING & EVAL LOOPS - Same as train_aida.py
def train(loader, model, optimizer, criterion_mlt, device, step_idx=0):
    loss_val = 0.0
    with guard_training_context:
        try:
            (inputs, ids, labels) = next(loader)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # This triggers the WeightsLab record (Prediction Map + Loss per sample)
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
            print(f"Error in step {step_idx}: {e}")
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
            preds = outputs.argmax(dim=1).detach()
            loss_batch = criterion_mlt(outputs.float(), labels.long(), batch_ids=ids, model_age=model.get_age(), preds=preds)
            losses += loss_batch.mean().item()
            metric_mlt.update(preds, labels)
            pbar.update(1)
    return float(losses / max_batches), float(metric_mlt.compute().item() * 100.0)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "aida_config.yaml")
    with open(config_path, "r") as fh: parameters = yaml.safe_load(fh) or {}

    # Defaults
    parameters.setdefault("experiment_name", "single_litenet_combo_loss")
    parameters.setdefault("training_steps_to_do", 1000)
    parameters.setdefault("eval_full_to_train_steps_ratio", 100)
    parameters.setdefault("num_classes", 6)
    parameters.setdefault("ignore_index", 255)
    parameters.setdefault("image_size", 192)

    exp_name = parameters["experiment_name"]
    num_classes = int(parameters["num_classes"])
    ignore_index = int(parameters["ignore_index"])
    image_size = int(parameters["image_size"])
    max_steps = int(parameters["training_steps_to_do"])
    eval_every = int(parameters["eval_full_to_train_steps_ratio"])

    # 2. DEVICE SELECTION
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
    parameters.setdefault("data", {})
    parameters["data"].setdefault("train_loader", {"batch_size": 128, "num_workers": 0})
    parameters["data"].setdefault("test_loader", {"batch_size": 128, "num_workers": 0})
    batch_size = int(parameters["data"]["train_loader"]["batch_size"])
    num_workers = int(parameters["data"]["train_loader"].get("num_workers", 0))

    # 3. LOGGING DIRECTORY
    if not parameters.get("root_log_dir"):
        os.makedirs("/tmp/weightslab_logs", exist_ok=True)
        parameters["root_log_dir"] = tempfile.mkdtemp(prefix=f"wl_{exp_name}_", dir="/tmp/weightslab_logs")
    log_dir = parameters["root_log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # 4. REGISTRATION
    wl.watch_or_edit(Logger(), flag="logger", name=exp_name, log_dir=log_dir)
    wl.watch_or_edit(parameters, flag="hyperparameters", name="main", defaults=parameters)
    wl.watch_or_edit(parameters, flag="hyperparameters", name=None, defaults=parameters)

    # 5. DATA
    data_root = parameters.get("data_root")
    train_ds = BDD100kSegDataset(data_root, split="train", image_size=image_size)
    val_ds = BDD100kSegDataset(data_root, split="val", image_size=image_size)

    train_loader = wl.watch_or_edit(train_ds, flag="data", name="train_loader", batch_size=batch_size, num_workers=num_workers, is_training=True)
    test_loader = wl.watch_or_edit(val_ds, flag="data", name="test_loader", batch_size=batch_size, num_workers=num_workers)

    # 6. MODEL (Loaded from Config)
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
        in_channels=int(caam_p.get("in_channels", 16)),
        num_classes=num_classes,
        out_channels=int(caam_p.get("out_channels", 8))
    )
    dec_h = SimpleNamespace(
        in_channels=int(dec_p.get("in_channels", 8)),
        level_0_ch=int(dec_p.get("level_0_ch", 4)),
        level_1_ch=int(dec_p.get("level_1_ch", 8)),
        out_channels=num_classes
    )
    
    model_raw = PatchedSingleLiteNetPlus(enc_h, caam_h, dec_h).to(device)

    from weightslab.backend.ledgers import register_model
    register_model(exp_name, model_raw)
    guard_training_context.model = model_raw
    guard_testing_context.model = model_raw

    # Combo Loss Implementation
    optimizer = wl.watch_or_edit(optim.AdamW(model_raw.parameters(), lr=float(parameters["optimizer"]["lr"])), flag="optimizer", name=exp_name)
    
    # WRAPPING THE CUSTOM COMBO LOSS
    combo_loss_module = WeightsLabSingleLoss(num_classes, ignore_index).to(device)
    criterion = wl.watch_or_edit(combo_loss_module, flag="loss", name="train_loss/total", log=True)

    metric = wl.watch_or_edit(JaccardIndex(task="multiclass", num_classes=num_classes, ignore_index=ignore_index).to(device), flag="metric", name="val_metric/miou", log=True)

    wl.serve(root_directory=log_dir, serving_ui=False, serving_grpc=True)
    pause_controller.resume()

    print(f"Training {exp_name} with COMBO LOSS on {device}...")
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

    print(f"Done. Logs: {log_dir}")
