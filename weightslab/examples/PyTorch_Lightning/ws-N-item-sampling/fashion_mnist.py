import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import weightslab as wl
from weightslab.components.global_monitoring import (
    guard_training_context,
    guard_testing_context,
    pause_controller
)
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch.optim as optim



import os

from torchmetrics.classification import Accuracy

# -----------------------------------------------------------------------------
# 1. Dataset: Flattened Fashion MNIST with Deterministic Pairing
# -----------------------------------------------------------------------------
class FashionMNISTSiamese(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        self.split = "train" if train else "val"
        self.base_ds = datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)
        
        # Deterministic shuffle for stable pairing
        self.indices = list(range(len(self.base_ds)))
        import random
        random.seed(42)
        random.shuffle(self.indices)

    def __len__(self):
        # We return PAIRS of images, so dataset length is halved
        return len(self.base_ds) // 2

    def __getitem__(self, idx):
        # Pick two images based on shifted indices
        idx1 = self.indices[idx * 2]
        idx2 = self.indices[idx * 2 + 1]
        
        img1, t1 = self.base_ds[idx1]
        img2, t2 = self.base_ds[idx2]
        
        # Unique IDs for each sample
        uid1 = f"{self.split}_sample_{idx1}_left"
        uid2 = f"{self.split}_sample_{idx2}_right"
        # Shared Group ID (int)
        offset = 1000000 if self.split == "val" else 0
        group_id = idx + offset
        
        # IMPORTANT: We return a tuple where:
        # 1. First element is a LIST of images (will be collated into [batch_l, batch_r])
        # 2. Second element is None (WeightsLab will manage it based on compute_hash=False)
        # 3. Third element is a LIST of targets (labels)
        # 4. Metadata dict contains 'group_id' for WeightsLab expansion/retrieval
        return (
            [img1, img2], 
            [idx1, idx2], 
            [t1, t2], 
            {
                "group_id": group_id,
                "pair_type": "same" if t1 == t2 else "different"
            }
        )

# -----------------------------------------------------------------------------
# 2. Multi-Task Model
# -----------------------------------------------------------------------------
class FashionHingeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = (1, 28, 28)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU()
        )
        self.cls_head = nn.Linear(128, 10)
        self.embed_head = nn.Linear(128, 64)

    def forward(self, x):
        features = self.backbone(x)
        return self.cls_head(features), self.embed_head(features)

    def step(self, batch, loss_clsf, loss_cosine, acc_metric, mode="train"):
        images, uids, targets, metadata = batch
        
        if isinstance(images, (list, tuple)):
            x_flat = torch.cat([img.float() for img in images], dim=0) 
        else:
            x_flat = images.float().view(-1, *images.shape[2:])
            
        if isinstance(targets, (list, tuple)):
            t_flat = torch.cat(targets, dim=0) 
        else:
            t_flat = targets.view(-1) 
        
        uids_flat = []
        for i in range(len(uids)):
            part = uids[i].detach().cpu() if hasattr(uids[i], "detach") else uids[i]
            # Handle list/tensor and convert to string
            if hasattr(part, "tolist"):
                uids_flat.extend([str(x) for x in part.tolist()])
            elif isinstance(part, (list, tuple)):
                uids_flat.extend([str(x) for x in part])
            else:
                uids_flat.append(str(part))
        
        logits, embed = self(x_flat)
        preds = logits.argmax(dim=1)
        
        err_cls = loss_clsf(logits, t_flat, batch_ids=uids_flat, preds=preds)
        
        e1 = embed[:len(embed)//2]
        e2 = embed[len(embed)//2:]
        t1 = t_flat[:len(t_flat)//2]
        t2 = t_flat[len(t_flat)//2:]
        
        y = (t1 == t2).float() * 2 - 1 
        loss_embed = loss_cosine(e1, e2, y) # No batch_ids here, return vector
        
        # Log per-group losses correctly (will be broadcast to both members of the pair)
        wl.save_group_signals(
            signals={"loss_embed_cosine": loss_embed},
            group_ids=metadata["group_id"],
            origin=mode
        )
        
        total_loss = err_cls.mean() + loss_embed.mean()
        
        if mode != "train":
            acc_metric.update(logits, t_flat)
            
        return total_loss

class FashionHingeModel(LightningModule):
    def __init__(self, lr=0.001, backbone=None, loss_clsf=None, loss_cosine=None, acc_metric=None):
        super().__init__()
        self.save_hyperparameters(ignore=['backbone', 'loss_clsf', 'loss_cosine', 'acc_metric'])
        self.lr = lr
        self.model = backbone
        self.loss_clsf = loss_clsf
        self.loss_cosine = loss_cosine
        self.acc_metric = acc_metric

    def training_step(self, batch, batch_idx):
        if hasattr(self.model, "current_step"):
            self.model.current_step = self.global_step
            
        with guard_training_context:
            return self.model.step(batch, self.loss_clsf, self.loss_cosine, self.acc_metric, mode="train")

    def validation_step(self, batch, batch_idx):
        if hasattr(self.model, "current_step"):
            self.model.current_step = self.global_step
            
        with guard_testing_context:
            return self.model.step(batch, self.loss_clsf, self.loss_cosine, self.acc_metric, mode="val")

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# -----------------------------------------------------------------------------
# 3. Training/Evaluation Callback
# -----------------------------------------------------------------------------
class TrainEvalCallback(Callback):
    def __init__(self, train_loader, every_n_steps=50):
        super().__init__()
        self.train_loader = train_loader
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.every_n_steps == 0:
            pl_module.eval()
            with torch.no_grad():
                it = iter(self.train_loader)
                for _ in range(2):
                    try:
                        batch = next(it)
                        # Recursive device transfer helper
                        def to_device(item):
                            if isinstance(item, torch.Tensor):
                                return item.to(pl_module.device)
                            if isinstance(item, (list, tuple)):
                                return [to_device(i) for i in item]
                            return item
                        
                        batch = [to_device(b) for b in batch]
                        pl_module.model.step(batch, pl_module.loss_clsf, pl_module.loss_cosine, pl_module.acc_metric, mode="audit")
                    except StopIteration:
                        break
            pl_module.train()

# -----------------------------------------------------------------------------
# 4. Serving
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    wl.clear_all()
    params = wl.watch_or_edit({
        "experiment_name": "fashion_mnist_hinge_full", 
        "lr": 0.001, 
        "batch_size": 32,
        "root_log_dir": "./root_log_dir/fashion/4"
    }, flag="hp")
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = FashionMNISTSiamese(train=True, transform=transform)
    val_ds = FashionMNISTSiamese(train=False, transform=transform)
    
    train_loader = wl.watch_or_edit(
        train_ds, 
        flag="data", 
        loader_name="train_loader", 
        batch_size=params["batch_size"],
        shuffle=True,
        compute_hash=False,
    )
    
    val_loader = wl.watch_or_edit(
        val_ds, 
        flag="data", 
        loader_name="val_loader", 
        batch_size=params["batch_size"],
        shuffle=False,
        compute_hash=False,
    )
    
    # Determine device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create loss/metric objects and move to device
    loss_clsf = wl.watch_or_edit(nn.CrossEntropyLoss(reduction='none').to(device), flag='loss', signal_name='loss_clsf')
    loss_cosine = wl.watch_or_edit(nn.CosineEmbeddingLoss(margin=0.5, reduction='none').to(device), flag='loss', signal_name='loss_embed_cosine')
    acc_metric = wl.watch_or_edit(Accuracy(task="multiclass", num_classes=10).to(device), flag='metric', signal_name='val_metric/accuracy')

    # Initialize sub-model on device
    _backbone = FashionHingeBackbone().to(device)
    backbone_proxy = wl.watch_or_edit(_backbone, flag="model", device=device)
    
    # Wrap in clean LightningModule (NOT watched directly)
    model = FashionHingeModel(lr=params["lr"], backbone=backbone_proxy, loss_clsf=loss_clsf, loss_cosine=loss_cosine, acc_metric=acc_metric)
    
    wl.serve(serving_grpc=True)
    pause_controller.resume()
    
    # Custom callback to eval on training set periodically
    train_eval_callback = TrainEvalCallback(train_loader, every_n_steps=20)
    
    print("🚀 Starting training with PyTorch Lightning + WeightsLab Guards...")
    trainer = Trainer(
        max_epochs=5, 
        log_every_n_steps=5, 
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False, # WL SDK handles logging
        callbacks=[train_eval_callback],
        val_check_interval=50 # How often to run the official validation
    )
    trainer.fit(model, train_loader, val_loader)

    wl.keep_serving()
