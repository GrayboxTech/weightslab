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
import torch.optim as optim
import os
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

# -----------------------------------------------------------------------------
# 1. Dataset: Fashion MNIST with Deterministic Pairing
# -----------------------------------------------------------------------------
class FashionMNISTSiamese(torch.utils.data.Dataset):
    def __init__(self, train=True, transform=None):
        self.split = "train" if train else "val"
        self.base_ds = datasets.FashionMNIST(root="./data", train=train, download=True, transform=transform)

        # Deterministic shuffle for stable pairing
        import random
        self.indices = list(range(len(self.base_ds)))
        random.seed(42)
        random.shuffle(self.indices)

    def __len__(self):
        return len(self.base_ds) // 2

    def __getitem__(self, idx):
        idx1 = self.indices[idx * 2]
        idx2 = self.indices[idx * 2 + 1]

        img1, t1 = self.base_ds[idx1]
        img2, t2 = self.base_ds[idx2]

        # String UIDs for per-sample tracking
        uid1 = f"{self.split}_sample_{idx1}_left"
        uid2 = f"{self.split}_sample_{idx2}_right"
        # group_id as str — avoids tensor-wrapping mismatch during collation
        offset = 1_000_000 if self.split == "val" else 0
        group_id = str(idx + offset)

        return (
            [img1, img2],
            [uid1, uid2],            # string UIDs, match metadata['uids']
            [t1, t2],
            {
                "group_id": group_id,
                "uids": [uid1, uid2],  # tells ledger to expand to 2 rows
                "pair_type": "same" if t1 == t2 else "different"
            }
        )


# -----------------------------------------------------------------------------
# 2. Backbone (plain nn.Module)
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


# -----------------------------------------------------------------------------
# 3. LightningModule — used as an organised container; NOT passed to Trainer
# -----------------------------------------------------------------------------
class LitFashionHinge(pl.LightningModule):
    def __init__(self, model, optimizer, loss_clsf, loss_cosine, acc_metric):
        super().__init__()
        self.model       = model
        self.optimizer   = optimizer
        self.loss_clsf   = loss_clsf
        self.loss_cosine = loss_cosine
        self.acc_metric  = acc_metric

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, origin):
        """Shared logic for train and val steps."""
        images, uids, targets, metadata = batch

        x_flat = torch.cat([img.float() for img in images], dim=0)
        t_flat = torch.cat(targets, dim=0)

        # Flatten UIDs to a list of strings
        uids_flat = []
        for part in uids:
            part = part.detach().cpu() if hasattr(part, "detach") else part
            uids_flat.extend([str(x) for x in (part.tolist() if hasattr(part, "tolist") else part)])

        logits, embed = self(x_flat)
        preds = logits.argmax(dim=1)

        # Classification loss — per-sample via batch_ids
        # WeightsLab backend automatically masks discarded samples
        err_cls = self.loss_clsf(logits, t_flat, batch_ids=uids_flat, preds=preds)

        # Cosine embedding loss — per pair
        e1, e2 = embed[:len(embed) // 2], embed[len(embed) // 2:]
        t1, t2 = t_flat[:len(t_flat) // 2], t_flat[len(t_flat) // 2:]
        y = (t1 == t2).float() * 2 - 1
        
        pair_ids = uids_flat[:len(uids_flat) // 2]
        group_ids = [str(g) for g in metadata["group_id"]]
        
        # WeightsLab backend automatically masks tainted groups (broken pairs)
        loss_embed = self.loss_cosine(e1, e2, y, batch_ids=pair_ids, group_id=group_ids)

        wl.save_group_signals(
            signals={"loss_embed_cosine": loss_embed},
            group_ids=group_ids,
            origin=origin,
        )

        total_loss = err_cls.mean() + loss_embed.mean()

        # Update accuracy metric during validation
        if origin == "val_loader":
            self.acc_metric.update(logits, t_flat)

        return total_loss

    def training_step(self, batch):
        with guard_training_context:
            return self._step(batch, origin="train_loader")

    def validation_step(self, batch):
        with guard_testing_context:
            loss = self._step(batch, origin="val_loader")
            # acc_metric update runs inside the step via logits reuse
            return loss

    def configure_optimizers(self):
        return self.optimizer


# -----------------------------------------------------------------------------
# 4. Manual training loop — no Trainer involved
# -----------------------------------------------------------------------------
def to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, (list, tuple)):
        return [to_device(i, device) for i in item]
    return item


if __name__ == "__main__":
    wl.clear_all()
    params = wl.watch_or_edit({
        "experiment_name": "fashion_mnist_n_item",
        "lr": 0.001,
        "batch_size": 32,
        "root_log_dir": "./root_log_dir/fashion/9",
        "data": {
            "train_loader": {"batch_size": 32},
            "val_loader":   {"batch_size": 32},
        }
    }, flag="hp")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_ds = FashionMNISTSiamese(train=True,  transform=transform)
    val_ds   = FashionMNISTSiamese(train=False, transform=transform)

    train_loader = wl.watch_or_edit(
        train_ds, flag="data", loader_name="train_loader",
        batch_size=params["batch_size"], shuffle=True, compute_hash=False,
    )
    val_loader = wl.watch_or_edit(
        val_ds, flag="data", loader_name="val_loader",
        batch_size=params["batch_size"], shuffle=False, compute_hash=False,
    )

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Watched loss / metric objects
    loss_clsf   = wl.watch_or_edit(nn.CrossEntropyLoss(reduction='none').to(device),                flag='loss',   signal_name='loss_clsf')
    loss_cosine = wl.watch_or_edit(nn.CosineEmbeddingLoss(margin=0.5, reduction='none').to(device), flag='loss',   signal_name='loss_embed_cosine')
    acc_metric  = wl.watch_or_edit(Accuracy(task="multiclass", num_classes=10).to(device),          flag='metric', signal_name='val_metric/accuracy')

    # WL-wrapped model and optimizer
    _model    = FashionHingeBackbone().to(device)
    model     = wl.watch_or_edit(_model, flag="model", device=device)
    _optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    optimizer  = wl.watch_or_edit(_optimizer, flag="optimizer")

    # Instantiate the LightningModule as a pure container
    lit = LitFashionHinge(
        model=model, optimizer=optimizer,
        loss_clsf=loss_clsf, loss_cosine=loss_cosine, acc_metric=acc_metric
    )

    wl.serve(serving_grpc=True)
    pause_controller.resume()

    print("🚀 Starting manual training loop (LightningModule as container)...")
    max_epochs = 5
    eval_every = 50  # validate every N steps

    for epoch in range(max_epochs):
        lit.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = to_device(batch, device)

            # --- training step ---
            optimizer.zero_grad()
            loss = lit.training_step(batch)   # uses guard_training_context internally
            loss.backward()
            optimizer.step()

            model_age = model.get_age() if hasattr(model, "get_age") else (epoch * len(train_loader) + batch_idx)

            # --- periodic validation ---
            if model_age > 0 and model_age % eval_every == 0:
                lit.eval()
                val_ran = False
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_batch = to_device(val_batch, device)
                        lit.validation_step(val_batch)
                        val_ran = True

                if val_ran:
                    val_acc = acc_metric.compute()
                    print(f"Step {model_age:>5} | Val Acc: {val_acc:.4f}")
                    acc_metric.reset()
                lit.train()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:>4}/{len(train_loader)} | Loss: {loss.item():.4f}")

    print("✅ Training complete.")
    wl.keep_serving()
