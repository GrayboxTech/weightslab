"""
Face embedding model.

Architecture
------------
Pretrained backbone  (ResNet-18 / ResNet-50 / MobileNet-V3-Small)
    ?
EmbeddingHead  Linear ? BN ? ReLU ? Linear
    ?
L2-normalised D-dimensional embedding

The backbone is optionally frozen so that only the lightweight head is trained
(recommended toy-example setup).  The combined graph is registered with
WeightsLAB for model tracking.

Public interface
----------------
FaceEmbeddingModel.get_embeddings(images)          ? normalised embeddings (B, D)
FaceEmbeddingModel.train_step(images, labels, ...)  ? scalar loss float
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import weightslab as wl

from torchvision import models
from typing import List

from face.utils import mine_batch_hard

logger = logging.getLogger(__name__)


# ============================================================
# Network modules
# ============================================================

class EmbeddingHead(nn.Module):
    """Projection head: backbone_feature_dim ? embedding_dim (L2-normalised)."""

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class FaceEmbeddingNet(nn.Module):
    """Single nn.Module combining the frozen backbone and trainable head."""

    def __init__(self, backbone: nn.Module, head: EmbeddingHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)      # (B, feature_dim)
        embeddings = self.head(features)   # (B, embedding_dim), L2-normalised
        return embeddings


class TripletLossWithMetadata(nn.Module):
    """Triplet loss wrapper that logs anchor-level triplet metadata in WeightsLAB.

    The anchor UID is used as batch ID. For each anchor we persist the selected
    positive and negative UIDs into per-sample signal columns so they remain
    attached to image/sample metadata in the ledger.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2, reduce=False)

    def forward(
        self,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        anchor_ids: List[str],
        pos_ids: List[str],
        neg_ids: List[str],
        name: str = "train",
    ) -> torch.Tensor:
        loss = self.criterion(anchor_emb, pos_emb, neg_emb)

        wl.save_signals(
            batch_ids=anchor_ids,
            signals={f"{name}/loss/triplet": loss.detach().cpu().numpy()},
            preds_raw=anchor_emb,
            targets=neg_emb,
            preds=None,
            log=True,
        )

        # Store pivot triplet composition as per-sample metadata-like signals.
        wl.save_signals(
            batch_ids=anchor_ids,
            signals={
                f"{name}/meta/triplet_pos_uid": pos_ids,
                f"{name}/meta/triplet_neg_uid": neg_ids,
            },
            preds_raw=None,
            targets=None,
            preds=None,
            log=False,
        )
        return loss


# ============================================================
# High-level model wrapper
# ============================================================

class FaceEmbeddingModel:
    """Wrapper that manages the backbone + head, optimiser, and WeightsLAB tracking.

    Args:
        backbone_name:   "resnet18" | "resnet50" | "mobilenet_v3_small"
        embedding_dim:   Output embedding dimensionality (default 128).
        head_hidden_dim: Hidden size of the projection MLP (default 256).
        lr:              Learning rate for AdamW (default 1e-3).
        weight_decay:    AdamW weight decay (default 1e-4).
        freeze_backbone: When True, only the head's parameters receive
                         gradients ? recommended for quick toy runs.
        device:          "cpu", "cuda", or "cuda:N".
        pretrained:      Load ImageNet-pretrained weights for the backbone.
        margin:          Triplet margin (default 0.3).
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        embedding_dim: int = 128,
        head_hidden_dim: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = True,
        device: str = "cpu",
        pretrained: bool = True,
        margin: float = 0.3,
    ):
        self.device = torch.device(device)
        self.margin = margin
        self.embedding_dim = embedding_dim

        # ---- Build backbone ----
        weights = "DEFAULT" if pretrained else None
        backbone_feature_dim = self._build_backbone(backbone_name, weights, freeze_backbone)

        # ---- Build head ----
        head = EmbeddingHead(
            in_dim=backbone_feature_dim,
            hidden_dim=head_hidden_dim,
            embedding_dim=embedding_dim,
        )

        # ---- Compose & move to device ----
        self.net = FaceEmbeddingNet(backbone=self._backbone, head=head).to(self.device)

        # ---- WeightsLAB graph tracking ----
        self.net = wl.watch_or_edit(
            self.net,
            flag="model",
            compute_dependencies=False,
            device=str(self.device),
        )

        # ---- WeightsLAB tracked loss ----
        self.triplet_loss_fn = TripletLossWithMetadata(margin=self.margin)
        self.triplet_loss_fn.__name__ = "tripletcriterion"
        watched_loss = wl.watch_or_edit(
            self.triplet_loss_fn,
            flag="loss",
            signal_name="triplet_with_metadata",
            log=False,
            use_batch_ids_as_x=True,
            use_batch_value_as_y=False,
        )
        if watched_loss is not None and hasattr(watched_loss, "forward"):
            self.triplet_loss_fn = watched_loss

        # ---- Optimiser (head-only when backbone is frozen) ----
        trainable = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=lr, weight_decay=weight_decay
        )

        n_trainable = sum(p.numel() for p in trainable)
        logger.info(
            f"FaceEmbeddingModel | backbone={backbone_name} pretrained={pretrained} "
            f"frozen={freeze_backbone} | emb_dim={embedding_dim} | "
            f"trainable_params={n_trainable:,}"
        )
        print(
            f"  Backbone  : {backbone_name}  (pretrained={pretrained}, frozen={freeze_backbone})\n"
            f"  Emb dim   : {embedding_dim}\n"
            f"  Head dim  : {head_hidden_dim}\n"
            f"  Trainable : {n_trainable:,} params\n"
            f"  Device    : {self.device}"
        )

    def _build_backbone(
        self,
        backbone_name: str,
        weights,
        freeze: bool,
    ) -> int:
        """Instantiate backbone, strip its classifier, optionally freeze.

        Sets self._backbone and returns the feature dimensionality.
        """
        if backbone_name == "resnet18":
            base = models.resnet18(weights=weights)
            feature_dim = 512
            base.fc = nn.Identity()
        elif backbone_name == "resnet50":
            base = models.resnet50(weights=weights)
            feature_dim = 2048
            base.fc = nn.Identity()
        elif backbone_name == "mobilenet_v3_small":
            base = models.mobilenet_v3_small(weights=weights)
            feature_dim = 576
            base.classifier = nn.Identity()
        else:
            raise ValueError(
                f"Unsupported backbone {backbone_name!r}. "
                "Choose from 'resnet18', 'resnet50', 'mobilenet_v3_small'."
            )

        if freeze:
            for param in base.parameters():
                param.requires_grad_(False)
            logger.info("Backbone frozen ? training head only.")

        self._backbone = base
        return feature_dim

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Forward pass in eval mode; returns L2-normalised embeddings (B, D).

        Args:
            images: (B, C, H, W) float tensor (GPU/CPU ? moved internally)

        Returns:
            (B, D) embedding tensor on CPU
        """
        self.net.eval()
        with torch.no_grad():
            emb = self.net(images.to(self.device))
        return emb.cpu()

    # ----------------------------------------------------------
    # Training step
    # ----------------------------------------------------------

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_ids: List[str],
        loss_name: str = "triplet",
    ) -> float:
        """One gradient update using online batch-hard triplet mining.

        Args:
            images:    (B, C, H, W) float tensor
            labels:    (B,) long tensor of identity ids
            batch_ids: list of sample UIDs for WeightsLAB signal logging
            loss_name: "triplet" (contrastive support planned)

        Returns:
            Scalar loss value (Python float)
        """
        self.net.train()
        self.optimizer.zero_grad()

        images = images.to(self.device)
        labels = labels.to(self.device)

        embeddings = self.net(images)                           # (B, D)

        # Mine hardest triplets in the batch
        anc_idx, pos_idx, neg_idx = mine_batch_hard(embeddings, labels)

        if anc_idx.numel() == 0:
            logger.warning(
                "No valid triplets found in batch (all samples same class?); "
                "skipping gradient step."
            )
            return 0.0

        anc_emb = embeddings[anc_idx]
        pos_emb = embeddings[pos_idx]
        neg_emb = embeddings[neg_idx]

        triplet_ids = [batch_ids[i] for i in anc_idx.tolist()]
        pos_triplet_ids = [batch_ids[i] for i in pos_idx.tolist()]
        neg_triplet_ids = [batch_ids[i] for i in neg_idx.tolist()]

        if loss_name == "triplet":
            loss_by_sample = self.triplet_loss_fn(
                anchor_emb=anc_emb,
                pos_emb=pos_emb,
                neg_emb=neg_emb,
                anchor_ids=triplet_ids,
                pos_ids=pos_triplet_ids,
                neg_ids=neg_triplet_ids,
                name="train",
            )
        else:
            raise ValueError(
                f"Unsupported loss {loss_name!r}. Use 'triplet'."
            )
        # Aggregate per-sample losses into a single scalar and backprop
        loss = loss_by_sample.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.net.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()

        return float(loss.item())
