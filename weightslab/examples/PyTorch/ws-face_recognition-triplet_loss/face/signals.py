"""
Signals (losses and metrics) for face recognition training.

Follows the WeightsLAB pattern: every helper calls wl.save_signals
immediately after computing the value so the platform captures it.

Classes
-------
TripletLosses  differentiable loss functions (return torch.Tensor)
FaceMetrics    evaluation metrics and clustering-oriented test signals
"""

import numpy as np
import torch
import torch.nn as nn

import weightslab as wl

from typing import Dict

from face.utils import (
    compute_rank1_accuracy,
    compute_similarity_grouping,
    compute_verification_metrics,
)


# ============================================================
# Loss functions
# ============================================================

class TripletLosses:
    """Differentiable loss functions for face embedding training."""

    @staticmethod
    def triplet_loss(
        ids,
        anchor_emb: torch.Tensor,
        pos_emb: torch.Tensor,
        neg_emb: torch.Tensor,
        margin: float = 0.3,
        name: str = "train",
    ) -> torch.Tensor:
        """Standard triplet margin loss on pre-mined (a, p, n) embeddings."""
        loss = nn.TripletMarginLoss(margin=margin, p=2)(anchor_emb, pos_emb, neg_emb)
        wl.save_signals(
            batch_ids=ids,
            signals={f"{name}/loss/triplet": loss.detach().cpu().item()},
            preds_raw=anchor_emb,
            targets=neg_emb,
            preds=None,
            log=True,
        )
        return loss

    @staticmethod
    def contrastive_loss(
        ids,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        same_label: torch.Tensor,
        margin: float = 1.0,
        name: str = "train",
    ) -> torch.Tensor:
        """Contrastive loss (Hadsell et al., 2006)."""
        dist = nn.functional.pairwise_distance(emb_a, emb_b)
        loss = (
            same_label * dist.pow(2)
            + (1.0 - same_label) * (margin - dist).clamp(min=0.0).pow(2)
        ).mean()
        wl.save_signals(
            batch_ids=ids,
            signals={f"{name}/loss/contrastive": loss.detach().cpu().item()},
            preds_raw=emb_a,
            targets=emb_b,
            preds=None,
            log=True,
        )
        return loss


# ============================================================
# Evaluation metrics
# ============================================================

class FaceMetrics:
    """Evaluation metrics for face verification, retrieval, and grouping."""

    @staticmethod
    def verification_accuracy(
        ids,
        embeddings: np.ndarray,
        labels: np.ndarray,
        name: str = "test",
    ) -> Dict[str, float]:
        """Pairwise verification accuracy with optimal threshold search."""
        result = compute_verification_metrics(embeddings, labels)
        wl.save_signals(
            batch_ids=ids,
            signals={
                f"{name}/metric/verification_accuracy": result["verification_accuracy"],
                f"{name}/metric/far": result["far"],
                f"{name}/metric/frr": result["frr"],
                f"{name}/metric/best_threshold": result["best_threshold"],
            },
            preds_raw=None,
            targets=None,
            preds=None,
            log=True,
        )
        return result

    @staticmethod
    def rank1_accuracy(
        ids,
        embeddings: np.ndarray,
        labels: np.ndarray,
        name: str = "test",
    ) -> float:
        """1-NN Rank-1 retrieval accuracy (leave-one-out)."""
        acc = compute_rank1_accuracy(embeddings, labels)
        wl.save_signals(
            batch_ids=ids,
            signals={f"{name}/metric/rank1_accuracy": acc},
            preds_raw=None,
            targets=None,
            preds=None,
            log=True,
        )
        return acc

    @staticmethod
    def similarity_grouping_signals(
        ids,
        embeddings: np.ndarray,
        name: str = "test",
        cluster_eps: float = 0.6,
        cluster_min_samples: int = 2,
    ) -> Dict[str, float]:
        """Save per-sample grouping signals for later clustering/sorting in Studio."""
        grouping = compute_similarity_grouping(
            embeddings=embeddings,
            uids=list(ids),
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
        )

        wl.save_signals(
            batch_ids=ids,
            signals={
                f"{name}/cluster/id": grouping["cluster_id"],
                f"{name}/cluster/size": grouping["cluster_size"],
                f"{name}/cluster/nn1_distance": grouping["nn1_distance"],
                f"{name}/cluster/nn1_uid": grouping["nn1_uid"],
            },
            preds_raw=None,
            targets=None,
            preds=None,
            log=False,
        )

        summary = {
            "num_clusters": grouping["num_clusters"],
            "noise_ratio": grouping["noise_ratio"],
            "mean_nn1_distance": grouping["mean_nn1_distance"],
        }
        wl.save_signals(
            batch_ids=ids,
            signals={
                f"{name}/cluster/num_clusters": summary["num_clusters"],
                f"{name}/cluster/noise_ratio": summary["noise_ratio"],
                f"{name}/cluster/mean_nn1_distance": summary["mean_nn1_distance"],
            },
            preds_raw=None,
            targets=None,
            preds=None,
            log=True,
        )
        return summary

    @staticmethod
    def compute_all_metrics(
        ids,
        embeddings: np.ndarray,
        labels: np.ndarray,
        name: str = "test",
    ) -> Dict[str, float]:
        """Run all metrics and return a merged dict."""
        verif = FaceMetrics.verification_accuracy(ids, embeddings, labels, name=name)
        rank1 = FaceMetrics.rank1_accuracy(ids, embeddings, labels, name=name)
        grouping = FaceMetrics.similarity_grouping_signals(ids, embeddings, name=name)
        return {**verif, "rank1_accuracy": rank1, **grouping}
