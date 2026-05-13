"""
Utility functions for face recognition training.
- Pairwise distance computation
- Online batch-hard triplet mining
- Verification and Rank-1 evaluation helpers
- Similarity grouping signals for clustering-oriented analysis
"""

import numpy as np
import torch

from typing import Tuple, Dict, List, Any


# ============================================================
# Distance helpers
# ============================================================

def pairwise_distances(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """Compute pairwise L2 distance matrix for a batch of embeddings.

    Args:
        embeddings: (B, D) tensor
        squared:    return squared L2 distances when True

    Returns:
        (B, B) distance matrix
    """
    dot = torch.matmul(embeddings, embeddings.t())
    sq_norms = torch.diagonal(dot)                            # (B,)
    distances = sq_norms.unsqueeze(0) - 2.0 * dot + sq_norms.unsqueeze(1)
    distances = distances.clamp(min=0.0)

    if not squared:
        # Avoid NaN gradients at exactly 0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


# ============================================================
# Triplet mining
# ============================================================

def mine_batch_hard(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    squared: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online batch-hard triplet mining (Hermans et al., 2017).

    For each anchor selects:
    - Hardest positive : same-class sample with the **largest** distance
    - Hardest negative : different-class sample with the **smallest** distance

    Args:
        embeddings: (B, D) detached from graph during mining
        labels:     (B,) integer class ids
        squared:    use squared L2 distances for mining

    Returns:
        anc_idx, pos_idx, neg_idx 1-D LongTensors; only valid anchors
        (those that have at least one positive peer) are included.
    """
    dist_mat = pairwise_distances(embeddings.detach(), squared=squared)
    B = labels.shape[0]
    device = labels.device

    same = labels.unsqueeze(0) == labels.unsqueeze(1)   # (B, B)
    diff = ~same
    eye = torch.eye(B, dtype=torch.bool, device=device)

    # Hardest positive (exclude self)
    pos_mask = same & ~eye
    pos_dist = dist_mat * pos_mask.float()
    _, pos_idx = pos_dist.max(dim=1)

    # Hardest negative (mask invalid positions with large value)
    neg_dist = dist_mat + (~diff).float() * 1e9
    _, neg_idx = neg_dist.min(dim=1)

    # Keep only anchors that have at least one positive peer in the batch
    valid = pos_mask.any(dim=1)
    anc_idx = torch.where(valid)[0]

    return anc_idx, pos_idx[anc_idx], neg_idx[anc_idx]


# ============================================================
# Numpy-level evaluation helpers
# ============================================================

def compute_verification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Pairwise verification accuracy over a threshold grid."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 2.0, 100)

    n = len(embeddings)

    # Pairwise L2 distances
    dot = embeddings @ embeddings.T                           # (N, N)
    sq = np.sum(embeddings ** 2, axis=1)
    dist_mat = (sq[:, None] - 2.0 * dot + sq[None, :]).clip(min=0.0)
    dist_mat = np.sqrt(dist_mat.clip(min=1e-16)) * (dist_mat != 0.0)

    same_pair = labels[:, None] == labels[None, :]            # (N, N)

    # Upper triangle only (avoid double-counting)
    iu = np.triu_indices(n, k=1)
    dist_pairs = dist_mat[iu]
    is_same_pair = same_pair[iu]

    n_same = int(is_same_pair.sum())
    n_diff = int((~is_same_pair).sum())

    best_acc = 0.0
    best_threshold = float(thresholds[len(thresholds) // 2])
    best_far = 0.0
    best_frr = 0.0

    for thr in thresholds:
        pred_same = dist_pairs <= thr
        tp = int((pred_same & is_same_pair).sum())
        fp = int((pred_same & ~is_same_pair).sum())
        fn = int((~pred_same & is_same_pair).sum())
        tn = int((~pred_same & ~is_same_pair).sum())

        total = n_same + n_diff
        acc = (tp + tn) / total if total > 0 else 0.0
        far = fp / n_diff if n_diff > 0 else 0.0
        frr = fn / n_same if n_same > 0 else 0.0

        if acc > best_acc:
            best_acc = acc
            best_threshold = float(thr)
            best_far = far
            best_frr = frr

    return {
        "verification_accuracy": float(best_acc),
        "best_threshold": float(best_threshold),
        "far": float(best_far),
        "frr": float(best_frr),
    }


def compute_rank1_accuracy(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """1-NN Rank-1 retrieval accuracy (leave-one-out)."""
    n = len(embeddings)
    dot = embeddings @ embeddings.T
    sq = np.sum(embeddings ** 2, axis=1)
    dist_mat = (sq[:, None] - 2.0 * dot + sq[None, :]).clip(min=0.0)

    # Exclude self
    np.fill_diagonal(dist_mat, 1e9)
    nn_idx = dist_mat.argmin(axis=1)

    correct = int(np.sum(labels[nn_idx] == labels))
    return correct / n


def compute_similarity_grouping(
    embeddings: np.ndarray,
    uids: List[str],
    cluster_eps: float = 0.6,
    cluster_min_samples: int = 2,
) -> Dict[str, Any]:
    """Build per-sample similarity grouping signals for test-set clustering.

    Returns per-sample arrays that can be written as WeightsLAB signals:
    - cluster_id: DBSCAN cluster index (-1 for noise)
    - cluster_size: size of assigned cluster (1 for noise)
    - nn1_uid: nearest-neighbor UID in the evaluated set
    - nn1_distance: nearest-neighbor L2 distance
    """
    n = len(embeddings)
    if n == 0:
        return {
            "cluster_id": np.array([], dtype=np.int32),
            "cluster_size": np.array([], dtype=np.int32),
            "nn1_uid": [],
            "nn1_distance": np.array([], dtype=np.float32),
            "num_clusters": 0.0,
            "noise_ratio": 0.0,
            "mean_nn1_distance": float("nan"),
        }

    # Pairwise distances for NN lookup
    dot = embeddings @ embeddings.T
    sq = np.sum(embeddings ** 2, axis=1)
    dist_mat = (sq[:, None] - 2.0 * dot + sq[None, :]).clip(min=0.0)
    dist_mat = np.sqrt(dist_mat.clip(min=1e-16))

    if n == 1:
        nn_idx = np.array([0], dtype=np.int64)
        nn_dist = np.array([0.0], dtype=np.float32)
    else:
        np.fill_diagonal(dist_mat, 1e9)
        nn_idx = dist_mat.argmin(axis=1)
        nn_dist = dist_mat[np.arange(n), nn_idx].astype(np.float32)

    nn_uid = [uids[int(i)] for i in nn_idx.tolist()]

    # Unsupervised clustering on embeddings
    try:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samples, metric="euclidean").fit_predict(embeddings)
    except Exception:
        labels = np.full((n,), -1, dtype=np.int32)

    labels = labels.astype(np.int32)
    valid_labels = labels[labels >= 0]
    num_clusters = int(len(np.unique(valid_labels))) if valid_labels.size > 0 else 0

    counts = {}
    if valid_labels.size > 0:
        uniq, cnt = np.unique(valid_labels, return_counts=True)
        counts = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}

    cluster_size = np.array([counts.get(int(c), 1) if int(c) >= 0 else 1 for c in labels], dtype=np.int32)
    noise_ratio = float(np.mean(labels < 0))

    return {
        "cluster_id": labels,
        "cluster_size": cluster_size,
        "nn1_uid": nn_uid,
        "nn1_distance": nn_dist,
        "num_clusters": float(num_clusters),
        "noise_ratio": noise_ratio,
        "mean_nn1_distance": float(nn_dist.mean()) if len(nn_dist) > 0 else float("nan"),
    }
