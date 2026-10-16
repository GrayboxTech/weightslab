"""The notebook this example drops into the run directory.

Weights Studio's notebook runs Python IN the training process, against the live
model and data, so these cells see the converged model itself -- no export, no
reload. Running a cell pauses training first (the run has already paused itself
at convergence, so nothing moves while you work).

The cells are YOUR code: WeightsLab neither collects features nor fits anything
here. It is handed the finished coordinates with ``wl.save_projection_coords``
and draws them in the Projection Board.
"""

import json
from pathlib import Path

NOTEBOOK_NAME = "own-projection.ipynb"

INTRO = """\
# Your own projection of the converged model

WeightsLab's built-in UMAP is **off** in this run (`projection=False`). Everything
below is your own code: collect the representation, run your own t-SNE, then hand
the coordinates to the Projection Board.

1. **Features** — the 128-D input of the classifier head, for every test image.
2. **Your t-SNE** — exact t-SNE written from scratch in plain torch.
3. **Plug it in** — `wl.save_projection_coords(...)`, then open
   *Data Exploration → Projection* and pick `tsne`.
4. *(optional)* a preview here, and a second algorithm (PCA) to compare.

Running a cell pauses training first; press **Play** in the header to resume.
"""

FEATURES = """\
# 1) The representation to project: what flows INTO the classifier (fc2),
#    for every test image, from the live converged model.
import torch
from weightslab.backend.ledgers import get_dataloader, get_model

SPLIT = "test_loader"        # or "train_loader" (more points, slower t-SNE)

net = wl.projection.resolve_module(get_model())    # the FashionCNN itself
device = next(net.parameters()).device
was_training = net.training
net.eval()
features, ids, labels = [], [], []
with torch.no_grad():
    for x, batch_ids, y in get_dataloader(SPLIT):
        features.append(net.embed(x.to(device)).cpu())
        ids.extend(batch_ids.tolist() if hasattr(batch_ids, "tolist") else list(batch_ids))
        labels.append(torch.as_tensor(y))
net.train(was_training)

features = torch.cat(features)
labels = torch.cat(labels).numpy()
print(f"{features.shape[0]} samples x {features.shape[1]} features from {SPLIT}")
"""

TSNE = """\
# 2) Your own t-SNE (van der Maaten & Hinton, 2008) -- exact, O(N^2), plain torch.
#    Fine for a few thousand points; swap in sklearn / openTSNE / umap-learn the
#    same way if you prefer -- only the coordinates matter to WeightsLab.
import math, time

def tsne(X, dim=2, perplexity=30.0, iters=750, lr=200.0, exaggeration=12.0, seed=0,
         device="cuda" if torch.cuda.is_available() else "cpu"):
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    X = X - X.mean(0)
    if X.shape[1] > 50:                                    # usual t-SNE practice
        _, _, V = torch.pca_lowrank(X, q=50, center=False)
        X = X @ V[:, :50]
    n = X.shape[0]

    def sq_dists(A):                                        # one matmul, no sqrt
        s = (A * A).sum(1)
        return (s[:, None] + s[None, :] - 2.0 * A @ A.T).clamp_min(0.0)

    # P: Gaussian affinities, one bandwidth per point so every row has the target
    # perplexity. Distances are shifted by each row's nearest neighbour first, so
    # exp() never underflows -- the shift cancels in the normalisation.
    D = sq_dists(X)
    D.fill_diagonal_(float("inf"))
    D = D - D.min(1, keepdim=True).values
    D.fill_diagonal_(0.0)
    target = math.log(perplexity)
    beta = 1.0 / D.mean(1, keepdim=True).clamp_min(1e-12)
    lo, hi = torch.zeros_like(beta), torch.full_like(beta, float("inf"))
    for _ in range(64):
        P = torch.exp(-D * beta)
        P.fill_diagonal_(0.0)
        sum_p = P.sum(1, keepdim=True)
        H = torch.log(sum_p) + beta * (D * P).sum(1, keepdim=True) / sum_p
        too_flat = H > target                               # sharpen those rows
        lo = torch.where(too_flat, beta, lo)
        hi = torch.where(too_flat, hi, beta)
        beta = torch.where(torch.isinf(hi), beta * 2, (lo + hi) / 2)
    P = P / sum_p
    P = ((P + P.T) / (2 * n)).clamp_min(1e-12)

    # Y: Student-t affinities, gradient descent with momentum and gains.
    g = torch.Generator().manual_seed(seed)
    Y = (torch.randn(n, dim, generator=g) * 1e-4).to(device)
    update, gains = torch.zeros_like(Y), torch.ones_like(Y)
    for it in range(iters):
        num = 1.0 / (1.0 + sq_dists(Y))
        num.fill_diagonal_(0.0)
        Q = (num / num.sum()).clamp_min(1e-12)
        W = ((exaggeration if it < 250 else 1.0) * P - Q) * num
        grad = 4.0 * (W.sum(1, keepdim=True) * Y - W @ Y)
        gains = torch.where((grad > 0) == (update > 0), gains * 0.8, gains + 0.2).clamp_min(0.01)
        update = (0.5 if it < 250 else 0.8) * update - lr * gains * grad
        Y = Y + update
        Y = Y - Y.mean(0)
        if it % 150 == 0 or it == iters - 1:
            print(f"  iter {it:4d}   KL(P||Q) = {float((P * torch.log(P / Q)).sum()):.3f}")
    return Y.cpu().numpy()

t0 = time.time()
tsne_coords = tsne(features.numpy(), dim=2)
print(f"t-SNE of {len(tsne_coords)} points in {time.time() - t0:.1f}s")
"""

PLUG = """\
# 3) Plug it into the board. Any name works as long as it is not "umap" while
#    the built-in projection runs; a 2-D result is drawn on the z = 0 plane.
columns = wl.save_projection_coords(tsne_coords, batch_ids=ids, prefix="tsne")
print("written:", columns)
print("Now open Data Exploration -> Projection: 'tsne' is in the picker.")
print("Lasso a region there to load those images into the grid.")
"""

PREVIEW = """\
# 4a) Optional: a quick look here, coloured by class.
import matplotlib.pyplot as plt

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
fig, ax = plt.subplots(figsize=(7, 6))
for c, name in enumerate(CLASSES):
    m = labels == c
    ax.scatter(tsne_coords[m, 0], tsne_coords[m, 1], s=3, label=name)
ax.legend(markerscale=4, fontsize=7, loc="best")
ax.set_title("Your t-SNE of the converged representation")
ax.set_xticks([]); ax.set_yticks([])
"""

PCA = """\
# 4b) Optional: a second algorithm, to compare in the board's picker. A 3-D PCA,
#     three lines of torch -- linear, so it shows how much of the class structure
#     is already linearly separable in the representation.
centered = features - features.mean(0)
_, _, V = torch.pca_lowrank(centered, q=3, center=False)
pca_coords = (centered @ V[:, :3]).numpy()
print("written:", wl.save_projection_coords(pca_coords, batch_ids=ids, prefix="pca"))
"""


def _cell(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def notebook() -> dict:
    return {
        "cells": [
            _cell("markdown", INTRO),
            _cell("code", FEATURES),
            _cell("code", TSNE),
            _cell("code", PLUG),
            _cell("code", PREVIEW),
            _cell("code", PCA),
        ],
        "metadata": {"kernelspec": {"name": "weightslab",
                                    "display_name": "WeightsLab (shared)"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_projection_notebook(root_log_dir) -> Path:
    """Write the notebook into *root_log_dir*, unless one of that name is there.

    An existing file is left alone: it may hold your own edits from an earlier
    run in the same directory. The Studio opens the most recently written
    notebook of the run directory, so this one is what it shows first.
    """
    path = Path(root_log_dir) / NOTEBOOK_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook(), indent=1), encoding="utf-8")
    return path
