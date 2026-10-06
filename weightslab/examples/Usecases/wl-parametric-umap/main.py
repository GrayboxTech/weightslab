"""Live parametric-UMAP projection, end to end, on synthetic clustered data.

Nothing here opts the projection in: wrapping the model with
``wl.watch_or_edit(..., flag="model")`` attaches it, because the feature is on
by default. What this example shows is that the coordinates land on the samples
as ordinary per-sample signals, and that the 3-D layout recovers the structure
the data actually has.

Synthetic blobs rather than MNIST on purpose: the ground-truth cluster of every
sample is known, so the run can *assert* that the projection separated them
instead of asking you to squint at a scatter plot.

Run it::

    python main.py

Turn the projection off and watch the columns disappear::

    WEIGHTSLAB_PROJECTION=0 python main.py

Other knobs: ``WEIGHTSLAB_PROJECTION_EVERY`` (fit cadence, default 50 steps),
``WEIGHTSLAB_PROJECTION_DIM`` (3), ``WEIGHTSLAB_PROJECTION_NEIGHBORS`` (15),
``WEIGHTSLAB_PROJECTION_GRAPH`` (512).

It ends by plugging in projections of its own -- the same path a t-SNE or a
umap-learn layout of yours would take -- so the board's picker offers
``umap``, ``umap_trunk``, ``pca`` and ``pca_inputs`` side by side.
"""

import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

# Fit often, because this demo only trains for a few hundred steps.
os.environ.setdefault("WEIGHTSLAB_PROJECTION_EVERY", "5")

import weightslab as wl  # noqa: E402
from weightslab.proto import experiment_service_pb2 as pb2  # noqa: E402
from weightslab.trainer.services import projection_service as ps  # noqa: E402

LOSS = "loss_sample"
CLUSTERS = 6
PER_CLUSTER = 256
DIM = 32
BATCH = 64
OUT = os.environ.get("WL_UMAP_OUT") or os.path.join(tempfile.gettempdir(), "wl_umap_demo")


class Blobs(Dataset):
    """Well-separated Gaussian blobs in 32-D. Yields ``(x, uid, label)`` -- the
    uid is what ties a row in the dataframe to a point in the projection."""

    def __init__(self, clusters=CLUSTERS, per_cluster=PER_CLUSTER, dim=DIM, seed=0):
        rng = np.random.default_rng(seed)
        centers = rng.normal(scale=5.0, size=(clusters, dim))
        self.x = np.concatenate(
            [centers[c] + rng.normal(scale=0.6, size=(per_cluster, dim))
             for c in range(clusters)]).astype(np.float32)
        self.y = np.concatenate(
            [np.full(per_cluster, c) for c in range(clusters)]).astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), i, int(self.y[i])

    def fast_get_label(self, i):
        return int(self.y[i])


class Net(nn.Module):
    """trunk -> 16-D features -> classifier. The projection auto-hooks the
    classifier's INPUT, so it watches the 16-D representation, not the logits."""

    def __init__(self, dim=DIM, feat=16, classes=CLUSTERS):
        super().__init__()
        self.input_shape = (1, dim)
        self.trunk = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, feat), nn.ReLU())
        self.classifier = nn.Linear(feat, classes)

    def forward(self, x):
        return self.classifier(self.trunk(x))


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hp = {
        "experiment_name": "parametric-umap-demo",
        "device": device,
        "root_log_dir": os.path.join(OUT, "wl_logs"),
        "serving_grpc": False,
        "serving_cli": False,
        "ledger_enable_h5_persistence": False,
        "experiment_dump_to_train_steps_ratio": 10_000_000,
        "data": {"train_loader": {"batch_size": BATCH, "shuffle": True}},
    }
    wl.watch_or_edit(hp, flag="hyperparameters", defaults=hp)

    dataset = Blobs()
    # Wrapping the model is all it takes -- the projection attaches here.
    model = wl.watch_or_edit(Net().to(device), flag="model", device=device)
    opt = wl.watch_or_edit(optim.Adam(model.parameters(), lr=3e-3), flag="optimizer")
    loader = wl.watch_or_edit(dataset, flag="data", loader_name="train_loader",
                              batch_size=BATCH, shuffle=True, is_training=True,
                              preload_labels=True)
    crit = wl.watch_or_edit(nn.CrossEntropyLoss(reduction="none"),
                            flag="loss", signal_name=LOSS, per_sample=True, log=True)

    wl.serve(serving_grpc=False, serving_cli=False)
    wl.start_training(timeout=0)

    tracker = wl.projection.get_tracker()
    if tracker is None:
        print("projection is DISABLED (WEIGHTSLAB_PROJECTION=0) -- no coordinates "
              "will be written; the board stays hidden in the studio.")
    else:
        print(f"projection attached: every {tracker.every_n_steps} steps -> "
              f"{tracker.out_dim}-D, columns {tracker.stats()['columns']}")

    print(f"\ntraining {CLUSTERS} clusters x {PER_CLUSTER} samples on {device}...")
    for epoch in range(6):
        for x, ids, y in loader:
            x, y = x.to(device), y.to(device)
            with wl.guard_training_context:
                logits = model(x)
                loss = crit(logits, y, batch_ids=ids, preds=logits.argmax(1, keepdim=True))
                opt.zero_grad()
                loss.mean().backward()
                opt.step()
        if tracker is not None:
            print(f"  epoch {epoch + 1}: {tracker.steps_trained} fits, "
                  f"{tracker.samples_written} coordinates written, "
                  f"umap loss {tracker.last_loss:.4f}")

    if tracker is None:
        return

    # ---- read the projection back exactly as the UI does -------------------
    # save_signals buffers; a live server sees the rows via the manager's own
    # background flush thread, but this process is about to exit, so drain it.
    from weightslab.backend.ledgers import get_dataframe
    manager = get_dataframe()
    manager.flush()
    frame = manager._df
    # Colour by the ground-truth class: the response then carries a per-point
    # label aligned with the coordinates, so no id lookup is needed -- and it
    # exercises the same colour path the board's "colour by" picker uses.
    response = ps.build_projection_response(
        frame, pb2.ProjectionRequest(max_points=100_000, color_column="target"))
    print(f"\nGetProjection: success={response.success} :: {response.message}")
    if not response.success:
        return

    coords = np.array(response.coords, dtype=np.float32).reshape(-1, 3)
    # Ground truth arrives aligned with the points (color_column="target"), so
    # there is nothing to join: WeightsLab assigns its own sample uids and they
    # are what the projection is keyed by, not the dataset's own index.
    # Class ids come back as categorical LABELS ("0.0".."5.0"), not as a numeric
    # ramp: a class id is a name, and ramping it would paint neighbouring
    # classes near-identical shades.
    labels = np.array(response.color_labels).astype(float).astype(np.int64)
    print(f"  {response.returned} points, extent "
          f"x[{response.extent_min_x:.2f}, {response.extent_max_x:.2f}] "
          f"y[{response.extent_min_y:.2f}, {response.extent_max_y:.2f}] "
          f"z[{response.extent_min_z:.2f}, {response.extent_max_z:.2f}]")

    # Did the 3-D layout keep the structure? Compare the mean distance between
    # cluster centroids against the mean spread within a cluster.
    centroids = np.stack([coords[labels == c].mean(axis=0) for c in range(CLUSTERS)])
    within = float(np.mean([
        np.linalg.norm(coords[labels == c] - centroids[c], axis=1).mean()
        for c in range(CLUSTERS)]))
    between = float(np.mean([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(CLUSTERS) for j in range(i + 1, CLUSTERS)]))
    print(f"  cluster separation: between/within = {between / max(within, 1e-9):.1f}x "
          f"(within {within:.3f}, between {between:.3f})")

    # ---- re-projecting offline, from a different layer ----------------------
    # The live projection watched `classifier`'s input (the 16-D features). Say
    # we decide that was the wrong layer: re-project from the trunk instead,
    # with no training step, under its own prefix so both survive and the board
    # can switch between them.
    print("\nre-projecting offline from 'trunk.2' (no training step)...")
    offline = wl.project_dataset(model, loader, layer="trunk.2",
                                 prefix="umap_trunk", epochs=8, verbose=False)
    print(f"  {offline['samples']} samples x {offline['feature_dim']} features "
          f"-> {offline['columns']} (loss {offline['final_loss']:.4f})")

    manager.flush()
    frame = manager._df
    print(f"  projections now stored: {ps.available_prefixes(frame)}")

    alt = ps.build_projection_response(
        frame, pb2.ProjectionRequest(prefix="umap_trunk", color_column="target"))
    alt_xyz = np.array(alt.coords, dtype=np.float32).reshape(-1, 3)
    alt_labels = np.array(alt.color_labels).astype(float).astype(np.int64)
    alt_cent = np.stack([alt_xyz[alt_labels == c].mean(axis=0) for c in range(CLUSTERS)])
    alt_within = float(np.mean([
        np.linalg.norm(alt_xyz[alt_labels == c] - alt_cent[c], axis=1).mean()
        for c in range(CLUSTERS)]))
    alt_between = float(np.mean([
        np.linalg.norm(alt_cent[i] - alt_cent[j])
        for i in range(CLUSTERS) for j in range(i + 1, CLUSTERS)]))
    print(f"  'umap_trunk' separation: {alt_between / max(alt_within, 1e-9):.1f}x "
          f"(vs {between / max(within, 1e-9):.1f}x for the live 'umap')")

    # ---- bring your own projection -----------------------------------------
    # Any algorithm plugs into the same board: it draws every <prefix>_x/_y(/_z)
    # it finds. A 3-component PCA here (torch only, so the demo needs nothing
    # extra); sklearn's TSNE(n_components=3) or umap.UMAP(n_components=3) drop
    # in the same way, since anything with fit_transform is accepted.
    def pca3(features):
        centered = torch.from_numpy(features)
        centered = centered - centered.mean(dim=0)
        _, _, v = torch.pca_lowrank(centered, q=3)
        return (centered @ v[:, :3]).numpy()

    print("\nyour own projection, on features WeightsLab collects (method=)...")
    mine = wl.project_dataset(model, loader, method=pca3, prefix="pca", verbose=False)
    print(f"  {mine['samples']} samples x {mine['feature_dim']} features -> {mine['columns']}")

    # ...or compute everything yourself and hand over the coordinates: a PCA of
    # the raw INPUTS, to set against what the model learned.
    xs, uids = [], []
    for x, ids, _ in loader:
        xs.append(x)
        uids.extend(ids.tolist() if hasattr(ids, "tolist") else list(ids))
    raw = pca3(torch.cat(xs).numpy())
    print(f"  wl.save_projection_coords -> "
          f"{wl.save_projection_coords(raw, batch_ids=uids, prefix='pca_inputs')}")

    manager.flush()
    frame = manager._df
    print(f"  projections now stored: {ps.available_prefixes(frame)}")

    # ---- the level-of-detail contract the 3-D board relies on --------------
    box = pb2.ProjectionRequest(
        has_bounds=True,
        min_x=float(coords[:, 0].min()), min_y=float(coords[:, 1].min()),
        min_z=float(coords[:, 2].min()),
        max_x=float(np.median(coords[:, 0])), max_y=float(coords[:, 1].max()),
        max_z=float(coords[:, 2].max()),
        max_points=200)
    zoomed = ps.build_projection_response(frame, box)
    print(f"  zoomed box: {zoomed.returned} of {zoomed.total_in_view} in view "
          f"(budget 200, {zoomed.total_available} projected overall)")


if __name__ == "__main__":
    main()
