"""Stage ul06_wl04 — manual train/val loop using Ultralytics primitives.

Diff from ul08_wl02:
  * Replace `model.train()` with an explicit loop. Use Ultralytics building
    blocks directly: subclass `DetectionTrainer` (or call `setup_model` +
    `_setup_train`), iterate batches manually, call `v8DetectionLoss` and
    step optimizer/scheduler.
  * Keeps Ultralytics' dataset + loss + model + optimizer build — gives WL
    a clean surface to attach to in the next stage.

Diff toward main.py (ul04_wl06):
  * No WL imports, no `watch_or_edit`, no logger, no guard contexts, no
    serve. The leap to main.py is purely additive: wrap the model, optimizer,
    dataset, losses, and hyperparameters with `wl.watch_or_edit(...)`, gate
    train/val phases with `wl.guard_*_context`, and start `wl.serve()`.
"""
# TODO: implement
