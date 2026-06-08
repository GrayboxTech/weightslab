"""WLAwareTrainer — UL DetectionTrainer subclass that wires WL through the
canonical pattern.

What this trainer does for you:
  * Builds both `train_loader` and `test_loader` through
    `wl.watch_or_edit(flag='data', loader_name=...)` so each split has its
    own `DataSampleTrackingWrapper` with a disjoint uid range — no
    cross-origin collisions in the shared ledger.
  * Registers the optimizer and model with WL on `on_train_start`.
  * Installs per-sample TRAIN signals (cls/box/dfl per image + live NMS
    predictions overlay).
  * Installs per-sample VAL signals (per-image IoU + AP@0.5 + post-NMS
    predictions overlay).
  * Ships aggregate train losses + val metrics (P/R/mAP50/mAP50-95/fitness)
    as curves through WL channels.
  * Wraps each train/val batch in WL's training/testing guard contexts.

Aggregate curves shipped via UL callbacks:
  * `train/{box,cls,dfl}` from `trainer.loss_items`
  * `val/{precision,recall,mAP50,mAP50-95,fitness}` from
    `validator.metrics.results_dict`
"""
from __future__ import annotations

import torch
from torch.nn import Identity

from ultralytics.models.yolo.detect import DetectionTrainer

import weightslab as wl
from weightslab.components.global_monitoring import pause_controller

from .collate import wl_ul_dict_collate
from .dataset import WLAwareDataset
from .signals import install_per_sample_signals, install_per_sample_val_signals


class WLAwareTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        state = {"channels": {}}

        def _register_channels():
            ch = state["channels"]
            for n in ("box", "cls", "dfl"):
                ch[f"train/{n}"] = wl.watch_or_edit(
                    Identity(), flag="loss", name=f"train/{n}", log=True,
                )
            for key in ("precision", "recall", "mAP50", "mAP50-95", "fitness"):
                ch[f"val/{key}"] = wl.watch_or_edit(
                    Identity(), flag="metric", name=f"val/{key}", log=True,
                )

        def _on_train_start(trainer):
            underlying = trainer.model
            trainer.optimizer = wl.watch_or_edit(trainer.optimizer, flag="optimizer")
            trainer.model = wl.watch_or_edit(
                trainer.model, flag="model", forced_model_wrapping=True,
                compute_dependencies=False,
            )

            install_per_sample_signals(underlying)
            install_per_sample_val_signals(trainer.validator)

            _register_channels()

            @wl.eval_fn
            def _validate(loader):
                trainer.validator(model=(trainer.ema.ema if trainer.ema else trainer.model))

            pause_controller.resume(force=True)

        def _on_train_batch_start(trainer):
            wl.guard_training_context.__enter__()

        def _on_train_batch_end(trainer):
            ch = state["channels"]
            li = getattr(trainer, "loss_items", None)
            if li is not None and ch:
                for i, n in enumerate(("box", "cls", "dfl")):
                    ch[f"train/{n}"](li[i:i+1].detach())
            wl.guard_training_context.__exit__(None, None, None)

        def _on_val_batch_start(validator):
            wl.guard_testing_context.__enter__()

        def _on_val_batch_end(validator):
            wl.guard_testing_context.__exit__(None, None, None)

        def _on_val_end(validator):
            ch = state["channels"]
            md = getattr(validator, "metrics", None)
            rd = getattr(md, "results_dict", None) if md is not None else None
            if not rd or not ch:
                return
            for ul_key, wl_key in (
                ("metrics/precision(B)", "val/precision"),
                ("metrics/recall(B)",    "val/recall"),
                ("metrics/mAP50(B)",     "val/mAP50"),
                ("metrics/mAP50-95(B)",  "val/mAP50-95"),
                ("fitness",              "val/fitness"),
            ):
                if ul_key in rd and wl_key in ch:
                    ch[wl_key](torch.tensor([float(rd[ul_key])]))

        # WL_CORRECTNESS=1: at every 50 steps print
        #   model_sum_abs   — sum |p| over all weights. Must change → optimizer
        #                     stepping. Constant ⇒ training is dead.
        #   ema_sum_abs     — same on EMA. Should drift apart from model_sum_abs.
        #   ema_model_l2    — L2 of (ema - model) flattened. Should grow then
        #                     plateau. Stuck at 0 ⇒ EMA never updates;
        #                     stuck at init value ⇒ EMA frozen.
        import os as _os
        _correctness = _os.environ.get("WL_CORRECTNESS", "0") == "1"
        _state2 = {"step": 0}

        def _on_train_batch_end_correctness(trainer):
            if not _correctness:
                return
            _state2["step"] += 1
            if _state2["step"] % 50 != 0:
                return
            import sys
            mi = trainer.model
            inner = getattr(mi, "_obj", mi)
            inner = inner.__dict__.get("model", inner)
            try:
                msum = sum(p.detach().float().abs().sum().item() for p in inner.parameters())
            except Exception as e:
                msum = float("nan")
            ema = getattr(getattr(trainer, "ema", None), "ema", None)
            if ema is not None:
                esum = sum(p.detach().float().abs().sum().item() for p in ema.parameters())
                m_flat = torch.cat([p.detach().flatten() for p in inner.parameters()])
                e_flat = torch.cat([p.detach().flatten() for p in ema.parameters()])
                ediff = float((e_flat.float() - m_flat.float()).norm())
            else:
                esum, ediff = float("nan"), float("nan")
            print(f"[WL correctness] step #{_state2['step']}: "
                  f"model_sum_abs={msum:.4f} ema_sum_abs={esum:.4f} "
                  f"||ema-model||_2={ediff:.6f}",
                  file=sys.stderr, flush=True)

        self.add_callback("on_train_start",       _on_train_start)
        self.add_callback("on_train_batch_start", _on_train_batch_start)
        self.add_callback("on_train_batch_end",   _on_train_batch_end)
        self.add_callback("on_train_batch_end",   _on_train_batch_end_correctness)
        self.add_callback("on_val_batch_start",   _on_val_batch_start)
        self.add_callback("on_val_batch_end",     _on_val_batch_end)
        self.add_callback("on_val_end",           _on_val_end)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Both train and val go through `wl.watch_or_edit(flag='data')` —
        the canonical WL pattern. Each loader gets its own
        `DataSampleTrackingWrapper` with its own uid range from the global
        `_UID_CNT` counter, so train and val never collide in the shared
        ledger."""
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        dataset.__class__ = WLAwareDataset
        is_train = (mode == "train")
        loader = wl.watch_or_edit(
            dataset, flag="data",
            loader_name="train_loader" if is_train else "val_loader",
            batch_size=batch_size, shuffle=is_train,
            num_workers=0, drop_last=False,
            is_training=is_train, compute_hash=False,
            collate_fn=wl_ul_dict_collate,
            preload_labels=True, preload_metadata=True,
        )
        if not hasattr(loader, "reset"):
            loader.reset = lambda *a, **k: None
        return loader
