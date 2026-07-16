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
from weightslab.backend import ledgers

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

            # Get signals configuration from ledger
            signals_cfg = ledgers.get_hyperparams().get('signals_cfg', {})
            install_per_sample_signals(underlying, signals_cfg)
            install_per_sample_val_signals(trainer.validator, signals_cfg)

            _register_channels()

            # Decorated function for evaluation mode using trainer hook init.
            @wl.eval_fn
            def _validate(loader):
                # Clean pause ctrl callbacks
                import copy
                raised_exc = None
                val_loader = trainer.validator.dataloader
                try:
                    trainer.validator.dataloader = loader
                    # validator(model=...) takes UL's *standalone* path, which
                    # wraps the model in AutoBackend and FUSES it in place
                    # (conv+bn -> conv with bias, bn deleted). Fusing the live
                    # EMA/model rewrites its state_dict keys, so the next
                    # trainer.ema.update(model) raises KeyError 'model.*.conv.bias'.
                    # Validate on a throwaway deep copy so the live EMA / training
                    # model are never mutated. (`underlying` is the unwrapped model
                    # captured above; avoids deep-copying the WL wrapper.)
                    src = trainer.ema.ema if trainer.ema else underlying
                    eval_model = copy.deepcopy(src).eval()
                    trainer.validator(model=eval_model)
                except Exception as e:
                    raised_exc = e
                finally:
                    trainer.validator.dataloader = val_loader # Reset val loader

                # Finally raise exc.
                if raised_exc is not None:
                    raise raised_exc

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
                ("metrics/recall(B)", "val/recall"),
                ("metrics/mAP50(B)", "val/mAP50"),
                ("metrics/mAP50-95(B)", "val/mAP50-95"),
                ("fitness", "val/fitness"),
            ):
                if ul_key in rd and wl_key in ch:
                    ch[wl_key](torch.tensor([float(rd[ul_key])]))

        self.add_callback("on_train_start", _on_train_start)
        self.add_callback("on_train_batch_start", _on_train_batch_start)
        self.add_callback("on_train_batch_end", _on_train_batch_end)
        self.add_callback("on_val_batch_start", _on_val_batch_start)
        self.add_callback("on_val_batch_end", _on_val_batch_end)
        self.add_callback("on_val_end", _on_val_end)

    def validate(self):
        # UL's metrics.process does np.concatenate([]) → ValueError when val
        # is fully discarded. WL's sampler subtracts deny-listed uids in __len__,
        # so loader len reflects the active set. ({}, 0.0) unpacks cleanly into
        # UL's `{**self.metrics, ...}` which (None, None) does not.
        if len(self.test_loader) == 0:
            print("[WLAwareTrainer] val skipped: all val samples are discarded",
                  flush=True)
            return {}, 0.0
        return super().validate()

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        dataset.__class__ = WLAwareDataset
        is_train = (mode == "train")

        # Get data configuration from ledger
        data_cfg = ledgers.get_hyperparams()
        loader_name = "train_loader" if is_train else (data_cfg.get('data', {}).get('loader_name', 'val_loader') if not isinstance(data_cfg.get('data'), str) else 'val_loader')
        cfg = data_cfg.get('data', {}).get(loader_name, {}) if not isinstance(data_cfg.get('data'), str) else {}

        # Respect ledger-configured rect mode. Default False so val and train
        # share the same square letterbox geometry — UL otherwise hardcodes
        # rect=True for val, giving non-square per-batch padding that differs
        # from train.
        dataset.rect = cfg.get('rect', False)
        loader = wl.watch_or_edit(
            dataset,
            flag="data",
            loader_name=loader_name,
            batch_size=cfg.get('batch_size', batch_size),
            shuffle=is_train,
            num_workers=cfg.get('num_workers', 0),
            drop_last=cfg.get('drop_last', False),
            is_training=is_train,
            compute_hash=cfg.get('compute_hash', False),
            collate_fn=wl_ul_dict_collate,
            preload_labels=cfg.get('preload_labels', True),
            preload_metadata=cfg.get('preload_metadata', True),
        )

        # NOTE: no `reset` shim needed here. `loader` is a ledger Proxy whose
        # __getattr__ returns None for missing attrs (so `hasattr(loader,
        # "reset")` is always True); DataLoaderInterface.reset() provides the
        # real Ultralytics-compatible implementation that the proxy forwards to.
        return loader
