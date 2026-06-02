"""YOLO-detection pipeline builder + prediction-decoder. Shared by main_ddp.py
(the DDP spawn shim) and ddp_test_suite.py (the integration suite). This module
is YOLO-specific example glue, NOT part of the WL SDK.
"""
import os
import shutil
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_IMGSZ = int(os.environ.get("WL_DDP_IMGSZ", "320"))
_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]


def _decode_preds_to_6col(raw_preds, image, conf, cls_thresh, device=None):
    from utils.criterions import _decode_predictions
    if isinstance(raw_preds, (tuple, list)):
        raw_preds = raw_preds[1]
    img_h, img_w = image[0].shape[-2:]
    pred = torch.cat([raw_preds['boxes'], raw_preds['scores']], dim=1)
    preds_bboxes, preds_cls = _decode_predictions(
        pred, img_h, img_w, conf=conf, iou_thres=cls_thresh)
    imgsz = float(image.shape[-1])
    return [
        torch.cat([b.detach() / imgsz, c[:, 1:2], c[:, 0:1]], dim=-1).to(device) if b.numel() > 0
        else torch.zeros((0, 6), device=device)
        for b, c in zip(preds_bboxes, preds_cls)
    ]


def _build_pipeline(cfg, device, rank, world_size):
    """Build the WL-wrapped dataset/loader/model/loss exactly like main.py, minus
    serve. Rank 0 uses a fixed, inspectable log dir wiped at startup; children
    use throwaway tmp dirs so concurrent writes can't corrupt a shared store.

    Returns (trainer, train_loader, criterions, my_uids, all_uids).
    """
    import weightslab as wl
    from weightslab.utils.logger import LoggerQueue
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from utils.data import YOLODatasetWL, _wl_yolo_collate
    from utils.criterions import PerSampleDetectionLoss, PerSampleIoU

    exp_name = cfg["experiment_name"]
    data_root = cfg["data_root"]
    model_name = cfg["model"]["name"]
    train_cfg = dict(cfg["data"]["train_loader"])
    # env overrides so the test suite can sweep batch / num_workers without
    # editing config.yaml. DDP default is 16 (not config's mono batch): two ranks
    # share one GPU using ~1.1GB at batch=4 of ~8GB available, so a bigger batch
    # = fewer steps = less per-step collective overhead (flush/reconcile/grad are
    # paid per-step regardless of batch) + better GPU use. Override via WL_DDP_BATCH.
    # (config.yaml batch_size is left as-is for the mono main.py example.)
    batch_size = int(os.environ.get("WL_DDP_BATCH", "16"))
    num_workers = int(os.environ.get("WL_DDP_WORKERS", "0"))

    # Clean slate each run: no checkpoint resume + start empty.
    cfg = dict(cfg)
    if rank == 0:
        cfg["root_log_dir"] = os.path.join(_HERE, "ddp_run")
        shutil.rmtree(cfg["root_log_dir"], ignore_errors=True)
        os.makedirs(cfg["root_log_dir"], exist_ok=True)
    else:
        cfg["root_log_dir"] = tempfile.mkdtemp(prefix=f"wl_ddp_r{rank}_")
    cfg["checkpoint_manager"] = {"load_config": False, "load_data": False, "load_model": False}
    cfg["experiment_dump_to_train_steps_ratio"] = 10**12  # no periodic dumps
    log_dir = cfg["root_log_dir"]
    wl.watch_or_edit(LoggerQueue(), flag="logger", name=exp_name, log_dir=log_dir)
    wl.watch_or_edit(dict(cfg), flag="hyperparameters", name=exp_name,
                     defaults=dict(cfg), poll_interval=1.0)

    import time as _time
    _bt = os.environ.get("WL_DDP_BUILD_TIMING") == "1" and rank == 0
    _t0 = _time.perf_counter()
    def _lap(label):
        nonlocal _t0
        if _bt:
            now = _time.perf_counter()
            print(f"[build_timing] {label:28s} {1000*(now-_t0):8.0f} ms", flush=True)
            _t0 = now

    ucfg = get_cfg()
    ucfg.imgsz = _IMGSZ; ucfg.rect = False; ucfg.single_cls = False
    ucfg.task = "detect"; ucfg.classes = None; ucfg.fraction = 1.0
    for k in ("mosaic", "mixup", "copy_paste", "hsv_h", "hsv_s", "hsv_v",
              "degrees", "translate", "scale", "shear", "perspective",
              "flipud", "fliplr", "erasing"):
        setattr(ucfg, k, 0.0)
    ucfg.auto_augment = None
    _lap("ucfg setup")
    checked = check_det_dataset(data_root)
    _lap("check_det_dataset")

    def _build_dataset():
        ds = YOLODataset(
            img_path=checked["train"], imgsz=ucfg.imgsz, batch_size=batch_size,
            augment=False, hyp=ucfg, rect=ucfg.rect, cache=False,
            single_cls=ucfg.single_cls or False, stride=32, pad=0.0,
            task=ucfg.task, classes=ucfg.classes, data=checked, fraction=1.0,
        )
        ds.__class__ = YOLODatasetWL
        return ds

    _ds = _build_dataset()
    _lap("YOLODataset build")
    train_loader = wl.watch_or_edit(
        _ds, flag="data", loader_name="train_loader",
        batch_size=batch_size, shuffle=train_cfg.get("shuffle", False),
        num_workers=num_workers, drop_last=True, compute_hash=False,
        is_training=True, collate_fn=_wl_yolo_collate,
        preload_labels=True, preload_metadata=True,
    )
    _lap("LEDGER INIT (data watch_or_edit)")

    # Rank-aware sharding is built into WeightsLabDataSampler (auto-detects rank/world);
    # we only read this rank's shard out for the partition check in the smoke report.
    tds = train_loader.tracked_dataset
    uid_src = getattr(tds, "physical_uids", None)
    if uid_src is None:
        uid_src = tds.unique_ids
    all_uids = [str(u) for u in uid_src]
    sampler = train_loader._mutable_batch_sampler
    my_uids = {all_uids[i] for i in sampler._rank_indices_snapshot()}

    class _SmokeTrainer(DetectionTrainer):
        def __init__(self, *a, **kw):
            self.data_train_loader = kw.pop("train_loader")
            # POP (not get) so the torch.device isn't forwarded to ultralytics'
            # DetectionTrainer, which expects a str/int device. self.device drives
            # the model placement below; without honoring it the model stayed on
            # CPU even under WL_DDP_CUDA=1 (the override device= was for ultralytics
            # only and never reached self.device).
            self.device = kw.pop("device", torch.device("cpu"))
            super().__init__(*a, **kw)
            super().setup_model()
            self.model = self.model.to(self.device)
            self._init_loss()
            self._setup_train()
            self.optimizer = wl.watch_or_edit(self.optimizer, flag="optimizer")
            self.model = wl.watch_or_edit(
                self.model, flag="model", device=self.device, compute_dependencies=False)
            self.model = self.model.to(self.device)  # watch_or_edit drops device= (known bug)

        def _init_loss(self):
            if not hasattr(self.model, "args"):
                self.model.args = get_cfg()
            self.criterions = {}
            for t, n in _LOSS_PARTS:
                self.criterions[n] = wl.watch_or_edit(
                    PerSampleDetectionLoss(self.model, loss_type=t),
                    flag="loss", name=f"train/{n}", per_sample=True, log=True)
            self.iou = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name="miou/train", per_sample=True, log=True)

    # ultralytics device string ("0" for cuda:0, "cpu" otherwise); self.device
    # (the torch.device) drives the actual model placement via .to() above.
    _ul_dev = str(device.index if device.type == "cuda" else "cpu")
    trainer = _SmokeTrainer(
        overrides=dict(model=model_name, data=str(data_root), epochs=1, imgsz=_IMGSZ,
                       batch=batch_size, resume=False, device=_ul_dev, workers=0,
                       cache=False, optimizer="SGD", lr0=0.01, plots=False),
        train_loader=train_loader,
        device=device,
    )
    _lap("trainer/model build (ultralytics)")
    return trainer, train_loader, trainer.criterions, my_uids, all_uids
