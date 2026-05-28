"""Benchmark variant of main.py: configurable num_workers + epochs, with timing and progress prints.

Env vars:
    WL_BENCH_WORKERS: num_workers for both train & val loaders (default 0)
    WL_BENCH_EPOCHS:  number of train epochs to run (default 1)
    WL_BENCH_LOG_EVERY: log loss/iou every N steps (default 20)
"""
import os
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")

import warnings
from tables import NaturalNameWarning
warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="Setting an item of incompatible dtype is deprecated.*")

import logging
import tempfile
import time

import torch
import yaml

import weightslab as wl
from weightslab.utils.logger import LoggerQueue
from weightslab.components.global_monitoring import pause_controller
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset

from utils.data import YOLODatasetWL, _wl_yolo_collate
from utils.criterions import PerSampleDetectionLoss, PerSampleIoU, _decode_predictions

logging.getLogger("weightslab.watchdog.grpc_watchdog").setLevel(logging.ERROR)
logging.getLogger("weightslab.trainer.services.agent.agent").setLevel(logging.ERROR)


_WORKERS = int(os.environ.get("WL_BENCH_WORKERS", "0"))
_EPOCHS = int(os.environ.get("WL_BENCH_EPOCHS", "1"))
_WALL_S = float(os.environ.get("WL_BENCH_WALL_S", "0") or "0")  # 0 disables wall-time mode
_LOG_EVERY = int(os.environ.get("WL_BENCH_LOG_EVERY", "20"))
_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]


def _decode_preds_to_6col(raw_preds, image, conf, cls_thresh, device=None):
    if isinstance(raw_preds, (tuple, list)):
        raw_preds = raw_preds[1]
    img_h, img_w = image[0].shape[-2:]
    pred = torch.cat([raw_preds['boxes'], raw_preds['scores']], dim=1)
    preds_bboxes, preds_cls = _decode_predictions(pred, img_h, img_w, conf=conf, iou_thres=cls_thresh)
    imgsz = float(image.shape[-1])
    return [
        torch.cat([b.detach() / imgsz, c[:, 1:2], c[:, 0:1]], dim=-1).to(device) if b.numel() > 0
        else torch.zeros((0, 6), device=device)
        for b, c in zip(preds_bboxes, preds_cls)
    ]


class BenchTrainer(DetectionTrainer):
    def __init__(self, *a, **kw):
        self.data_train_loader = kw.pop("train_loader")
        self.data_val_loader = kw.pop("val_loader")
        self.hparams = kw.pop("hparams", {})
        self.device = kw.get("device", torch.device("cpu"))
        self._bench_max_steps = kw.pop("max_steps")
        super().__init__(*a, **kw)
        self._init_experiment_modules()

    @property
    def val_every(self):
        if os.environ.get("WL_BENCH_NO_VAL") == "1":
            return 1 << 30  # effectively never, for clean training-throughput benchmarking
        return self.hparams.get("eval_full_to_train_steps_ratio", 1)

    def _init_experiment_modules(self):
        super().setup_model()
        self.model = self.model.to(self.device)
        self._init_training_loss()
        self._setup_train()
        self.optimizer = wl.watch_or_edit(self.optimizer, flag="optimizer")
        self.model = wl.watch_or_edit(
            self.model, flag="model", device=self.device, compute_dependencies=False)
        # watch_or_edit(flag="model") drops its device= kwarg (returns the proxy without
        # honoring it), leaving the model on CPU — force it back onto the target device.
        self.model = self.model.to(self.device)

    def _init_training_loss(self):
        if not hasattr(self.model, "args"):
            self.model.args = get_cfg()
        self.criterions = {"train": {}, "val": {}}
        self.iou = {}
        for split in ("train", "val"):
            for t, n in _LOSS_PARTS:
                self.criterions[split][n] = wl.watch_or_edit(
                    PerSampleDetectionLoss(self.model, loss_type=t),
                    flag="loss", name=f"{split}/{n}", per_sample=True, log=True,
                )
            self.iou[split] = wl.watch_or_edit(
                PerSampleIoU(conf=0.25, iou_thres=0.5),
                flag="metric", name=f"miou/{split}", per_sample=True, log=True,
            )

    def train(self):
        cs, m = self.criterions["train"], self.iou["train"]
        max_steps = self._bench_max_steps
        log_every = _LOG_EVERY

        def _infinite(loader):
            while True:
                yield from loader
        batches = _infinite(self.data_train_loader)

        t_train_start = time.perf_counter()
        wall_deadline = (t_train_start + _WALL_S) if _WALL_S > 0 else float("inf")
        step = 0
        loss_sum = 0.0
        last_loss = float("nan")
        step_t = time.perf_counter()
        while step < max_steps and time.perf_counter() < wall_deadline:
            with wl.guard_training_context:
                self.optimizer.zero_grad()
                inputs = next(batches)
                image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']
                raw_preds = self.model(image.to(self.device))
                preds_by_batch = _decode_preds_to_6col(raw_preds, image, conf=0.1, cls_thresh=0.1)
                per_sample = (
                    cs["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
                    + cs["clsf"](raw_preds, batch, batch_ids=batch_ids)
                    + cs["dfl"](raw_preds, batch, batch_ids=batch_ids)
                )
                loss = per_sample.mean()
                m(raw_preds, batch, batch_ids=batch_ids)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            last_loss = float(loss.detach().cpu())
            loss_sum += last_loss
            step += 1
            if step % log_every == 0 or step == max_steps:
                dt = time.perf_counter() - step_t
                ips = (log_every * image.shape[0]) / max(dt, 1e-9)
                print(f"  [step {step}/{max_steps}] loss={loss_sum/log_every:.4f}  {dt:.2f}s  ({ips:.1f} img/s)", flush=True)
                loss_sum = 0.0
                step_t = time.perf_counter()

            if step % self.val_every == 0 and step < max_steps:
                with wl.guard_testing_context:
                    val_start = time.perf_counter()
                    self.do_validate(self.data_val_loader)
                    print(f"  [val @ step {step}] {time.perf_counter()-val_start:.2f}s", flush=True)

        total = time.perf_counter() - t_train_start
        try:
            age = int(self.model.get_age())
        except Exception:
            age = -1
        print(f"  TRAIN DONE: {step} steps in {total:.2f}s  ({step/total:.2f} steps/s)  model_age={age}  last_loss={last_loss:.4f}", flush=True)
        return total

    def do_validate(self, loader):
        cs, m = self.criterions["val"], self.iou["val"]
        for inputs in loader:
            image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']
            raw_preds = self.model(image.to(self.device))[1]
            preds_by_batch = _decode_preds_to_6col(
                raw_preds, image, conf=0.25, cls_thresh=0.5, device=self.device)
            cs["bbxs"](raw_preds, batch, batch_ids=batch_ids, preds={'bboxes': preds_by_batch})
            cs["clsf"](raw_preds, batch, batch_ids=batch_ids)
            cs["dfl"](raw_preds, batch, batch_ids=batch_ids)
            m(raw_preds, batch, batch_ids=batch_ids)


def main():
    print('[T0] main() entered', flush=True)
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parameters = yaml.safe_load(open(config_path)) if os.path.exists(config_path) else {}
    print('[T1] config loaded', flush=True)

    if parameters.get("device", "auto") == "auto":
        parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not parameters.get("root_log_dir"):
        parameters["root_log_dir"] = tempfile.mkdtemp()
    os.makedirs(parameters["root_log_dir"], exist_ok=True)

    exp_name = parameters["experiment_name"]
    log_dir = parameters["root_log_dir"]
    device = parameters["device"]
    image_size = int(os.environ.get("WL_BENCH_IMGSZ") or parameters.get("image_size"))
    data_root = parameters["data_root"]
    model_name = parameters["model"]["name"]
    train_cfg = dict(parameters["data"]["train_loader"])
    test_cfg = dict(parameters["data"]["test_loader"])
    batch_size = train_cfg["batch_size"]
    serving_grpc = parameters.get("serving_grpc", True)
    serving_cli = parameters.get("serving_cli", False)

    wl.watch_or_edit(LoggerQueue(), flag="logger", name=exp_name, log_dir=log_dir)
    parameters = wl.watch_or_edit(parameters, flag="hyperparameters",
                                  name=exp_name, defaults=parameters, poll_interval=1.0)

    cfg = get_cfg()
    cfg.imgsz = image_size; cfg.rect = False; cfg.single_cls = False
    cfg.task = "detect"; cfg.classes = None; cfg.fraction = 1.0
    checked_data = check_det_dataset(data_root)

    def _build_dataset(split):
        is_train = split == "train"
        ds = YOLODataset(
            img_path=checked_data["train" if is_train else "val"],
            imgsz=cfg.imgsz, batch_size=batch_size,
            augment=is_train, hyp=cfg, rect=cfg.rect, cache=False,
            single_cls=cfg.single_cls or False,
            stride=32, pad=0.0 if is_train else 0.5,
            task=cfg.task, classes=cfg.classes, data=checked_data,
            fraction=cfg.fraction if is_train else 1.0,
        )
        ds.__class__ = YOLODatasetWL
        return ds

    def _build_loader(ds, split):
        c = train_cfg if split == "train" else test_cfg
        return wl.watch_or_edit(
            ds, flag="data", loader_name=f"{split}_loader",
            batch_size=c["batch_size"], shuffle=c["shuffle"],
            num_workers=_WORKERS,
            drop_last=False, compute_hash=False,
            is_training=(split == "train"),
            collate_fn=_wl_yolo_collate,
            preload_labels=True, preload_metadata=True,
        )

    print('[T2] building train dataset', flush=True)
    _tr_ds = _build_dataset("train")
    print('[T3] building train loader', flush=True)
    train_loader = _build_loader(_tr_ds, "train")
    print('[T4] building val dataset+loader', flush=True)
    val_loader = _build_loader(_build_dataset("val"), "val")
    print('[T5] loaders built', flush=True)

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * _EPOCHS

    print(f"=== BENCH START ===", flush=True)
    print(f"  workers={_WORKERS} epochs={_EPOCHS} wall_s={_WALL_S} steps_per_epoch={steps_per_epoch} max_steps={max_steps}", flush=True)
    print(f"  device={device}  batch_train={train_cfg['batch_size']} batch_val={test_cfg['batch_size']}", flush=True)
    print(f"  train_samples={len(train_loader.tracked_dataset)} val_samples={len(val_loader.tracked_dataset)}", flush=True)

    wl.serve(serving_grpc=serving_grpc, serving_cli=serving_cli)

    trainer = BenchTrainer(
        overrides=dict(
            model=model_name, data=str(data_root),
            epochs=_EPOCHS, imgsz=image_size,
            batch=batch_size, resume=False, device=device,
            workers=_WORKERS, cache=False, optimizer="SGD", lr0=0.001,
        ),
        train_loader=train_loader, val_loader=val_loader,
        hparams=parameters, max_steps=max_steps,
    )

    @wl.eval_fn
    def _wl_validate(loader):
        trainer.do_validate(loader)

    pause_controller.resume(force=True)
    t0 = time.perf_counter()
    trainer.train()
    wall = time.perf_counter() - t0
    print(f"=== BENCH END workers={_WORKERS} epochs={_EPOCHS} wall={wall:.2f}s ===", flush=True)
    # exit cleanly — do NOT call wl.keep_serving()


if __name__ == "__main__":
    main()
