"""DDP ablation: WeightsLab's TRUE internal tax, against the honest baseline.

The fair baseline isn't "no logging" — anyone wanting per-sample signals must decode
preds, compute per-sample loss/metrics, and store them. So we compare two modes with
identical model / batch / imgsz / data:
  ulmanual — ultralytics + a HAND-ROLLED minimal per-sample logger: decode + per-sample
              loss/IoU + append the scalars to a plain list. The "classic" way.
  wl — full WL pipeline: wrapped model/loss/loader, save_signals, anchor
              (reconcile DOWN + flush UP), decode-for-logging.

(wl - ulmanual) = WL's internal machinery (dataframe upserts + ledger/H5 + the DDP
anchor) ABOVE doing it by hand. The decode + per-sample loss are computed in BOTH, so
they cancel in the delta — what's left is purely WL. Per mode it prints per-section
ms/step + rank-0 RSS; `wl` also prints the global dataframe RAM + H5 store sizes.

  WL_ABLATE=ulmanual WL_DDP_CUDA=1 python ddp_ablation.py
  WL_ABLATE=wl WL_DDP_CUDA=1 python ddp_ablation.py
"""
import os
os.environ.setdefault("WEIGHTSLAB_SKIP_SECURE_INIT", "true")
os.environ.setdefault("GRPC_TLS_ENABLED", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "ERROR")
os.environ.setdefault("WEIGHTSLAB_LOG_TO_FILE", "false")
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WL_DDP_IMGSZ", "96")
os.environ.setdefault("WEIGHTSLAB_DISABLE_WATCHDOGS", "1")
os.environ.setdefault("WL_ENABLE_HP_SYNC", "0")

import sys, socket, time, statistics
import yaml, torch
import torch.distributed as dist
import torch.multiprocessing as mp

# This harness lives under tests/ but drives the wl-detection usecase: put the usecase
# src on the path (for yolo_pipeline / utils.*) and resolve config/data/ddp_run vs IT.
_USECASE_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../../examples/PyTorch/wl-detection/src"))
sys.path.insert(0, _USECASE_SRC)
_HERE = _USECASE_SRC
_HOST = "127.0.0.1"
_WARMUP = 5
_STEPS = int(os.environ.get("WL_ABLATE_STEPS", "25"))
MODE = os.environ.get("WL_ABLATE", "wl")
_LOSS_PARTS = [(0, "bbxs"), (1, "clsf"), (2, "dfl")]


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1024.0 # KB -> MB
    except Exception:
        pass
    return -1.0


def _proc_io():
    """Per-process I/O counters (Linux /proc/self/io). rchar/wchar = bytes through
    read()/write() syscalls; read_bytes/write_bytes = actual block-device I/O
    (0 when served from page cache); syscr/syscw = syscall counts."""
    out = {"rchar": 0, "wchar": 0, "read_bytes": 0, "write_bytes": 0, "syscr": 0, "syscw": 0}
    try:
        with open("/proc/self/io") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                if k in out:
                    out[k] = int(v)
    except Exception:
        pass
    return out


class T:
    def __init__(self): self.d = {}
    def add(self, k, dt): self.d.setdefault(k, []).append(dt)
    def ms(self, k): return 1000 * statistics.mean(self.d[k]) if self.d.get(k) else 0.0


def _build_ul(cfg, device, batch_size, num_workers):
    """Raw ultralytics: model + criterions + loader, NO weightslab."""
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.data import YOLODataset
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from torch.utils.data import DataLoader
    from utils.criterions import PerSampleDetectionLoss, PerSampleIoU
    from utils.data import YOLODatasetWL, _wl_yolo_collate

    data_root = cfg["data_root"]
    model_name = cfg["model"]["name"]
    imgsz = int(os.environ["WL_DDP_IMGSZ"])
    ucfg = get_cfg(overrides=dict(imgsz=imgsz, task="detect", rect=False, single_cls=False))
    checked = check_det_dataset(data_root)
    ds = YOLODataset(img_path=checked["train"], imgsz=imgsz, batch_size=batch_size,
                     augment=False, hyp=ucfg, rect=False, cache=False, single_cls=False,
                     stride=32, pad=0.0, task="detect", classes=None, data=checked, fraction=1.0)
    ds.__class__ = YOLODatasetWL
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, collate_fn=_wl_yolo_collate,
                        persistent_workers=num_workers > 0)

    trainer = DetectionTrainer(overrides=dict(
        model=model_name, data=str(data_root), epochs=1, imgsz=imgsz, batch=batch_size,
        resume=False, device=str(device.index if device.type == "cuda" else "cpu"),
        workers=0, cache=False, optimizer="SGD", lr0=0.01, plots=False))
    trainer.device = device
    trainer.setup_model()
    trainer.model = trainer.model.to(device)
    if not hasattr(trainer.model, "args"):
        trainer.model.args = get_cfg()
    trainer._setup_train()
    crit = {n: PerSampleDetectionLoss(trainer.model, loss_type=t).to(device) for t, n in _LOSS_PARTS}
    iou = PerSampleIoU(conf=0.25, iou_thres=0.5).to(device)
    return trainer.model, loader, crit, iou, trainer.optimizer


def _worker(rank, world, master_port):
    os.environ["MASTER_ADDR"] = _HOST
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    use_cuda = torch.cuda.is_available() and os.environ.get("WL_DDP_CUDA", "0") == "1"
    if use_cuda:
        torch.cuda.set_device(0)
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")

    cfg = yaml.safe_load(open(os.path.join(_HERE, "config.yaml")))
    batch_size = int(os.environ.get("WL_DDP_BATCH", "16"))
    num_workers = int(os.environ.get("WL_DDP_WORKERS", "0"))

    is_wl = MODE == "wl" # else: ulmanual (the hand-rolled classic baseline)
    if is_wl:
        import yolo_pipeline
        cfg["compute_natural_sort"] = False
        cfg["data"]["train_loader"]["shuffle"] = True
        if rank != 0:
            dist.barrier()
        trainer, loader, crit, _m, _a = yolo_pipeline._build_pipeline(cfg, device, rank, world)
        if rank == 0:
            dist.barrier()
        model, optimizer, iou = trainer.model, trainer.optimizer, trainer.iou
        from weightslab.components.parallel_primitives import (
            _ensure_core_ddp_registered, reconcile_all, flush_outbox)
        _ensure_core_ddp_registered()
        import weightslab as wl
        wl.serve(serving_grpc=True, serving_cli=False)
        decode = yolo_pipeline._decode_preds_to_6col
    else:
        model, loader, crit, iou, optimizer = _build_ul(cfg, device, batch_size, num_workers)
        from yolo_pipeline import _decode_preds_to_6col as decode
    _manual_store = [] # the "classic" sink: a plain in-memory list

    # identical initial weights on every rank (flattened broadcast)
    with torch.no_grad():
        params = [p for p in model.parameters()]
        flat = torch._utils._flatten_dense_tensors([p.data for p in params])
        dist.broadcast(flat, src=0)
        for p, s in zip(params, torch._utils._unflatten_dense_tensors(flat, [p.data for p in params])):
            p.data.copy_(s)

    def _inf(ld):
        while True:
            yield from ld
    batches = _inf(loader)
    t = T()
    io0 = None
    grad_bytes = 0

    # WL_DDP_SELFSPY=1 (run under sudo): rank-0 samples ITSELF with py-spy across the
    # steady-state loop -> folded stacks. Aggregate with ownership rules to get the
    # exact % of wall time the instruction pointer is in WL SDK code (vs model/loss/
    # decode/torch) — no A/B attribution guessing.
    if rank == 0 and os.environ.get("WL_DDP_SELFSPY") == "1":
        import subprocess
        try:
            subprocess.Popen(
                [os.environ.get("WL_DDP_SELFSPY_BIN", "py-spy"), "record",
                 "--pid", str(os.getpid()), "--rate", "200",
                 "--duration", os.environ.get("WL_DDP_SELFSPY_DUR", "120"),
                 "--format", "raw", "-o", os.environ.get("WL_DDP_SELFSPY_OUT", "/tmp/wl_ablation.folded")],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    for step in range(_WARMUP + _STEPS):
        timed = step >= _WARMUP
        if step == _WARMUP:
            io0 = _proc_io() # I/O counters at the start of the timed window
        t0 = time.perf_counter()
        inputs = next(batches)
        if is_wl:
            image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']
        else:
            image, _uids, _lbl, meta = inputs
            image, batch = image.float(), meta['batch']
        image = image.to(device)
        _sync(device); t1 = time.perf_counter()

        optimizer.zero_grad()
        raw = model(image)
        _sync(device); t2 = time.perf_counter()

        if is_wl:
            # decode (NMS) is USECASE compute — it produces predictions; WL only stores
            # them. Timed separately so it's NOT charged to the WL SDK tax.
            preds = decode(raw, image, conf=0.1, cls_thresh=0.1, device=device)
            _sync(device); t_dec = time.perf_counter()
            per = (crit["bbxs"](raw, batch, batch_ids=batch_ids, preds={'bboxes': preds})
                   + crit["clsf"](raw, batch, batch_ids=batch_ids)
                   + crit["dfl"](raw, batch, batch_ids=batch_ids))
            iou(raw, batch, batch_ids=batch_ids)
        else:
            # ulmanual: the irreducible cost of the GOAL done by hand — decode preds,
            # compute per-sample loss/IoU, write the scalars to a plain list (one CPU
            # transfer). Same decode + loss as wl, so they cancel in (wl - ulmanual).
            preds = decode(raw, image, conf=0.1, cls_thresh=0.1, device=device)
            _sync(device); t_dec = time.perf_counter()
            per = crit["bbxs"](raw, batch) + crit["clsf"](raw, batch) + crit["dfl"](raw, batch)
            iou(raw, batch)
            vals = per.detach().cpu().tolist()
            _manual_store.extend((step, i, v) for i, v in enumerate(vals))
        _sync(device); t3 = time.perf_counter()

        loss = per.mean()
        loss.backward()
        _sync(device); t4 = time.perf_counter()

        params = [p for p in model.parameters() if p.requires_grad]
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)
        grads = [p.grad for p in params]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat /= world
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)
        if timed and not grad_bytes:
            grad_bytes = flat.numel() * flat.element_size()
        _sync(device); t5 = time.perf_counter()

        if is_wl:
            reconcile_all(); flush_outbox()
        _sync(device); t6 = time.perf_counter()

        optimizer.step()
        _sync(device); t7 = time.perf_counter()

        if timed:
            t.add("data", t1 - t0); t.add("forward", t2 - t1)
            t.add("decode(usecase)", t_dec - t2)
            t.add("criterions+save", t3 - t_dec); t.add("backward", t4 - t3)
            t.add("grad_allreduce", t5 - t4); t.add("anchor(WL)", t6 - t5)
            t.add("optimizer", t7 - t6)

    io1 = _proc_io()
    io_d = {k: io1[k] - io0[k] for k in io1} if io0 else {k: 0 for k in io1}
    rss = _rss_mb()
    df_mb = h5_mb = -1.0
    if is_wl:
        try:
            from weightslab.backend.ledgers import get_dataframe
            df = get_dataframe().get_combined_df()
            df_mb = float(df.memory_usage(deep=True).sum()) / 1e6
        except Exception:
            pass
        try:
            ddir = os.path.join(_HERE, "ddp_run", "checkpoints", "data")
            h5_mb = sum(os.path.getsize(os.path.join(ddir, f))
                        for f in os.listdir(ddir) if f.endswith(".h5")) / 1e6
        except Exception:
            pass

    order = ["data", "forward", "decode(usecase)", "criterions+save", "backward",
             "grad_allreduce", "anchor(WL)", "optimizer"]
    # Each rank prints its OWN per-rank line (no gather collective — it was flaky on
    # gloo+CUDA; per-process I/O reads are independent anyway).
    io = io_d
    print(f"[mode={MODE} rank {rank}] RSS={rss:7.0f}MB anchor={t.ms('anchor(WL)'):6.1f}ms "
          f"IO(MB): rchar={io.get('rchar',0)/1e6:7.1f} wchar={io.get('wchar',0)/1e6:7.1f} "
          f"read_dsk={io.get('read_bytes',0)/1e6:6.1f} write_dsk={io.get('write_bytes',0)/1e6:6.1f}",
          flush=True)
    if rank == 0:
        total = sum(t.ms(k) for k in order)
        print("\n" + "=" * 74)
        print(f"ABLATION mode={MODE} device={device} world={world} batch={batch_size} steps={_STEPS}")
        print("=" * 74)
        for k in order:
            print(f" {k:18s} {t.ms(k):8.1f} ms/step")
        print(f" {'STEP TOTAL':18s} {total:8.1f} ms/step")
        print(f" {'grad on the wire':18s} {grad_bytes/1e6:8.1f} MB/step")
        if is_wl:
            print(f" WL df RAM {df_mb:.1f} MB | WL H5 {h5_mb:.1f} MB disk | "
                  f"H5 cfg: persist={cfg.get('ledger_enable_h5_persistence')} "
                  f"max_rows={cfg.get('ledger_flush_max_rows')} "
                  f"interval={cfg.get('ledger_flush_interval')}s "
                  f"threads={cfg.get('ledger_enable_flushing_threads')}")
        print("=" * 74, flush=True)

    dist.barrier(); dist.destroy_process_group()


def main():
    with socket.socket() as s:
        s.bind((_HOST, 0)); port = s.getsockname()[1]
    mp.spawn(_worker, args=(2, port), nprocs=2, join=True)


if __name__ == "__main__":
    main()
