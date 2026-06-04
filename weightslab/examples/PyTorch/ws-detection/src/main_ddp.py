"""DDP spawn shim for YOLO detection — train_worker body is the single-process
loop, unchanged. All DDP plumbing (anchor, pause-spin, consistent-state reconcile)
lives inside the SDK now — `guard_training_context.__enter__` is the one
DDP-aware site, and the core states (hparams / deny-list / paused) auto-register
on its first call.

Run:
    python main_ddp.py                # WL_DDP_WORLD_SIZE=2 (default)
    WL_DDP_LOG=1 python main_ddp.py   # rank-prefixed per-step DDP trace

Or import `train_worker` from the test suite to drive it via gRPC.
"""
import os
os.environ.setdefault("WEIGHTSLAB_SKIP_SECURE_INIT", "true")
os.environ.setdefault("GRPC_TLS_ENABLED", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")
os.environ.setdefault("WEIGHTSLAB_LOG_TO_FILE", "false")
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WL_DDP_IMGSZ", "96")
os.environ.setdefault("WEIGHTSLAB_DISABLE_WATCHDOGS", "1")

import socket
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yolo_pipeline  # reuse _build_pipeline / _decode_preds_to_6col / _HERE

_HOST = "127.0.0.1"
_WORLD = int(os.environ.get("WL_DDP_WORLD_SIZE", "2"))


def train_worker(rank, world, master_port, grpc_port):
    """SPMD worker. Body is identical to a single-process trainer — the DDP
    plumbing (reconcile, pause-spin, deny-list/hparam sync) lives inside the
    guard's __enter__ and the SDK touchpoints. Only differences from main.py:
      * dist.init_process_group at the top.
      * broadcast the freshly-built model so every replica starts identical.
      * a manual grad all-reduce in the loop body (until DDP wraps the model).
    """
    import time as _time
    _st = _time.perf_counter()
    _stime = os.environ.get("WL_DDP_START_TIMING") == "1"
    def _lap(label):
        nonlocal _st
        if _stime and rank == 0:
            now = _time.perf_counter()
            print(f"[start_timing] {label:32s} {1000*(now-_st):8.0f} ms", flush=True)
            _st = now

    os.environ["MASTER_ADDR"] = _HOST
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["GRPC_BACKEND_PORT"] = str(grpc_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    _lap("dist.init_process_group")

    import weightslab as wl
    from weightslab.utils.tools import seed_everything
    seed_everything(1234)
    _lap("import weightslab + seed")
    # Multi-rank on ONE GPU: keep gloo for the collectives (works multi-rank on a
    # single host; gloo all_reduce/broadcast on CUDA tensors is supported and
    # host-staged), but run the forward/backward on the shared GPU so the heavy
    # compute is fast. Both ranks share cuda:0 (time-sliced; use CUDA MPS for real
    # overlap). Opt in with WL_DDP_CUDA=1; default stays CPU so nothing changes
    # unless asked. NCCL is intentionally NOT used — 2 nccl ranks on 1 device hang.
    _use_cuda = torch.cuda.is_available() and os.environ.get("WL_DDP_CUDA", "0") == "1"
    if _use_cuda:
        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    cfg = yaml.safe_load(open(os.path.join(yolo_pipeline._HERE, "config.yaml")))
    cfg["compute_natural_sort"] = False
    cfg["data"]["train_loader"]["shuffle"] = True

    _lap("cfg load")
    if rank != 0:
        dist.barrier()
    trainer, train_loader, criterions, _my_uids, _all_uids = \
        yolo_pipeline._build_pipeline(cfg, device, rank, world)
    if rank == 0:
        dist.barrier()
    _lap("build_pipeline (+barriers)")

    model, optimizer = trainer.model, trainer.optimizer
    # Flatten the initial weight sync into ONE broadcast (was one per parameter
    # tensor — ~256 gloo collectives, ~11s startup; flattened → ~1s). Same
    # per-tensor-collective antipattern as the grad reduce.
    with torch.no_grad():
        params = list(model.parameters())
        flat = torch._utils._flatten_dense_tensors([p.data for p in params])
        dist.broadcast(flat, src=0)
        for p, s in zip(params, torch._utils._unflatten_dense_tensors(flat, [p.data for p in params])):
            p.data.copy_(s)
    _lap("model broadcast (flattened, 1 collective)")

    wl.serve(serving_grpc=True, serving_cli=False)
    _lap("wl.serve (gRPC up)")

    cs = criterions
    miou = trainer.iou   # PerSampleIoU criterion, registered with flag="metric"

    def _infinite(loader):
        while True:
            yield from loader
    batches = _infinite(train_loader)

    # Uneven inputs are handled at the sampler (equal-length rebalanced shards ->
    # matched batch counts), so the grad all_reduce below can't deadlock on a
    # discard-starved rank. See dataloader_interface._ddp_rebalanced_shard.
    _first_step = True
    while True:
        with wl.guard_training_context:
            if _first_step:
                _lap("serve->first step body (client wait_ready+train_steps+unpause)")
                _first_step = False
            optimizer.zero_grad()
            inputs = next(batches)
            image, batch_ids, batch = inputs[0].float(), inputs[1], inputs[3]['batch']
            raw = model(image.to(device))
            preds = yolo_pipeline._decode_preds_to_6col(
                raw, image, conf=0.1, cls_thresh=0.1, device=device)
            per_sample = (
                cs["bbxs"](raw, batch, batch_ids=batch_ids, preds={'bboxes': preds})
                + cs["clsf"](raw, batch, batch_ids=batch_ids)
                + cs["dfl"](raw, batch, batch_ids=batch_ids)
            )
            # Per-sample IoU metric — populates miou/train signals.
            miou(raw, batch, batch_ids=batch_ids)
            loss = per_sample.mean()
            loss.backward()

            # Bucketed grad sync: ONE all_reduce over the FLATTENED grads (vs one
            # collective per parameter tensor — ~256 for yolo11s; perf ~480→65ms).
            # Zero-fill missing grads first so the flattened buffer has identical
            # size/order on every rank (data-independent) — else gloo desyncs.
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
            optimizer.step()


def main():
    with socket.socket() as s:
        s.bind((_HOST, 0))
        master_port = s.getsockname()[1]
    with socket.socket() as s:
        s.bind((_HOST, 0))
        grpc_port = s.getsockname()[1]
    print(f"[main_ddp] spawning {_WORLD} ranks "
          f"(master={master_port}, grpc={grpc_port}, rank-0 serves)", flush=True)
    mp.spawn(train_worker, args=(_WORLD, master_port, grpc_port), nprocs=_WORLD, join=True)


if __name__ == "__main__":
    main()
