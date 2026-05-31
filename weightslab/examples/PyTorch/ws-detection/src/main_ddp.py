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
    os.environ["MASTER_ADDR"] = _HOST
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["GRPC_BACKEND_PORT"] = str(grpc_port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    import weightslab as wl
    from weightslab.utils.tools import seed_everything
    seed_everything(1234)
    device = torch.device("cpu")

    cfg = yaml.safe_load(open(os.path.join(yolo_pipeline._HERE, "config.yaml")))
    cfg["compute_natural_sort"] = False
    cfg["data"]["train_loader"]["shuffle"] = True

    if rank != 0:
        dist.barrier()
    trainer, train_loader, criterions, _my_uids, _all_uids = \
        yolo_pipeline._build_pipeline(cfg, device, rank, world)
    if rank == 0:
        dist.barrier()

    model, optimizer = trainer.model, trainer.optimizer
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    wl.serve(serving_grpc=True, serving_cli=False)

    cs = criterions
    miou = trainer.iou   # PerSampleIoU criterion, registered with flag="metric"

    def _infinite(loader):
        while True:
            yield from loader
    batches = _infinite(train_loader)

    while True:
        with wl.guard_training_context:
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
            # Invoke the per-sample IoU metric — populates miou/train signals
            # (was missing from the DDP loop, causing the metric to be silently
            # absent from both per-sample and scalar logger paths).
            miou(raw, batch, batch_ids=batch_ids)
            loss = per_sample.mean()
            loss.backward()

            # Reduce EVERY trainable param's grad (zero-fill missing) so the
            # collective set + order is data-INdependent — else gloo desyncs.
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    p.grad = torch.zeros_like(p.data)
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
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
