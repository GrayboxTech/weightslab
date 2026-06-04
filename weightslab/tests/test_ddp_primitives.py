"""Trivial, YOLO-free validation of the DDP base primitives (batch 0).

Spawns N gloo/CPU ranks, registers a fake consistent state, and checks that
`reconcile_all` makes children converge to rank 0's snapshot — initial sync,
idempotency, and change-propagation — while accounting collectives. Pure
torch.distributed + the building blocks; no model, no dataset, runs in seconds.

    python -m weightslab.tests.test_ddp_primitives        # 3 ranks
    WL_DDP_WORLD_SIZE=2 python -m weightslab.tests.test_ddp_primitives
"""
import os
os.environ.setdefault("WL_DDP_LOG", "1")                    # show the per-rank trace
os.environ.setdefault("WEIGHTSLAB_SKIP_SECURE_INIT", "true")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "ERROR")
os.environ.setdefault("WEIGHTSLAB_LOG_TO_FILE", "false")

import socket
import torch.distributed as dist
import torch.multiprocessing as mp

from weightslab.components.parallel_primitives import (
    register_consistent_state, reconcile_all, clear_registry,
    reset_collectives, collective_count,
)

_WORLD = int(os.environ.get("WL_DDP_WORLD_SIZE", "3"))


def _worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)

    # A fake "consistent state": a local mutable dict. rank 0 is the authority;
    # children start with stale/divergent values and must converge via reconcile.
    knob = {"lr": 0.0, "discarded": set()}

    def snapshot():                       # rank 0 only
        return {"lr": knob["lr"], "discarded": set(knob["discarded"])}

    def apply(state):                     # children
        knob["lr"] = state["lr"]
        knob["discarded"] = set(state["discarded"])

    clear_registry()
    register_consistent_state("knob", snapshot, apply)

    checks = []

    def check(label, exp_lr, exp_disc):
        # gather every rank's knob to rank 0 and assert they all match rank 0's target
        mine = {"lr": knob["lr"], "discarded": sorted(knob["discarded"])}
        gathered = [None] * world
        dist.all_gather_object(gathered, mine)
        if rank == 0:
            lrs = {g["lr"] for g in gathered}
            discs = {tuple(g["discarded"]) for g in gathered}
            ok = (lrs == {exp_lr}) and (discs == {tuple(sorted(exp_disc))})
            checks.append(ok)
            print(f"[check {label}] consistent={ok}  lrs={lrs} discs={discs}", flush=True)

    # rank 0 sets authoritative state; children start divergent
    if rank == 0:
        knob.update(lr=0.05, discarded={1, 4})

    reset_collectives()
    reconcile_all()                       # children converge to rank 0
    check("initial-sync", 0.05, {1, 4})

    reconcile_all()                       # idempotent: same state, no drift
    check("idempotent", 0.05, {1, 4})

    if rank == 0:                         # async-style edit on rank 0
        knob.update(lr=0.5, discarded={1, 4, 9})
    reconcile_all()                       # change propagates
    check("after-change", 0.5, {1, 4, 9})

    if rank == 0:
        allok = bool(checks) and all(checks)
        print(f"\nRESULT: {'ALL PASS' if allok else 'FAIL'}   "
              f"(reconcile collectives = {collective_count()}, expect {3})", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        raise SystemExit(0 if (bool(checks) and all(checks)) else 1)


def main():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    print(f"[suite] spawning {_WORLD} gloo/CPU ranks for primitive test", flush=True)
    mp.spawn(_worker, args=(_WORLD, port), nprocs=_WORLD, join=True)


if __name__ == "__main__":
    main()
