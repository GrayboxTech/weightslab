"""Collective-resilience test: a throwing user callback must NOT hang the group.

Spawns 2 gloo/CPU ranks and registers, alongside a healthy state/outbox, a
"bad" consistent state whose snapshot() raises on rank 0 and whose apply() raises
on children, plus a "bad" outbox whose dump() raises everywhere and whose merge()
raises on rank 0. Then it runs two full anchor rounds (reconcile_all +
flush_outbox) and a final barrier.

If snapshot/apply/dump/merge were unguarded, the first round would crash rank 0
BEFORE the broadcast (or a child before the gather) and every other rank would
block forever on that collective — the spawn would hang and the outer `timeout`
would kill it. Passing (reaching the barrier + agreeing the healthy state synced)
proves every collective is still reached despite the failing callbacks.

    python -m weightslab.tests.test_ddp_collective_resilience
"""
import os
os.environ.setdefault("WL_DDP_LOG", "0")
os.environ.setdefault("WEIGHTSLAB_SKIP_SECURE_INIT", "true")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "ERROR")
os.environ.setdefault("WEIGHTSLAB_LOG_TO_FILE", "false")

import socket
import torch.distributed as dist
import torch.multiprocessing as mp

from weightslab.components.parallel_primitives import (
    register_consistent_state, register_outbox, reconcile_all, flush_outbox,
    clear_registry, reset_collectives,
)

_WORLD = 2


def _worker(rank, world, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    clear_registry()

    good = {"v": 0}
    register_consistent_state("good", lambda: {"v": good["v"]},
                              lambda s: good.update(v=s["v"]))

    def bad_snap():
        raise RuntimeError("snapshot boom")       # raises on rank 0

    def bad_apply(_s):
        raise RuntimeError("apply boom")          # raises on children

    register_consistent_state("bad", bad_snap, bad_apply)

    def bad_dump():
        raise RuntimeError("dump boom")           # raises on every rank

    def bad_merge(_parts):
        raise RuntimeError("merge boom")          # raises on rank 0

    register_outbox("bad_out", bad_dump, bad_merge)
    register_outbox("good_out", lambda: {"r": rank}, lambda parts: None)

    if rank == 0:
        good["v"] = 42

    # Two full anchor rounds — the bad callbacks fire each round but must be
    # swallowed so the collectives still happen in lockstep.
    for _ in range(2):
        reset_collectives()
        reconcile_all()
        flush_outbox()

    ok_good = (good["v"] == 42)                    # healthy state converged anyway
    dist.barrier()                                 # ultimate no-hang proof
    gathered = [None] * world
    dist.all_gather_object(gathered, ok_good)
    dist.destroy_process_group()
    if rank == 0:
        allok = all(gathered)
        print(f"RESULT: {'PASS' if allok else 'FAIL'} "
              f"(healthy state synced on all ranks={gathered})", flush=True)
        raise SystemExit(0 if allok else 1)


def main():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    print(f"[suite] spawning {_WORLD} gloo/CPU ranks for collective-resilience test",
          flush=True)
    mp.spawn(_worker, args=(_WORLD, port), nprocs=_WORLD, join=True)


if __name__ == "__main__":
    main()
