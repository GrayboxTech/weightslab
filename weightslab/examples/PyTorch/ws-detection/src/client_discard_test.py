"""Scenario test for the deny-aware sampler fix — gRPC-only (no H5 reads).

Avoids H5 lag entirely. Uses ApplyDataQuery for trainer state polling
(monotonic, in-memory) and GetDataSamples for per-victim last_seen probes.

Workflow:
  1. Connect via plaintext gRPC (requires GRPC_TLS_ENABLED=0 on the trainer).
  2. Poll ApplyDataQuery until trainer is past ~1 epoch.
  3. GetDataSamples → pick N victims with highest last_seen.
  4. Snapshot their last_seen.
  5. EditDataSample → mark them discarded (this pauses training server-side).
  6. ExperimentCommand(is_training=True) → resume.
  7. Poll until trainer has trained ≥ TRAIN_STEPS_TO_WAIT post-discard.
  8. GetDataSamples → re-read victim last_seen.
  9. Compare: any advanced ⇒ leak.
"""
import sys
import time

import grpc
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.proto import experiment_service_pb2_grpc as pb2_grpc

GRPC_ADDR = "localhost:50051"
ORIGIN_TRAIN = "train_loader"
N_DISCARD = 30
TRAIN_STEPS_TO_WAIT = 300     # > 1 full epoch (237 steps) post-discard
POLL_INTERVAL = 5


def _get_stat(rec, name):
    for s in rec.data_stats:
        if s.name == name:
            if s.value:
                return s.value[0]
            if s.value_string:
                return s.value_string
    return None


def fetch_records(stub):
    req = pb2.DataSamplesRequest(
        start_index=0, records_cnt=5000,
        stats_to_retrieve=["last_seen", "discarded"],
    )
    r = stub.GetDataSamples(req, timeout=30.0)
    return {rec.sample_id: rec for rec in r.data_records}


def max_last_seen(records):
    m = 0
    for rec in records.values():
        ls = _get_stat(rec, "last_seen")
        if ls is None:
            continue
        try:
            v = int(float(ls))
            if v > m:
                m = v
        except (TypeError, ValueError):
            pass
    return m


def main():
    chan = grpc.insecure_channel(GRPC_ADDR)
    stub = pb2_grpc.ExperimentServiceStub(chan)

    print(f"[client] waiting for trainer to ramp up…", flush=True)
    # Wait until ~enough samples have been ticked (epoch 1 done-ish)
    while True:
        try:
            recs = fetch_records(stub)
        except grpc.RpcError as e:
            print(f"[client] gRPC not ready: {e.code()}", flush=True)
            time.sleep(POLL_INTERVAL); continue
        cur = max_last_seen(recs)
        ticked = sum(1 for rec in recs.values()
                     if (ls := _get_stat(rec, "last_seen")) is not None
                     and ls not in ("-1", -1) and int(float(ls)) > 0)
        print(f"[client] ticked={ticked} max_last_seen={cur}", flush=True)
        if cur >= 200 and ticked >= N_DISCARD * 3:
            break
        time.sleep(POLL_INTERVAL)

    # Pick victims (highest last_seen = currently in the active sampling rotation)
    pairs = []
    for sid, rec in recs.items():
        ls = _get_stat(rec, "last_seen")
        if ls is None: continue
        try:
            pairs.append((sid, int(float(ls))))
        except (TypeError, ValueError):
            pass
    pairs.sort(key=lambda x: x[1], reverse=True)
    victims = [p[0] for p in pairs[:N_DISCARD]]
    pre_snapshot = {p[0]: p[1] for p in pairs[:N_DISCARD]}
    snapshot_max = max(pre_snapshot.values())
    print(f"[client] picked {len(victims)} victims, last_seen range "
          f"[{min(pre_snapshot.values())}, {max(pre_snapshot.values())}]", flush=True)
    print(f"[client] sample victims: {victims[:5]} → "
          f"{[pre_snapshot[v] for v in victims[:5]]}", flush=True)

    # Discard
    req = pb2.DataEditsRequest(
        stat_name="discarded",
        bool_value=True,
        type=pb2.SampleEditType.EDIT_OVERRIDE,
        samples_ids=victims,
        sample_origins=[ORIGIN_TRAIN] * len(victims),
    )
    resp = stub.EditDataSample(req, timeout=30.0)
    print(f"[client] EditDataSample → success={resp.success} msg='{resp.message}'", flush=True)

    # Resume training (EditDataSample paused it)
    hp = pb2.HyperParameters(is_training=True)
    cmd = pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(hyper_parameters=hp)
    )
    rresp = stub.ExperimentCommand(cmd, timeout=10.0)
    print(f"[client] resumed training (success={rresp.success})", flush=True)

    # Wait for ≥ TRAIN_STEPS_TO_WAIT post-discard steps via gRPC polling
    target = snapshot_max + TRAIN_STEPS_TO_WAIT
    print(f"[client] waiting for max_last_seen ≥ {target}…", flush=True)
    while True:
        try:
            recs = fetch_records(stub)
        except grpc.RpcError as e:
            print(f"[client] gRPC err: {e.code()}", flush=True)
            time.sleep(POLL_INTERVAL); continue
        cur = max_last_seen(recs)
        print(f"[client]   current max_last_seen={cur} (target {target})", flush=True)
        if cur >= target:
            break
        time.sleep(POLL_INTERVAL)

    # Verdict
    leaked = []; frozen = []
    for v in victims:
        rec = recs.get(v)
        if rec is None: continue
        ls_raw = _get_stat(rec, "last_seen")
        if ls_raw is None: continue
        try:
            ls = int(float(ls_raw))
        except (TypeError, ValueError):
            continue
        before = pre_snapshot[v]
        if ls > before:
            leaked.append((v, before, ls))
        else:
            frozen.append((v, before, ls))

    print("=" * 60, flush=True)
    if not leaked:
        print(f"[client] PASS — {len(frozen)}/{len(victims)} victims frozen, "
              f"0 leaks across ≥{TRAIN_STEPS_TO_WAIT} post-discard train steps.",
              flush=True)
    else:
        print(f"[client] FAIL — {len(leaked)}/{len(victims)} victims leaked.",
              flush=True)
        for v, b, a in leaked[:10]:
            print(f"  id={v} before={b} after={a} (+{a-b})", flush=True)
    print("=" * 60, flush=True)
    chan.close()


if __name__ == "__main__":
    main()
