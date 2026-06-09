"""Per-sample signal semantics + deny-aware sampler test (gRPC-only).

Stronger version of the earlier discard-leak probe: after waiting ~1 epoch,
discard the **top-N highest train-box-loss** samples from train_loader and
the **top-N highest val-IoU** samples from val_loader (proxies for "we just
got a per-sample read"). After another epoch's worth of steps, check that:

  1. `last_seen` did NOT advance for any victim on either split.
  2. Per-sample SIGNAL VALUES (train box/cls/dfl, val iou) did NOT change.

(1) verifies the sampler skipped the victims; (2) verifies WL never
overwrote those rows behind the sampler's back. Both must hold for the
"discard = no further reads/writes" invariant.
"""
import sys
import time

import grpc
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.proto import experiment_service_pb2_grpc as pb2_grpc

GRPC_ADDR = "localhost:50051"
ORIGIN_TRAIN = "train_loader"
ORIGIN_VAL = "val_loader"
N_DISCARD = 15
EPOCHS_TO_WAIT = 1     # 1 full epoch post-discard proves sampler skipped victims
POLL_INTERVAL = 5

# Train epoch length derived at runtime from dataset size + batch size by
# inferring batch size from observed step-rate vs sample count. See
# `wait_target()` below.

# gRPC serves signal columns with literal slashes (H5 file uses __SLASH__,
# but the data_service layer un-flattens before returning).
SIG_BOX = "signals//train/box_per_sample"
SIG_CLS = "signals//train/cls_per_sample"
SIG_DFL = "signals//train/dfl_per_sample"
SIG_IOU = "signals//val/iou_per_sample"
TRAIN_SIGNALS = [SIG_BOX, SIG_CLS, SIG_DFL]
VAL_SIGNALS = [SIG_IOU]


def _get_stat(rec, name):
    for s in rec.data_stats:
        if s.name == name:
            if s.value:
                return s.value[0]
            if s.value_string:
                return s.value_string
    return None


def _as_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Origin per sample is cached on first fetch and reused — the server builds
# a heavy response (image-path strings, bbox JSON, etc.) for unfiltered
# queries, and it OOMs the trainer if we repeat it every poll.
_origin_cache: dict = {}


def fetch_records(stub, origin, signals):
    # First call: no filter, capture origin per sample for future use.
    if not _origin_cache:
        req = pb2.DataSamplesRequest(start_index=0, records_cnt=5000)
        r = stub.GetDataSamples(req, timeout=30.0)
        for rec in r.data_records:
            o = _get_stat(rec, "origin")
            if o is not None:
                _origin_cache[rec.sample_id] = o
        # Use this first response for the immediate result too.
        return {sid: rec for rec in r.data_records
                if (sid := rec.sample_id) and _origin_cache.get(sid) == origin}
    # Subsequent: ask only for the lightweight columns we need.
    req = pb2.DataSamplesRequest(
        start_index=0, records_cnt=5000,
        stats_to_retrieve=["last_seen", "discarded"] + signals,
    )
    r = stub.GetDataSamples(req, timeout=30.0)
    return {rec.sample_id: rec for rec in r.data_records
            if _origin_cache.get(rec.sample_id) == origin}


def snapshot_sample(rec, signals):
    """Capture (last_seen, {signal: value}) for one record."""
    ls = _as_float(_get_stat(rec, "last_seen"))
    sigs = {s: _as_float(_get_stat(rec, s)) for s in signals}
    return ls, sigs


def max_last_seen(records):
    m = 0
    for rec in records.values():
        v = _as_float(_get_stat(rec, "last_seen"))
        if v is not None and v > m:
            m = int(v)
    return m


def pick_top_by(records, signals, sort_key, n):
    """Return n sample_ids with the highest value at sort_key. Requires only
    sort_key to be present (other signals can be None — we'll skip them
    in the verify step instead of pre-filtering here)."""
    rows = []
    for sid, rec in records.items():
        v = _as_float(_get_stat(rec, sort_key))
        if v is None:
            continue
        rows.append((sid, v))
    rows.sort(key=lambda r: r[1], reverse=True)
    return [sid for sid, _ in rows[:n]]


def verify_frozen(label, victims, pre, post, signals):
    """Check last_seen + signal values unchanged. Returns (leaks, frozen)."""
    leaks_ls, leaks_sig, frozen = [], [], []
    for v in victims:
        if v not in pre or v not in post:
            continue
        pre_ls, pre_sigs = pre[v]
        post_ls, post_sigs = post[v]
        if pre_ls is None or post_ls is None:
            continue
        # last_seen check
        if post_ls > pre_ls:
            leaks_ls.append((v, pre_ls, post_ls))
            continue
        # signal-value check. NaN != NaN in float semantics, so treat
        # (NaN, NaN) as unchanged — it means the signal hasn't fired
        # for this sample either before or after, not that it changed.
        import math
        changed = []
        for s in signals:
            a, b = pre_sigs[s], post_sigs[s]
            if a is None or b is None:
                continue
            both_nan = (
                isinstance(a, float) and math.isnan(a)
                and isinstance(b, float) and math.isnan(b)
            )
            if a != b and not both_nan:
                changed.append((s, a, b))
        if changed:
            leaks_sig.append((v, changed))
        else:
            frozen.append(v)

    print(f"\n[{label}] ─── verdict ──────────────────────────────")
    if leaks_ls:
        print(f"[{label}] FAIL — last_seen advanced on {len(leaks_ls)}/{len(victims)} victims:")
        for v, b, a in leaks_ls[:5]:
            print(f"  id={v} last_seen {b} → {a}")
    if leaks_sig:
        print(f"[{label}] FAIL — signal value changed on {len(leaks_sig)}/{len(victims)} victims:")
        for v, ch in leaks_sig[:5]:
            print(f"  id={v}: {ch}")
    if not leaks_ls and not leaks_sig:
        print(f"[{label}] PASS — {len(frozen)}/{len(victims)} frozen on both last_seen AND all signals.")
    return leaks_ls, leaks_sig


def main():
    chan = grpc.insecure_channel(
        GRPC_ADDR,
        options=[
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ],
    )
    stub = pb2_grpc.ExperimentServiceStub(chan)

    print(f"[client] waiting for trainer ramp-up (need val cycle + per-sample signals)…",
          flush=True)
    while True:
        try:
            train_recs = fetch_records(stub, ORIGIN_TRAIN, TRAIN_SIGNALS)
            val_recs = fetch_records(stub, ORIGIN_VAL, VAL_SIGNALS)
        except grpc.RpcError as e:
            print(f"[client] gRPC not ready: {e.code()}", flush=True)
            time.sleep(POLL_INTERVAL); continue
        max_train = max_last_seen(train_recs)
        train_with_box = sum(1 for r in train_recs.values()
                             if _as_float(_get_stat(r, SIG_BOX)) is not None)
        val_with_iou = sum(1 for r in val_recs.values()
                           if _as_float(_get_stat(r, SIG_IOU)) is not None)
        print(f"[client] max_train_last_seen={max_train} "
              f"train_with_box_sig={train_with_box} val_with_iou_sig={val_with_iou}",
              flush=True)
        # Ramp-up: enough samples carry the per-sample signals we need to
        # discard against. No hardcoded step count — scales with batch size.
        if (train_with_box >= N_DISCARD * 3
                and val_with_iou >= N_DISCARD * 3):
            break
        time.sleep(POLL_INTERVAL)

    # ── pick top-N for each split by the requested metric ───────────
    train_victims = pick_top_by(train_recs, TRAIN_SIGNALS, SIG_BOX, N_DISCARD)
    val_victims = pick_top_by(val_recs, VAL_SIGNALS, SIG_IOU, N_DISCARD)
    print(f"[client] train victims (top-{N_DISCARD} by box_loss): {train_victims[:5]}…")
    print(f"[client] val victims   (top-{N_DISCARD} by iou):      {val_victims[:5]}…")

    pre_max = max_last_seen(train_recs)

    # ── discard (auto-pauses the trainer) ───────────────────────────
    def discard(samples, origin):
        if not samples:
            return
        req = pb2.DataEditsRequest(
            stat_name="discarded", bool_value=True,
            type=pb2.SampleEditType.EDIT_OVERRIDE,
            samples_ids=samples,
            sample_origins=[origin] * len(samples),
        )
        resp = stub.EditDataSample(req, timeout=30.0)
        print(f"[client] discard {origin} ({len(samples)}): "
              f"success={resp.success} msg='{resp.message}'", flush=True)

    discard(train_victims, ORIGIN_TRAIN)
    discard(val_victims, ORIGIN_VAL)

    # ── pre-snapshot AFTER discard (trainer is paused) ──────────────
    # Snapshotting BEFORE discard would race: the trainer keeps stepping
    # for the 1-2 in-flight batches between our snapshot and the discard
    # taking effect, and those writes look like a signal-value "leak"
    # even though the sample was correctly skipped from that point on.
    train_recs = fetch_records(stub, ORIGIN_TRAIN, TRAIN_SIGNALS)
    val_recs = fetch_records(stub, ORIGIN_VAL, VAL_SIGNALS)
    pre_train = {v: snapshot_sample(train_recs[v], TRAIN_SIGNALS)
                 for v in train_victims if v in train_recs}
    pre_val = {v: snapshot_sample(val_recs[v], VAL_SIGNALS)
               for v in val_victims if v in val_recs}

    # ── resume ──────────────────────────────────────────────────────
    cmd = pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(
            hyper_parameters=pb2.HyperParameters(is_training=True)
        )
    )
    stub.ExperimentCommand(cmd, timeout=10.0)
    print("[client] resumed training", flush=True)

    # ── wait for EPOCHS_TO_WAIT full epochs over non-victims ──
    # 1 epoch = "every non-victim has been re-ticked at least once since
    # the previous checkpoint". Track per-sample advancement to handle
    # samplers that don't reach every sample evenly.
    victims_set = set(train_victims)
    non_victim_train = [sid for sid in train_recs if sid not in victims_set]
    ls_at_discard = {
        sid: int(_as_float(_get_stat(train_recs[sid], "last_seen")) or 0)
        for sid in non_victim_train
    }
    print(f"[client] waiting {EPOCHS_TO_WAIT} epochs over "
          f"{len(non_victim_train)} non-victim train samples…", flush=True)
    for epoch_i in range(EPOCHS_TO_WAIT):
        while True:
            try:
                train_recs = fetch_records(stub, ORIGIN_TRAIN, TRAIN_SIGNALS)
                val_recs = fetch_records(stub, ORIGIN_VAL, VAL_SIGNALS)
            except grpc.RpcError:
                time.sleep(POLL_INTERVAL); continue
            advanced = sum(
                1 for sid in non_victim_train
                if sid in train_recs
                and (_as_float(_get_stat(train_recs[sid], "last_seen")) or 0) > ls_at_discard[sid]
            )
            pct = 100 * advanced / max(1, len(non_victim_train))
            print(f"[client]   epoch {epoch_i+1}/{EPOCHS_TO_WAIT}: "
                  f"non-victims re-ticked: {advanced}/{len(non_victim_train)} ({pct:.0f}%)",
                  flush=True)
            # 95% is enough — the long tail can come from samples the sampler
            # naturally hasn't reached yet in this batch ordering.
            if advanced >= 0.95 * len(non_victim_train):
                # Reset baseline for next epoch
                ls_at_discard = {
                    sid: int(_as_float(_get_stat(train_recs[sid], "last_seen")) or 0)
                    for sid in non_victim_train if sid in train_recs
                }
                break
            time.sleep(POLL_INTERVAL)

    # ── verify ──────────────────────────────────────────────────────
    post_train = {v: snapshot_sample(train_recs[v], TRAIN_SIGNALS)
                  for v in train_victims if v in train_recs}
    post_val = {v: snapshot_sample(val_recs[v], VAL_SIGNALS)
                for v in val_victims if v in val_recs}

    leaks_tl, leaks_ts = verify_frozen("train", train_victims, pre_train, post_train, TRAIN_SIGNALS)
    leaks_vl, leaks_vs = verify_frozen("val",   val_victims,   pre_val,   post_val,   VAL_SIGNALS)

    chan.close()
    if leaks_tl or leaks_ts or leaks_vl or leaks_vs:
        sys.exit(1)


if __name__ == "__main__":
    main()
