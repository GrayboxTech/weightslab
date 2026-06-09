"""Tag-then-discard scenario test (gRPC-only).

Companion to `client_discard_test.py`. Same freeze-invariant, different
victim-selection path: instead of picking top-N by signal value, we

  1. set a custom `tag:probe` on a chosen subset of train + val samples,
  2. re-query the server to find every sample where `tag:probe == True`,
  3. discard the result.

(2) is the path the agent/LLM uses: tag → query-by-tag → bulk-edit. By
bypassing the LLM (we set the tags directly via gRPC) we exercise the
same code path without needing an API key or studio session.

Then we re-use the existing freeze invariant: after one epoch of
non-victim training, the discarded samples must not have advanced
`last_seen` nor changed signal values.
"""
import math
import sys
import time

import grpc
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.proto import experiment_service_pb2_grpc as pb2_grpc

GRPC_ADDR = "localhost:50051"
ORIGIN_TRAIN = "train_loader"
ORIGIN_VAL = "val_loader"
N_TAG_TRAIN = 15
N_TAG_VAL = 15
TAG_NAME = "probe"
# WL server flattens `tag:<name>` writes into a single `tag:` read column.
EPOCHS_TO_WAIT = 1
POLL_INTERVAL = 5

SIG_BOX = "signals//train/box_per_sample"
SIG_CLS = "signals//train/cls_per_sample"
SIG_DFL = "signals//train/dfl_per_sample"
SIG_IOU = "signals//val/iou_per_sample"
TRAIN_SIGNALS = [SIG_BOX, SIG_CLS, SIG_DFL]
VAL_SIGNALS = [SIG_IOU]
TAG_WRITE_COL = f"tag:{TAG_NAME}"   # what we send to EditDataSample
TAG_READ_COL = "tag:"               # what GetDataSamples surfaces on read


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


def _is_tag_set(rec):
    """`tag:<name>` columns return a string when present (e.g. "1" / "true"
    / the tag value itself). Treat any non-empty stat as set."""
    v = _get_stat(rec, TAG_READ_COL)
    if v is None:
        return False
    if isinstance(v, str):
        return v.lower() not in ("", "false", "0", "none", "nan")
    return bool(v)


_origin_cache: dict = {}


def fetch_records(stub, origin, signals, include_tag=False):
    """First call: full fetch (captures origin per sample). Subsequent:
    request only the columns we need so the server doesn't OOM building
    image-path / bbox-JSON strings for every sample every poll."""
    if not _origin_cache:
        req = pb2.DataSamplesRequest(start_index=0, records_cnt=5000)
        r = stub.GetDataSamples(req, timeout=120.0)
        for rec in r.data_records:
            o = _get_stat(rec, "origin")
            if o is not None:
                _origin_cache[rec.sample_id] = o
        return {sid: rec for rec in r.data_records
                if (sid := rec.sample_id) and _origin_cache.get(sid) == origin}
    cols = ["last_seen", "discarded"] + signals
    if include_tag:
        cols.append(TAG_READ_COL)
    req = pb2.DataSamplesRequest(
        start_index=0, records_cnt=5000, stats_to_retrieve=cols,
    )
    r = stub.GetDataSamples(req, timeout=30.0)
    return {rec.sample_id: rec for rec in r.data_records
            if _origin_cache.get(rec.sample_id) == origin}


def snapshot_sample(rec, signals):
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


def verify_frozen(label, victims, pre, post, signals):
    leaks_ls, leaks_sig, frozen = [], [], []
    for v in victims:
        if v not in pre or v not in post:
            continue
        pre_ls, pre_sigs = pre[v]
        post_ls, post_sigs = post[v]
        if pre_ls is None or post_ls is None:
            continue
        if post_ls > pre_ls:
            leaks_ls.append((v, pre_ls, post_ls))
            continue
        changed = []
        for s in signals:
            a, b = pre_sigs[s], post_sigs[s]
            if a is None or b is None:
                continue
            both_nan = (isinstance(a, float) and math.isnan(a)
                        and isinstance(b, float) and math.isnan(b))
            # tolerance covers gRPC float round-trip noise (~7-decimal precision).
            if not both_nan and abs(a - b) > 1e-5:
                changed.append((s, a, b))
        if changed:
            leaks_sig.append((v, changed))
        else:
            frozen.append(v)
    print(f"\n[{label}] ─── verdict ──────────────────────────────")
    if leaks_ls:
        print(f"[{label}] FAIL — last_seen advanced on {len(leaks_ls)}/{len(victims)}:")
        for v, b, a in leaks_ls[:5]:
            print(f"  id={v} last_seen {b} → {a}")
    if leaks_sig:
        print(f"[{label}] FAIL — signal value changed on {len(leaks_sig)}/{len(victims)}:")
        for v, ch in leaks_sig[:5]:
            print(f"  id={v}: {ch}")
    if not leaks_ls and not leaks_sig:
        print(f"[{label}] PASS — {len(frozen)}/{len(victims)} frozen on last_seen AND signals.")
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

    print("[client] waiting for trainer ramp-up (need val cycle + per-sample signals)…",
          flush=True)
    while True:
        try:
            train_recs = fetch_records(stub, ORIGIN_TRAIN, TRAIN_SIGNALS)
            val_recs = fetch_records(stub, ORIGIN_VAL, VAL_SIGNALS)
        except grpc.RpcError as e:
            print(f"[client] gRPC not ready: {e.code()}", flush=True)
            time.sleep(POLL_INTERVAL); continue
        train_with_box = sum(1 for r in train_recs.values()
                             if _as_float(_get_stat(r, SIG_BOX)) is not None)
        val_with_iou = sum(1 for r in val_recs.values()
                           if _as_float(_get_stat(r, SIG_IOU)) is not None)
        print(f"[client] train_with_box={train_with_box} val_with_iou={val_with_iou}",
              flush=True)
        if train_with_box >= N_TAG_TRAIN * 3 and val_with_iou >= N_TAG_VAL * 3:
            break
        time.sleep(POLL_INTERVAL)

    # ── pick N train + N val samples to tag (any with a populated signal) ──
    candidates_train = [sid for sid, rec in train_recs.items()
                        if _as_float(_get_stat(rec, SIG_BOX)) is not None][:N_TAG_TRAIN]
    candidates_val = [sid for sid, rec in val_recs.items()
                      if _as_float(_get_stat(rec, SIG_IOU)) is not None][:N_TAG_VAL]
    print(f"[client] tagging {len(candidates_train)} train + "
          f"{len(candidates_val)} val with tag:{TAG_NAME}…", flush=True)

    def set_tag(samples, origin):
        if not samples:
            return
        req = pb2.DataEditsRequest(
            stat_name=TAG_WRITE_COL, bool_value=True,
            type=pb2.SampleEditType.EDIT_OVERRIDE,
            samples_ids=samples,
            sample_origins=[origin] * len(samples),
        )
        resp = stub.EditDataSample(req, timeout=30.0)
        print(f"[client] tag {origin} ({len(samples)}): "
              f"success={resp.success} msg='{resp.message}'", flush=True)

    set_tag(candidates_train, ORIGIN_TRAIN)
    set_tag(candidates_val, ORIGIN_VAL)

    # ── re-query: who currently has tag:probe set? (the agent's path) ──
    train_recs = fetch_records(stub, ORIGIN_TRAIN, TRAIN_SIGNALS, include_tag=True)
    val_recs = fetch_records(stub, ORIGIN_VAL, VAL_SIGNALS, include_tag=True)
    train_victims = [sid for sid, rec in train_recs.items() if _is_tag_set(rec)]
    val_victims = [sid for sid, rec in val_recs.items() if _is_tag_set(rec)]
    print(f"[client] tag-query found train={len(train_victims)} val={len(val_victims)}",
          flush=True)

    if not train_victims or not val_victims:
        print(f"[client] FAIL — tag-query returned empty; "
              f"EditDataSample stat_name='{TAG_WRITE_COL}' write or "
              f"'{TAG_READ_COL}' read may be broken.", flush=True)
        sys.exit(2)

    # ── discard the tagged set (auto-pauses trainer) ───────────────
    def discard(samples, origin):
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

    # ── pre-snapshot AFTER discard (trainer paused, no race) ───────
    train_recs = fetch_records(stub, ORIGIN_TRAIN, TRAIN_SIGNALS)
    val_recs = fetch_records(stub, ORIGIN_VAL, VAL_SIGNALS)
    pre_train = {v: snapshot_sample(train_recs[v], TRAIN_SIGNALS)
                 for v in train_victims if v in train_recs}
    pre_val = {v: snapshot_sample(val_recs[v], VAL_SIGNALS)
               for v in val_victims if v in val_recs}

    # ── resume ─────────────────────────────────────────────────────
    cmd = pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(
            hyper_parameters=pb2.HyperParameters(is_training=True)
        )
    )
    stub.ExperimentCommand(cmd, timeout=10.0)
    print("[client] resumed training", flush=True)

    # ── wait for EPOCHS_TO_WAIT epochs over non-victims (95% re-ticked) ──
    victims_set = set(train_victims)
    non_victim_train = [sid for sid in train_recs if sid not in victims_set]
    ls_at_discard = {
        sid: int(_as_float(_get_stat(train_recs[sid], "last_seen")) or 0)
        for sid in non_victim_train
    }
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
            if advanced >= 0.95 * len(non_victim_train):
                ls_at_discard = {
                    sid: int(_as_float(_get_stat(train_recs[sid], "last_seen")) or 0)
                    for sid in non_victim_train if sid in train_recs
                }
                break
            time.sleep(POLL_INTERVAL)

    # ── verify freeze ──────────────────────────────────────────────
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
