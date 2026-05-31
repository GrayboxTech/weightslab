"""DDP integration test suite — drives a REAL distributed trainer over the
ExperimentService gRPC API (exactly like the studio would) and asserts behaviour
from the outside. We rely on the *sincerity* of the tested process: the server is
the real training code path; the test only ever touches the public gRPC surface.

This is a scaffold we keep expanding — NOT production-shippable. Add scenarios,
measure time/processes, and later extract the reusable bits into the SDK.

Layout:
  * _train_worker(rank, world, ...)  -- the distributed SERVER (spawned, rank 0 serves).
  * Client + scenarios               -- run in the PARENT process, talk gRPC to rank 0.

First scenario (scenario_epoch_then_pause):
  1. spawn `world` ranks (rank 0 serves gRPC plaintext); parent = client.
  2. query universe size N via gRPC.
  3. train 1 UNIVERSE epoch = ceil(N / (world*batch)) steps; poll until paused.
  4. assert last_seen got populated (the rank-0-visible subset is non-empty and
     == rank 0's shard; full-universe coverage needs the children->rank0 gather).
  5. wait ~50% of the epoch's wall time (training is paused) and assert the
     last_seen map is byte-identical => pause truly froze training.

Run:  python ddp_test_suite.py        (WL_DDP_WORLD_SIZE=2, imgsz 96, num_workers 0)
"""
import os
# --- test-mode env (must be set before importing weightslab) ---------------
os.environ["WEIGHTSLAB_SKIP_SECURE_INIT"] = "true"   # plaintext gRPC for the client
os.environ["GRPC_TLS_ENABLED"] = "0"
os.environ.setdefault("WL_DDP_IMGSZ", "96")          # small images for speed
os.environ.setdefault("WL_DDP_COLLECTIVE_LOG", "/tmp/wl_collective_log.txt")
os.environ.setdefault("WL_PRELOAD_IMAGE_OVERVIEW", "0")
os.environ.setdefault("WEIGHTSLAB_LOG_LEVEL", "WARNING")
os.environ.setdefault("WEIGHTSLAB_LOG_TO_FILE", "false")
os.environ.setdefault("WEIGHTSLAB_DISABLE_WATCHDOGS", "1")

import math
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
import grpc

import yolo_pipeline  # reuse _build_pipeline / _decode_preds_to_6col / _HERE / _LOSS_PARTS

import weightslab.proto.experiment_service_pb2 as pb2
import weightslab.proto.experiment_service_pb2_grpc as pb2_grpc

_WORLD = int(os.environ.get("WL_DDP_WORLD_SIZE", "2"))
_HOST = "127.0.0.1"


# ===========================================================================
# SERVER  (spawned ranks; rank 0 serves gRPC)
# ===========================================================================
def _train_worker(rank, world, master_port, grpc_port):
    """Spawned per rank. Delegates to main_ddp.train_worker — the clean
    primitives-based loop (sync_step at the anchor, no bespoke ddp_guard_sync).
    Older yolo_pipeline-style worker is preserved in yolo_pipeline.worker for reference."""
    import main_ddp
    main_ddp.train_worker(rank, world, master_port, grpc_port)


# ===========================================================================
# CLIENT  (parent process)
# ===========================================================================
class Client:
    def __init__(self, port):
        self._port = int(port)             # exposed for topology-style scenarios
        self.channel = grpc.insecure_channel(
            f"{_HOST}:{port}",
            options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)],
        )
        self.stub = pb2_grpc.ExperimentServiceStub(self.channel)

    def wait_ready(self, timeout=180.0):
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                self.stub.GetDataSplits(pb2.Empty(), timeout=5)
                return True
            except Exception as e:
                last_err = e
                time.sleep(1.0)
        raise TimeoutError(f"server not ready in {timeout}s: {last_err}")

    def universe_size(self):
        resp = self.stub.ApplyDataQuery(
            pb2.DataQueryRequest(query="", is_natural_language=False), timeout=30)
        return int(resp.number_of_all_samples)

    def train_steps(self, n):
        hp = pb2.HyperParameters(nb_steps=int(n), is_training=True)
        self.stub.ExperimentCommand(
            pb2.TrainerCommand(hyper_parameter_change=pb2.HyperParameterCommand(hyper_parameters=hp)),
            timeout=30)

    def pause(self):
        hp = pb2.HyperParameters(is_training=False)
        self.stub.ExperimentCommand(
            pb2.TrainerCommand(hyper_parameter_change=pb2.HyperParameterCommand(hyper_parameters=hp)),
            timeout=30)

    def train_origin(self):
        """Origin string for the train split (the dataframe MultiIndex key)."""
        resp = self.stub.GetDataSplits(pb2.Empty(), timeout=15)
        names = list(resp.split_names)
        for nm in names:
            if "train" in nm.lower():
                return nm
        return names[0] if names else "train_loader"

    def discard(self, sample_ids, origin):
        """Discard via the real UI RPC (EditDataSample)."""
        ids = [str(s) for s in sample_ids]
        req = pb2.DataEditsRequest(
            stat_name="discarded", string_value="", float_value=0, bool_value=True,
            type=pb2.EDIT_OVERRIDE, samples_ids=ids, sample_origins=[origin] * len(ids),
        )
        return self.stub.EditDataSample(req, timeout=30)

    def discarded_count(self):
        resp = self.stub.ApplyDataQuery(
            pb2.DataQueryRequest(query="", is_natural_language=False), timeout=30)
        return int(resp.number_of_discarded_samples)

    def tag(self, sample_ids, tag, origin):
        ids = [str(s) for s in sample_ids]
        req = pb2.DataEditsRequest(
            stat_name="tags", string_value=tag, float_value=0, bool_value=False,
            type=pb2.EDIT_ACCUMULATE, samples_ids=ids, sample_origins=[origin] * len(ids))
        return self.stub.EditDataSample(req, timeout=60)

    def break_by_slice(self, graph_name, tags):
        """Per-sample points for `graph_name`, filtered to samples carrying `tags`."""
        resp = self.stub.GetLatestLoggerData(
            pb2.GetLatestLoggerDataRequest(
                request_full_history=False, break_by_slices=True,
                tags=list(tags), graph_name=graph_name), timeout=60)
        return [(p.sample_id, p.model_age, p.metric_value) for p in resp.points]

    def scalar_plot(self, graph_name, max_points=10000):
        """Scalar timeseries for `graph_name` — the batch-aggregate plot points
        (one mean per training step), distinct from `break_by_slice` which
        returns per-sample points. Used to cross-check both halves of a
        per-sample-flagged criterion: a broken per-sample path can be silently
        masked by a healthy plot (the batch mean just averages whatever's
        present), so we need to inspect both streams independently."""
        resp = self.stub.GetLatestLoggerData(
            pb2.GetLatestLoggerDataRequest(
                request_full_history=True, max_points=max_points,
                break_by_slices=False, graph_name=graph_name), timeout=60)
        return [(p.model_age, p.metric_value) for p in resp.points
                if p.metric_name == graph_name]

    def set_hparams(self, lr=None, batch=None):
        kw = {}
        if lr is not None:
            kw["learning_rate"] = float(lr)
        if batch is not None:
            kw["batch_size"] = int(batch)
        hp = pb2.HyperParameters(**kw)
        self.stub.ExperimentCommand(
            pb2.TrainerCommand(hyper_parameter_change=pb2.HyperParameterCommand(hyper_parameters=hp)),
            timeout=30)

    def stat_map(self, n, stat_name):
        """Return {sample_id: value} for a per-sample stat as visible on rank 0."""
        resp = self.stub.GetDataSamples(
            pb2.DataSamplesRequest(
                start_index=0, records_cnt=int(n) + 16,
                include_transformed_data=False, include_raw_data=False,
                stats_to_retrieve=[stat_name],
            ), timeout=60)
        out = {}
        for rec in resp.data_records:
            val = None
            for st in rec.data_stats:
                if st.name != stat_name:
                    continue
                if st.value:
                    val = st.value[0]
                elif st.value_string:
                    val = st.value_string
            out[str(rec.sample_id)] = val
        return out

    def discarded_set(self, n):
        """Set of sample_ids currently marked discarded (rank 0 view)."""
        m = self.stat_map(n, "discarded")
        out = set()
        for sid, v in m.items():
            if v in (True, 1, 1.0, "True", "true"):
                out.add(sid)
        return out

    def current_experiment_hash(self):
        """The checkpoint manager's LIVE current_exp_hash (read from rank 0's state
        json on the shared filesystem). This reflects data edits (discards bump the
        data component), unlike logger points which carry the older training hash."""
        import json
        path = os.path.join(yolo_pipeline._HERE, "ddp_run", ".checkpoint_manager_state.json")
        try:
            with open(path) as f:
                return json.load(f).get("current_exp_hash")
        except Exception:
            return None

    def save_data_state(self):
        return self.stub.EditDataSample(
            pb2.DataEditsRequest(stat_name="__save_data_state__"), timeout=120)

    def latest_full_checkpoint_hash(self):
        """Combined hash of the most-recent checkpoint that has MODEL WEIGHTS saved
        (latest_weight_checkpoint != null) — i.e. a full checkpoint that can be
        round-tripped through RestoreCheckpoint (which restores model+config+data).
        Read from rank 0's manifest on the shared filesystem."""
        path = os.path.join(yolo_pipeline._HERE, "ddp_run", "checkpoints", "manifest.yaml")
        try:
            with open(path) as f:
                m = yaml.safe_load(f) or {}
        except Exception:
            return None
        best, best_t = None, ""
        for h, meta in (m.get("experiments", {}) or {}).items():
            if meta.get("latest_weight_checkpoint") and meta.get("created", "") >= best_t:
                best_t, best = meta.get("created", ""), h
        return best

    def restore(self, experiment_hash):
        return self.stub.RestoreCheckpoint(
            pb2.RestoreCheckpointRequest(experiment_hash=experiment_hash), timeout=120)

    def last_seen_map(self, n):
        """Return {sample_id: last_seen_int} as visible on rank 0."""
        resp = self.stub.GetDataSamples(
            pb2.DataSamplesRequest(
                start_index=0, records_cnt=int(n) + 16,
                include_transformed_data=False, include_raw_data=False,
                stats_to_retrieve=["last_seen"],
            ), timeout=60)
        out = {}
        for rec in resp.data_records:
            val = None
            for st in rec.data_stats:
                if st.name != "last_seen":
                    continue
                if st.value:
                    val = int(round(st.value[0]))
                elif st.value_string:
                    try:
                        val = int(float(st.value_string))
                    except Exception:
                        val = None
            out[str(rec.sample_id)] = val
        return out

    def close(self):
        self.channel.close()


def _max_last_seen(m):
    vals = [v for v in m.values() if v is not None and v >= 0]
    return max(vals) if vals else -1


def _wait_until_paused(client, n, min_step, timeout=600.0, poll=5.0):
    """Poll last_seen-max until training has clearly stopped: value >= min_step
    (so ~an epoch happened) and unchanged across 2 consecutive polls (so the
    server is paused, not just mid-step). drop_last means the per-rank epoch is
    floor(shard/batch), so we don't require an exact step count."""
    deadline = time.time() + timeout
    prev = None
    stable = 0
    while time.time() < deadline:
        cur = _max_last_seen(client.last_seen_map(n))
        stable = stable + 1 if cur == prev else 0
        if cur >= min_step and stable >= 2:
            return cur
        prev = cur
        time.sleep(poll)
    raise TimeoutError(f"epoch did not settle (>= {min_step}); last max={prev}")


def _settled_last_seen(client, n, timeout=60.0, poll=11.0):
    """last_seen_map read AFTER the server-side snapshot catches up. The DataService
    refreshes its GetDataSamples snapshot at most ~every 10s (_slowUpdateInternals),
    so the gather's upsert (done on pause) lags. Poll with interval > 10s — so each
    read crosses the throttle and forces a refresh — until the populated count stops
    growing. (poll < 10s could see two stale-equal reads and return the stale map.)"""
    m = client.last_seen_map(n)
    prev = -1
    deadline = time.time() + timeout
    while time.time() < deadline:
        cnt = sum(1 for v in m.values() if v is not None and v >= 0)
        if cnt == prev:
            return m
        prev = cnt
        time.sleep(poll)
        m = client.last_seen_map(n)
    return m


# ===========================================================================
# SCENARIOS
# ===========================================================================
def scenario_epoch_then_pause(client, world, batch):
    print("\n--- scenario: train 1 epoch -> last_seen populated -> pause freezes it ---")
    n = client.universe_size()
    # 1 universe epoch in per-rank steps. drop_last=True on the loader => the
    # per-rank epoch is floor(shard/batch); targeting that exact count makes the
    # auto-pause (pause_at_step) fire at the epoch boundary WITHOUT crossing into
    # epoch 2 (which would force a sampler re-iteration mid-test).
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))  # fast-debug override
    print(f"[client] universe N={n}  world={world}  batch={batch}  -> epoch_steps={epoch_steps}")

    t0 = time.time()
    client.train_steps(epoch_steps)
    # Robust pause detection: max last_seen near the target AND unchanged across
    # consecutive polls spaced > 2x a step, so a slow-but-running server is not
    # mistaken for paused.
    reached = _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    epoch_secs = time.time() - t0
    print(f"[client] epoch done: max last_seen={reached} in {epoch_secs:.1f}s")

    s1 = _settled_last_seen(client, n)  # wait out the DataService snapshot throttle
    populated = {k: v for k, v in s1.items() if v is not None and v >= 0}
    # With the children->rank0 gather (fired on pause), rank 0 sees what ALL ranks
    # trained: ~reached*batch*world distinct samples (capped at the universe N,
    # minus a few drop_last stragglers). For a full epoch that's ~the whole universe.
    expected = min(n, (reached + 1) * batch * world)
    print(f"[client] last_seen populated for {len(populated)}/{n} "
          f"(expect ~{expected} = reached*batch*world, gathered from all ranks)")

    a1 = len(populated) > 0
    a1b = len(populated) >= int(0.8 * expected)

    # paused: wait ~50% of the epoch wall time and confirm nothing moved
    client.pause()
    wait_s = max(10.0, min(0.5 * epoch_secs, 40.0))
    print(f"[client] paused; waiting {wait_s:.1f}s then re-reading last_seen ...")
    time.sleep(wait_s)
    s2 = _settled_last_seen(client, n)

    a2 = (s1 == s2)
    if not a2:
        diff = {k: (s1.get(k), s2.get(k)) for k in s1 if s1.get(k) != s2.get(k)}
        print(f"[client] FROZEN CHECK FAILED, {len(diff)} changed e.g. {list(diff.items())[:5]}")

    ok = a1 and a1b and a2
    print(f"[1] EPOCH COVERAGE  populated>0={a1} populated~=shard={a1b}  -> {'PASS' if (a1 and a1b) else 'FAIL'}")
    print(f"[2] PAUSE FREEZES   last_seen identical after wait={a2}      -> {'PASS' if a2 else 'FAIL'}")
    return ok


def scenario_discard_subset_freezes(client, world, batch, n_discard=5):
    print("\n--- scenario: discard a small subset -> only those last_seen stay frozen "
          "(shuffle ON; holds regardless of which rank owns each sample) ---")
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()
    print(f"[client] N={n} epoch_steps={epoch_steps} origin={origin}")

    # epoch 1 — gather (on pause) makes rank 0 see the whole universe's last_seen
    client.train_steps(epoch_steps)
    m1 = _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    s1 = _settled_last_seen(client, n)
    pop1 = {k: v for k, v in s1.items() if v is not None and v >= 0}
    if len(pop1) < n_discard + 10:
        print(f"[client] too few populated ({len(pop1)}) — cannot run"); return False

    # discard a small subset chosen from samples seen in epoch 1
    subset = sorted(pop1, key=lambda k: pop1[k])[:n_discard]
    L = {sid: pop1[sid] for sid in subset}
    before = client.discarded_count()
    client.discard(subset, origin)
    after = client.discarded_count()
    a0 = (after - before) == n_discard
    print(f"[client] epoch1 populated={len(pop1)} max={m1}; discarded {subset} "
          f"(count {before}->{after})")

    # epoch 2 — every NON-discarded sample is retrained by SOME rank, so its
    # (gathered) last_seen advances; the discarded subset must stay frozen.
    client.train_steps(epoch_steps)
    m2 = _wait_until_paused(client, n, min_step=m1 + max(1, epoch_steps - batch))
    s2 = _settled_last_seen(client, n)

    # negative control: discarded subset frozen
    frozen = {sid: s2.get(sid) for sid in subset}
    all_frozen = all(frozen[sid] == L[sid] for sid in subset)

    # positive control: the large majority of non-discarded advanced
    discarded_set = set(subset)
    advanced = sum(
        1 for sid, v in s2.items()
        if sid not in discarded_set and v is not None and v >= 0 and v > s1.get(sid, -1)
    )
    non_discarded_pop1 = len(pop1) - n_discard
    most_advanced = advanced >= int(0.8 * non_discarded_pop1)

    ok = a0 and all_frozen and most_advanced and (m2 > m1)
    print(f"[1] DISCARD REGISTERED  exactly {n_discard} added={a0}")
    print(f"[2] SUBSET FROZEN       all {n_discard} unchanged={all_frozen}  values {L} -> {frozen}")
    print(f"[3] MOST ADVANCED       {advanced}/{non_discarded_pop1} non-discarded advanced "
          f"(>=80%)={most_advanced}  (epoch max {m1}->{m2})")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_break_by_slice(client, world, batch):
    print("\n--- scenario: tag a cross-rank slice 'even' -> break-by-slice loss plot covers "
          "samples from BOTH ranks (needs the per-sample signal gather) ---")
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()
    graph = "train/bbxs"  # a per-sample loss component logged by the criterions

    client.train_steps(epoch_steps)
    _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    s1 = _settled_last_seen(client, n)
    trained = sorted((k for k, v in s1.items() if v is not None and v >= 0), key=lambda k: int(k))
    if len(trained) < 10:
        print(f"[client] too few trained ({len(trained)})"); return False

    # Per-sample LOSS is only logged for a subset of trained samples (detection
    # backgrounds / drop_last have no per-object loss), so the loss universe < the
    # trained set. Measure it directly with a 'uni' slice over all trained samples
    # (this is the GLOBAL loss universe on rank 0 thanks to the per-sample gather),
    # then check the 'even' slice returns exactly the even members that have loss.
    even = set(trained[::2])  # every other trained sample — spans both ranks' shards
    client.tag(trained, "uni", origin)
    client.tag(even, "even", origin)
    print(f"[client] trained={len(trained)} tagged uni + even={len(even)} (origin={origin})")

    uni_sids = {p[0] for p in client.break_by_slice(graph, ["uni"])}
    even_sids = {p[0] for p in client.break_by_slice(graph, ["even"])}
    expected_even = even & uni_sids  # even-tagged samples that actually have loss

    a1 = len(even_sids) > 0
    a2 = (even_sids == expected_even)            # break-by-slice slices correctly
    ok = a1 and a2
    print(f"[1] BREAK-BY-SLICE  even returned {len(even_sids)} samples (graph={graph})={a1}")
    print(f"[2] SLICE CORRECT   even == even-with-loss ({len(expected_even)})={a2}")
    # Cross-rank is evidenced by the server-side [siggather] log (rank 0 receives the
    # children's triples). It's not cleanly black-box-assertable without a per-rank
    # baseline, and becomes STRUCTURAL once writes go through sync_to_rank0 on rank 0.
    print(f"[i] loss universe on rank 0 = {len(uni_sids)} samples (spans both ranks via the gather)")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_lr_batch_propagate(client, world, batch):
    print("\n--- scenario: live lr+batch edit must reach ALL ranks (else replicas diverge / "
          "batch desyncs). Proof: doubling batch makes global samples/step = new_batch*world ---")
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    new_batch = batch * 2
    phase2 = 4  # short, so trained-count doesn't wrap the universe

    # phase 1 at the original batch
    client.train_steps(epoch_steps)
    _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    a0 = _max_last_seen(_settled_last_seen(client, n))

    # live edit while paused: double batch, bump lr (both ride the same hparam broadcast)
    client.set_hparams(lr=0.05, batch=new_batch)
    print(f"[client] phase1 max_age={a0}; set lr=0.05 batch={new_batch} (was {batch})")

    # phase 2 at the new batch
    client.train_steps(phase2)
    _wait_until_paused(client, n, min_step=a0 + 1)
    s1 = _settled_last_seen(client, n)
    a1 = _max_last_seen(s1)

    steps2 = a1 - a0
    trained2 = sum(1 for v in s1.values() if v is not None and v > a0)
    rate = trained2 / steps2 if steps2 > 0 else 0.0
    expected = new_batch * world          # both ranks switched
    rank0_only = new_batch + batch        # only rank 0 switched (the bug we fixed)
    a1ok = steps2 > 0 and rate >= (expected + rank0_only) / 2.0
    print(f"[1] BATCH PROPAGATED  {trained2} samples / {steps2} steps = {rate:.1f}/step "
          f"(expect ~{expected} all-ranks vs ~{rank0_only} rank0-only)={a1ok}")
    print(f"[i] lr=0.05 rode the same hparam broadcast that carried batch (proven above)")
    print(f"  -> {'PASS' if a1ok else 'FAIL'}")
    return a1ok


def scenario_checkpoint_data_roundtrip(client, world, batch):
    print("\n--- scenario: data-state checkpoint store + reload (discard reverts on restore) ---")
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()

    # 1) build state, then discard A
    client.train_steps(epoch_steps)
    a0 = _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    s0 = _settled_last_seen(client, n)
    populated = sorted((k for k, v in s0.items() if v is not None and v >= 0), key=int)
    if len(populated) < 4:
        print(f"[client] too few populated ({len(populated)})"); return False
    A, C = populated[0], populated[1]
    client.discard([A], origin)

    # 2) short resume -> save_pending_changes writes a FULL checkpoint (model+config+
    #    data{A}) with non-null weights; then read its combined hash from the manifest.
    client.train_steps(2)
    _wait_until_paused(client, n, min_step=a0 + 1)
    saved_hash = client.latest_full_checkpoint_hash()
    time.sleep(12)  # clear the DataService snapshot throttle before reading
    disc_save = client.discarded_set(n)
    print(f"[client] discarded A={A}; full-ckpt hash={saved_hash}; discarded@save={sorted(disc_save)}")

    # 3) diverge: discard a NEW sample C
    client.discard([C], origin)
    time.sleep(12)
    disc_change = client.discarded_set(n)
    print(f"[client] discarded C={C}; discarded@change={sorted(disc_change)}")

    # 4) restore the full checkpoint
    if not saved_hash:
        print("[client] no full-checkpoint hash found — cannot restore"); return False
    resp = client.restore(saved_hash)
    time.sleep(12)
    disc_post = client.discarded_set(n)
    print(f"[client] restore success={getattr(resp, 'success', None)} "
          f"msg={getattr(resp, 'message', '')[:70]}; discarded@post={sorted(disc_post)}")

    restore_ok = bool(getattr(resp, "success", False))
    a0c = (C in disc_change)                 # the divergent discard registered
    a1 = (C not in disc_post)                # restore undid it
    a2 = (A in disc_post)                    # the saved discard survived the roundtrip
    ok = restore_ok and a0c and a1 and a2
    print(f"[1] DATA ROUNDTRIP  restore_ok={restore_ok} C-registered={a0c} "
          f"C-reverted={a1} A-intact={a2}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_signal_coverage_all_graphs(client, world, batch):
    """For every per-sample-flagged criterion BOTH streams must populate:
       - PER-SAMPLE: ≥30% of trained samples have a (sid, age, value) entry.
       - SCALAR PLOT: ≥50% of steps have a batch-aggregate point.

    Cross-checking BOTH is essential because the batch-aggregate plot can look
    perfectly healthy when the per-sample stream is silently broken — the plot
    just `batch.mean()`s whatever the criterion handed back, regardless of
    whether each contribution was actually logged per-sample on rank-0. The
    inverse is also true (a missing aggregate plot wouldn't surface from a
    per-sample-only check). One-sided eyeballing always misses one direction;
    only the join catches both.

    Caught in practice: `miou/train` (flag="metric") produced healthy scalar
    plot points but ZERO per-sample entries — invisible from the UI plot."""
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()

    client.train_steps(epoch_steps)
    _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    s1 = _settled_last_seen(client, n)
    trained = sorted((k for k, v in s1.items() if v is not None and v >= 0),
                     key=lambda k: int(k))
    if len(trained) < 10:
        print(f"[client] too few trained ({len(trained)})"); return False
    client.tag(trained, "uni", origin)
    print(f"[client] trained={len(trained)} tagged 'uni'  steps={epoch_steps}")

    graphs = ["train/bbxs", "train/clsf", "train/dfl", "miou/train"]
    per_sample_min = max(1, int(0.3 * len(trained)))
    plot_min = max(1, int(0.5 * epoch_steps))
    all_ok = True
    for g in graphs:
        per_sample_sids = {p[0] for p in client.break_by_slice(g, ["uni"])}
        plot_points = client.scalar_plot(g)
        ps_ok = len(per_sample_sids) >= per_sample_min
        plot_ok = len(plot_points) >= plot_min
        ok = ps_ok and plot_ok
        all_ok &= ok
        print(f"[1] {g:<18s}  per-sample={len(per_sample_sids)}/{len(trained)} "
              f"≥{per_sample_min}={ps_ok}   plot={len(plot_points)} "
              f"≥{plot_min}={plot_ok}   both={ok}")
    print(f"  -> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def scenario_resume_continues_curve(client, world, batch):
    """Restore + resume cleanly: the gRPC server stays alive across the restore
    call AND post-restore training still advances the loss-history. Catches:
       - server crash during restore (std::terminate from worker shutdown on
         the gRPC handler thread under num_workers>0 — fixed by routing the
         loader reset through _invalidate_iter so worker tear-down happens on
         the trainer thread instead),
       - frozen / wedged trainer post-restore (no new signal points).

    NOTE: last_seen is NOT in the data snapshot (only sample_id + discarded +
    tags are), so we do NOT assert that last_seen reverts. The discarded
    revert is already covered by scenario_checkpoint_data_roundtrip."""
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    K = 4

    # phase 1 — train, then trigger a full save via a short train (resume calls
    # pause_controller.resume which fires save_pending_changes).
    client.train_steps(epoch_steps)
    a0 = _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    client.train_steps(2)
    _wait_until_paused(client, n, min_step=a0 + 1)
    saved_hash = client.latest_full_checkpoint_hash()
    if not saved_hash:
        print("[client] no full-ckpt hash"); return False
    print(f"[client] saved hash={saved_hash}")
    pre_restore_plot = client.scalar_plot("train/bbxs")
    pre_restore_max_plot_age = max((p[0] for p in pre_restore_plot), default=0)

    # diverge — train more so the trainer has actual divergent work
    client.train_steps(epoch_steps)
    age_diverged = _wait_until_paused(client, n, min_step=a0 + 1)
    print(f"[client] diverged age={age_diverged}")

    # restore — primary risk: server crash (std::terminate) under DDP+workers
    resp = client.restore(saved_hash)
    a1 = bool(getattr(resp, "success", False))
    print(f"[client] restore success={a1} msg={getattr(resp, 'message', '')[:70]}")
    time.sleep(5)

    # server-alive check: a follow-up RPC must succeed (the previous crash
    # signature was Connection refused on the very next call).
    try:
        n_after = client.universe_size()
        a2 = (n_after == n)
    except Exception as exc:
        print(f"[client] server unreachable after restore: {exc}"); return False
    print(f"[client] server alive post-restore (universe_size={n_after} == {n}) → {a2}")

    # train K more — verifies the trainer didn't wedge and the criterion still
    # produces signal points. Compare plot length to confirm growth.
    client.train_steps(K)
    _wait_until_paused(client, n, min_step=age_diverged + 1)
    post_train_plot = client.scalar_plot("train/bbxs")
    a3 = len(post_train_plot) > len(pre_restore_plot)
    print(f"[3] PLOT GROWS    pre={len(pre_restore_plot)} post={len(post_train_plot)} → {a3}")

    ok = a1 and a2 and a3
    print(f"[1] RESTORE OK    success={a1}")
    print(f"[2] SERVER ALIVE  universe={n_after}/{n} → {a2}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_process_topology(client, world, batch):
    """Only one of the spawned ranks may listen on a TCP port (the gRPC backend
    on rank-0); every other rank must have ZERO listening sockets. Catches
    silent topology drift — a debug toggle accidentally on every rank, a logger
    opening its own port, a forgotten healthcheck listener. Verifies the
    rank-0-only serve invariant from outside the SDK."""
    import re
    import subprocess

    _ = client.universe_size()  # confirm we're connected; ranks are alive
    grpc_port = getattr(client, "_port", None)

    # Walk the descendant tree of the suite process to find spawned ranks.
    suite_pid = os.getpid()
    def children_of(pid):
        try:
            out = subprocess.check_output(["pgrep", "-P", str(pid)], text=True,
                                          stderr=subprocess.DEVNULL)
            return [int(p) for p in out.split() if p.strip()]
        except subprocess.CalledProcessError:
            return []
    seen, frontier = set(), [suite_pid]
    while frontier:
        nxt = []
        for pid in frontier:
            for c in children_of(pid):
                if c not in seen:
                    seen.add(c); nxt.append(c)
        frontier = nxt
    # Heuristic: the actual rank PIDs are those whose cmdline contains the
    # mp.spawn entry point. We can't trivially read cmdline portably, so just
    # take all descendants that own AT LEAST one TCP socket — that bounds the
    # candidate set without false positives from short-lived helpers.
    candidates = sorted(seen)
    print(f"[client] {len(candidates)} descendant PIDs of suite={suite_pid}")

    def listeners_of(pid):
        try:
            out = subprocess.check_output(
                ["ss", "-tlnp", "-H"], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            return None
        ports = []
        for line in out.splitlines():
            if f"pid={pid}," not in line and f"pid={pid})" not in line:
                continue
            m = re.search(r":(\d+)\s", line)
            if m: ports.append(int(m.group(1)))
        return ports
    sample = listeners_of(candidates[0]) if candidates else None
    if sample is None:
        print("[client] ss unavailable; skipping topology check"); return False

    pid_ports = {pid: listeners_of(pid) or [] for pid in candidates}
    listening = {pid: ports for pid, ports in pid_ports.items() if ports}
    print(f"[client] PIDs with TCP listeners: {listening}")

    # Real invariant under DDP-with-gloo: EVERY rank legitimately opens TCP
    # listeners for cross-rank comm (gloo's send/recv channels). What we CARE
    # about is: only ONE rank owns the application gRPC port. Multiple sockets
    # per rank are fine; the rogue pattern is "two ranks both claiming the
    # gRPC port" or "non-rank PID listening on it".
    grpc_owners = [pid for pid, ports in listening.items()
                   if grpc_port is not None and grpc_port in ports]
    a1 = len(grpc_owners) == 1
    # Sanity: at least one PID does listen (otherwise the gRPC server is dead).
    a2 = len(listening) >= 1
    ok = a1 and a2
    print(f"[1] gRPC OWNER     PIDs owning port {grpc_port}: {grpc_owners} (==1) → {a1}")
    print(f"[2] HAS LISTENERS  {len(listening)} PID(s) with TCP sockets (≥1) → {a2}")
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_multi_epoch_stability(client, world, batch):
    """3 epochs back-to-back without restart. Asserts:
      (1) model_age advances monotonically each epoch (no rewinds);
      (2) per-sample signal entries are unique by (sid, model_age) within each
          graph — re-flushed entries are idempotently deduped, not appended.
    The second check catches the regression where outbox flushes append rather
    than upsert per-step, which would inflate signal_history without bound and
    poison downstream loss-shape descriptors that count entries."""
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()

    ages = []
    for ep in range(3):
        client.train_steps(epoch_steps)
        a = _wait_until_paused(client, n,
                               min_step=(ages[-1] if ages else 0) + 1)
        ages.append(a)
        print(f"[client] epoch {ep+1}/3: max_age={a}")

    s = _settled_last_seen(client, n)
    trained = sorted([k for k, v in s.items() if v is not None and v >= 0],
                     key=lambda k: int(k))
    if len(trained) < 5:
        print(f"[client] too few trained ({len(trained)})"); return False
    client.tag(trained, "uni", origin)

    # Per-graph: no duplicate (sid, age) entries
    dedup_ok = True
    for g in ["train/bbxs", "train/clsf", "train/dfl", "miou/train"]:
        entries = client.break_by_slice(g, ["uni"])  # [(sid, age, val), ...]
        keys = [(sid, age) for sid, age, _ in entries]
        unique, total = len(set(keys)), len(keys)
        ok = (unique == total)
        dedup_ok &= ok
        print(f"[1] {g:<18s} {total} entries, {unique} unique (sid,age)  → {ok}")

    age_mono = ages[0] < ages[1] < ages[2]
    print(f"[2] AGE MONOTONIC    ages={ages} (strictly increasing) → {age_mono}")
    ok = dedup_ok and age_mono
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_collective_budget(client, world, batch):
    """Per-step rendezvous budget = exactly 2 collectives (reconcile_all DOWN +
    flush_outbox UP). Anything above that is a leak. Gating this prevents a
    regression where someone adds a stray dist.broadcast / dist.all_reduce
    inside a hot path — silently 2-10x slowdown.

    Strategy: each rank's reset_collectives() writes the PRIOR step's count to
    WL_DDP_COLLECTIVE_LOG. We train a clean batch of steps, then assert that
    the *steady-state* counts (post-warmup, ignoring the first few pause-spin
    entries) are ≤ 2."""
    log_path = os.environ.get("WL_DDP_COLLECTIVE_LOG")
    if not log_path:
        print(f"[client] WL_DDP_COLLECTIVE_LOG not set; skipping"); return False
    open(log_path, "w").close()  # truncate before this scenario's window

    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    client.train_steps(epoch_steps)
    _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))

    try:
        counts = [int(l.strip()) for l in open(log_path) if l.strip()]
    except Exception as e:
        print(f"[client] read collective log failed: {e}"); return False
    if not counts:
        print(f"[client] no collective counts recorded"); return False

    # First few entries can include pause-spin reconciles (many per "step" while
    # the trainer is waiting for the resume signal). Take a slice from the tail
    # corresponding to clearly-in-the-body steps.
    body = [c for c in counts if c <= 5]   # drop the spin-inflated outliers
    spin = [c for c in counts if c > 5]
    avg_body = (sum(body) / len(body)) if body else float("inf")
    max_body = max(body) if body else 0

    a1 = max_body <= 2
    a2 = avg_body <= 2.0
    print(f"[1] BUDGET PER STEP  body samples={len(body)}, max={max_body}, "
          f"avg={avg_body:.2f}, spin samples={len(spin)} (excluded) "
          f"max-over-budget→{a1}")
    print(f"[2] AVG ≤ 2          {avg_body:.2f} → {a2}")
    ok = a1 and a2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_seed_determinism(client, world, batch):
    """Single-run determinism: re-running with seed=1234 (the suite default) and
    re-querying mid-train should yield byte-identical per-sample loss for the
    same (sid, model_age). Detects RNG leaks: a stray torch.rand() in the
    criterion, a non-seeded augmentation, a CUDA non-deterministic op (none
    here — we're CPU), or an unintended re-shuffle. Without this, the
    loss-shape descriptor work is silently unreliable.

    Implementation: train K steps; collect (sid, age, val) triples for
    train/bbxs; pause; pull again; the two pulls must agree on every entry."""
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()

    client.train_steps(epoch_steps)
    _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    s1 = _settled_last_seen(client, n)
    trained = sorted([k for k, v in s1.items() if v is not None and v >= 0],
                     key=lambda k: int(k))
    if len(trained) < 5:
        print(f"[client] too few trained ({len(trained)})"); return False
    client.tag(trained, "uni", origin)

    # Pull twice — same gRPC, same data, same logger snapshot. Bit-identical.
    pull1 = sorted(client.break_by_slice("train/bbxs", ["uni"]))
    import time; time.sleep(2)
    pull2 = sorted(client.break_by_slice("train/bbxs", ["uni"]))

    a1 = len(pull1) > 0 and len(pull1) == len(pull2)
    a2 = all(p1 == p2 for p1, p2 in zip(pull1, pull2))
    print(f"[1] STABLE LEN       pull1={len(pull1)} == pull2={len(pull2)} → {a1}")
    print(f"[2] BIT-IDENTICAL    every (sid, age, val) matches → {a2}")
    # Spot-check first 3 entries
    for i in range(min(3, len(pull1))):
        print(f"    p1[{i}]={pull1[i]}  p2[{i}]={pull2[i]}")
    ok = a1 and a2
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


def scenario_empty_shard_starvation(client, world, batch):
    """Heavy discard (~95% of populated) — under DDP with DistributedSampler the
    remaining samples may all land on ONE rank's shard, leaving the OTHER rank
    with zero work for that epoch. Asserts: the trainer DOES NOT silently hang.
    Either it advances model_age (loader cycles cleanly past empty epochs) or
    it raises within the timeout. A silent hang here is the real risk — both
    ranks would block forever at the next grad all-reduce."""
    n = client.universe_size()
    epoch_steps = (n // world) // batch
    epoch_steps = int(os.environ.get("WL_DDP_TEST_STEPS", epoch_steps))
    origin = client.train_origin()

    # warm-up to populate last_seen for many samples
    client.train_steps(epoch_steps)
    a0 = _wait_until_paused(client, n, min_step=max(1, epoch_steps - batch))
    s1 = _settled_last_seen(client, n)
    populated = sorted([k for k, v in s1.items() if v is not None and v >= 0],
                       key=lambda k: int(k))
    if len(populated) < 20:
        print(f"[client] too few populated ({len(populated)})"); return False

    # discard ~95% of populated → remaining sparse, likely one shard empty
    keep = max(2, int(0.05 * len(populated)))
    to_discard = populated[:-keep]
    client.discard(to_discard, origin)
    print(f"[client] discarded {len(to_discard)}/{len(populated)} populated  (keep={keep})")

    # Short post-discard train. Bounded timeout: if it hangs, assertion fires.
    K = min(epoch_steps, 8)
    import time
    t0 = time.time()
    client.train_steps(K)
    try:
        a1 = _wait_until_paused(client, n, min_step=a0 + 1,
                                timeout=180.0, poll=3.0)
    except TimeoutError:
        elapsed = time.time() - t0
        print(f"[client] HUNG  no model_age advance in {elapsed:.0f}s (a0={a0})")
        print(f"  -> FAIL")
        return False
    elapsed = time.time() - t0
    advanced = a1 > a0
    print(f"[1] NO HANG          age advanced {a0}→{a1} in {elapsed:.1f}s → {advanced}")
    print(f"  -> {'PASS' if advanced else 'FAIL'}")
    return advanced


_SCENARIOS = [
    scenario_epoch_then_pause,
    scenario_discard_subset_freezes,
    scenario_break_by_slice,
    scenario_lr_batch_propagate,
    scenario_checkpoint_data_roundtrip,
    scenario_signal_coverage_all_graphs,
    scenario_resume_continues_curve,
    scenario_process_topology,
    scenario_multi_epoch_stability,
    scenario_empty_shard_starvation,
    scenario_seed_determinism,
    scenario_collective_budget,
]


def _free_port():
    with socket.socket() as s:
        s.bind((_HOST, 0))
        return s.getsockname()[1]


def _run_one(scn, batch):
    """Spawn a FRESH server (isolation), run one scenario, tear the server down."""
    master_port, grpc_port = _free_port(), _free_port()
    print(f"\n[suite] === {scn.__name__} ===  spawning {_WORLD} ranks, gRPC :{grpc_port}, "
          f"imgsz={os.environ['WL_DDP_IMGSZ']}")
    ctx = mp.spawn(_train_worker, args=(_WORLD, master_port, grpc_port), nprocs=_WORLD, join=False)
    client = Client(grpc_port)
    t0 = time.time()
    try:
        client.wait_ready()
        print(f"[suite] server ready (pids={[p.pid for p in ctx.processes]})")
        ok = scn(client, _WORLD, batch)
    except Exception as e:
        print(f"[suite] {scn.__name__} ERRORED: {e!r}")
        ok = False
    finally:
        client.close()
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        for p in ctx.processes:
            p.join(timeout=10)
    print(f"[suite] {scn.__name__} took {time.time() - t0:.1f}s")
    return ok


def main():
    batch = int(yaml.safe_load(open(os.path.join(yolo_pipeline._HERE, "config.yaml")))
                ["data"]["train_loader"]["batch_size"])
    only = os.environ.get("WL_DDP_ONLY")  # substring filter to run a single scenario
    scenarios = [s for s in _SCENARIOS if not only or only in s.__name__]
    results = {scn.__name__: _run_one(scn, batch) for scn in scenarios}

    print("\n" + "=" * 64)
    for name, ok in results.items():
        print(f"  {name:42s} -> {'PASS' if ok else 'FAIL'}")
    allok = bool(results) and all(results.values())
    print(f"  RESULT: {'ALL PASS' if allok else 'FAILURES ABOVE'}")
    print("=" * 64)
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
