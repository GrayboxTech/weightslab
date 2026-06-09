"""Empty-val crash scenario.

Discards every active val sample, then waits through 2 val cycles. Without
the `WLAwareTrainer.validate()` guard, UL crashes on `np.concatenate([])`
(or, with `(None, None)` returns, on `**self.metrics`). With the guard,
the trainer logs the val cycles as skipped and continues training.

PASS = trainer still alive + at least 2 val intervals elapsed without
crash markers in the log.
"""
import re
import sys
import time
import grpc
import subprocess
from weightslab.proto import experiment_service_pb2 as pb2
from weightslab.proto import experiment_service_pb2_grpc as pb2_grpc

GRPC_ADDR = "localhost:50051"
LOG_PATH = "/tmp/main_ul_run.log"
VAL_LINE_RE = re.compile(r"^\s+all\s+\d+\s+\d+")  # UL val summary line
CRASH_MARKERS = (
    "Traceback",
    "ValueError: need at least one array to concatenate",
    "TypeError: 'NoneType' object is not a mapping",
)


def is_trainer_alive():
    out = subprocess.run(
        ["pgrep", "-f", "main_ul_native"], capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def count_val_lines():
    try:
        with open(LOG_PATH) as f:
            return sum(1 for ln in f if VAL_LINE_RE.match(ln))
    except FileNotFoundError:
        return 0


def crash_in_log():
    try:
        with open(LOG_PATH) as f:
            data = f.read()
    except FileNotFoundError:
        return None
    for marker in CRASH_MARKERS:
        if marker in data:
            return marker
    return None


def main():
    chan = grpc.insecure_channel(
        GRPC_ADDR,
        options=[("grpc.max_receive_message_length", 256 * 1024 * 1024)],
    )
    stub = pb2_grpc.ExperimentServiceStub(chan)

    # Wait for first val.
    print("[client] waiting for first val cycle…", flush=True)
    while count_val_lines() < 1:
        if not is_trainer_alive():
            print("[client] FAIL — trainer died before first val", flush=True)
            sys.exit(1)
        time.sleep(5)
    val_before = count_val_lines()
    print(f"[client] first val seen (val_lines={val_before})", flush=True)

    # Fetch active val sids.
    r = stub.GetDataSamples(
        pb2.DataSamplesRequest(start_index=0, records_cnt=5000), timeout=60.0,
    )
    val_ids = []
    for rec in r.data_records:
        origin, discarded = None, 0
        for s in rec.data_stats:
            if s.name == "origin":
                origin = s.value_string
            if s.name == "discarded":
                discarded = (s.value[0] if s.value else 0)
        if origin == "val_loader" and not discarded:
            val_ids.append(rec.sample_id)
    print(f"[client] active val samples: {len(val_ids)}", flush=True)
    if not val_ids:
        print("[client] FAIL — no active val samples to discard", flush=True)
        sys.exit(1)

    # Discard all.
    req = pb2.DataEditsRequest(
        stat_name="discarded", bool_value=True,
        type=pb2.SampleEditType.EDIT_OVERRIDE,
        samples_ids=val_ids,
        sample_origins=["val_loader"] * len(val_ids),
    )
    resp = stub.EditDataSample(req, timeout=30.0)
    print(f"[client] discard: success={resp.success} msg='{resp.message}'", flush=True)

    # Resume (discard auto-paused).
    cmd = pb2.TrainerCommand(
        hyper_parameter_change=pb2.HyperParameterCommand(
            hyper_parameters=pb2.HyperParameters(is_training=True),
        ),
    )
    stub.ExperimentCommand(cmd, timeout=10.0)
    print("[client] resumed training", flush=True)

    # Watch for 2 more val cycles or crash markers.
    target = val_before + 2
    deadline = time.time() + 1800  # 30 min max on CPU
    while True:
        if not is_trainer_alive():
            print("[client] FAIL — trainer died after discard", flush=True)
            crash = crash_in_log()
            if crash:
                print(f"[client]   crash marker: {crash}", flush=True)
            sys.exit(1)
        crash = crash_in_log()
        if crash:
            print(f"[client] FAIL — crash in log: {crash}", flush=True)
            sys.exit(1)
        if count_val_lines() >= target:
            print(f"[client] PASS — 2 val cycles elapsed without crash", flush=True)
            sys.exit(0)
        if time.time() > deadline:
            print(f"[client] FAIL — timeout waiting for {target} val lines", flush=True)
            sys.exit(1)
        time.sleep(10)


if __name__ == "__main__":
    main()
