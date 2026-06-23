"""Aggregate a py-spy folded profile into where the instruction pointer actually is,
by OWNERSHIP (not A/B section deltas). Beyond "is it WL SDK", it carves out the
GREY ZONE — work that isn't WL SDK code but only exists BECAUSE of WL:

  WL-SDK            leaf in weightslab/ (save_signals, reconcile/flush, merge, dataframe,
                    the wrapper's own code) — EXCL weightslab/baseline_models; OR a
                    pandas/numpy/h5py leaf whose nearest non-lib caller is weightslab/.
  decode-for-log    `_decode_preds_to_6col` on the stack — NMS run ONLY to log predictions
                    (pure WL-motivated overhead; the UL baseline never decodes). [GREY]
  loss:per-sample   `criterions.py` on the stack — the per-sample loss wrapper AND the
                    ultralytics loss.py/tal.py it drives. Per-sample BECAUSE WL wants
                    per-sample signals; upper bound on WL-induced loss cost. [GREY]
  model:forward     ultralytics/nn leaf, no criterions/decode — the network itself.
  torch:bwd/sync    torch leaf — backward + grad all_reduce + optimizer.
  loss:compute      ultralytics loss/tal/ops leaf with NO criterions wrapper (UL baseline).
  usecase/data,     dataloader/collate, the driver loop, idle.
  harness, other

  "WL-attributable" = WL-SDK + the grey zone. Run:
    FOLDED=/tmp/wl_ablation.folded python aggregate_wl_ownership.py
"""
import os, re, collections

PATH = os.environ.get("FOLDED", "/tmp/wl_ablation.folded")
_LIB_DATA = ("pandas", "numpy", "h5py", "pyarrow")
_GREY = {"decode-for-log", "loss:per-sample"}


def _file_of(frame):
    m = re.search(r"\(([^)]*)\)\s*$", frame)
    return m.group(1) if m else frame


def _is_wl(f):
    return "weightslab/" in f and "baseline_models" not in f


_IMG = ("patches.py", "ultralytics/data/", "/cv2/", "albumentations", "PIL/", "imgaug")
_LOSS = ("ultralytics/utils/tal.py", "ultralytics/utils/loss.py", "ultralytics/utils/metrics.py")


def classify(frames):
    files = [_file_of(f) for f in frames]
    leaf = files[-1]
    blob = ";".join(frames)        # func names + files across the whole stack
    # 1. WL SDK = WL owns the leaf (its own code running now)
    if _is_wl(leaf):
        return "WL-SDK"
    # pandas/numpy/h5py leaf — attribute to the first non-lib caller (WL-induced?)
    if any(l in leaf for l in _LIB_DATA):
        for f in reversed(files[:-1]):
            if any(l in f for l in _LIB_DATA):
                continue
            if _is_wl(f):
                return "WL-SDK"
            if any(s in f for s in _IMG):   # numpy under imread = decode
                return "data:img-decode"
            break
    # 2. grey zone (WL-motivated) — most specific intent wins
    if "_decode_preds_to_6col" in blob:
        return "decode-for-log"
    # 3. image decode / augment (usecase; you'd load images regardless)
    if any(s in leaf for s in _IMG):
        return "data:img-decode"
    # 4. loss — per-sample wrapper, or the ultralytics loss/assigner/metrics it drives
    if "criterions.py" in blob:
        return "loss:per-sample"
    if any(s in leaf for s in _LOSS) or any(s in blob for s in _LOSS):
        return "loss:compute"
    # 5. model forward (network); torch leaf disambiguated by stack context
    if "ultralytics/nn/" in leaf or "ultralytics/nn/" in blob:
        return "model:forward"
    if "torch/" in leaf:
        return "torch:bwd/sync"
    if any(s in leaf for s in ("yolo_pipeline", "/utils/data", "/data.py")):
        return "data:img-decode"
    if "ddp_ablation" in leaf or "ddp_test_suite" in leaf:
        return "harness"
    return "other/idle"


def main():
    buckets = collections.Counter()
    wl_frames = collections.Counter()
    total = 0
    with open(PATH) as fh:
        for line in fh:
            m = re.match(r"^(.*)\s+(\d+)$", line.rstrip("\n"))
            if not m:
                continue
            stack, cnt = m.group(1), int(m.group(2))
            # drop py-spy's "process N:..." subprocess-header pseudo-frames
            frames = [f for f in stack.split(";") if not f.startswith("process ")]
            if not frames:
                continue
            total += cnt
            buckets[classify(frames)] += cnt
            seen = set()
            for f in frames:
                ff = _file_of(f)
                if _is_wl(ff):
                    key = ff.split("/weightslab/")[-1]
                    if key not in seen:
                        wl_frames[key] += cnt
                        seen.add(key)
    if not total:
        print(f"no samples in {PATH}"); return

    print(f"TOTAL SAMPLES: {total}  (~{total/200:.0f}s @ 200Hz)\n")
    print("OWNERSHIP PARTITION (where is the instruction pointer):")
    for b, c in buckets.most_common():
        tag = "  <- GREY (WL-motivated)" if b in _GREY else ""
        print(f"  {b:18s} {c:8d}  {100*c/total:5.1f}%{tag}")

    wl = buckets.get("WL-SDK", 0)
    grey = sum(buckets.get(g, 0) for g in _GREY)
    print(f"\n  WL-SDK code            = {100*wl/total:5.1f}%  (decode/loss/bridge EXCLUDED)")
    print(f"  + grey zone (decode+per-sample loss) = {100*grey/total:5.1f}%")
    print(f"  = WL-ATTRIBUTABLE      = {100*(wl+grey)/total:5.1f}%  (SDK + only-because-of-WL work)")

    print("\nTOP WL-SDK files (inclusive — on the call path, not necessarily the leaf):")
    for f, c in wl_frames.most_common(12):
        print(f"  {c:7d} {100*c/total:5.1f}%  {f[:70]}")


if __name__ == "__main__":
    main()
