# Interactivity for 100GB+ datasets — register of O(data) operations

Goal: every per-step and per-request operation should cost **O(change)** or
**O(page)**, never **O(dataset)**. Today several do, so cost grows with the
dataset while the actual work stays constant.

> **Baseline caveat.** The measurements below were taken against a stale
> in-place copy of weightslab (`~/weightslab_src`, 1.3.3+multiview), which
> differs from `dev` in 51 files. Two serving costs listed there are **already
> fixed on dev** (see §B). Serving numbers must be re-measured on this branch
> before any serving change is attributed an improvement. The storage findings
> (§A) were re-verified against dev and still hold.

Reference measurements (UltraEdit, 3,959,093 rows, ~19 cols, A10G box):

| | measured |
|---|---|
| bare-torch step (no WL) | 1,162 ms → 20.66 samples/s |
| with WL, no UI client | ~5.5 samples/s (**3.8× slower**) |
| with WL + image requests | ~1.5 samples/s (**13.8× slower**) |
| per-step signal write itself | **4 ms (0.34%)** — already fine |
| grid page latency (64 imgs) | p50 5.2 s, p95 9.6 s |

The signal path is not the problem. Storage write-amplification and the view
rebuild are.

---

## A. STORAGE — `data/h5_dataframe_store.py`

`upsert()` **receives** only dirty rows but **implements** a full table
replacement.

| line | operation | cost |
|---|---|---|
| 693 | `_create_backup()` — full file copy **before every upsert** | O(file) |
| 708 | `existing = store.select(key)` — read entire table | O(N) |
| ~768 | `pd.concat([existing, delta])` | O(N) |
| ~772 | `existing[~existing.index.duplicated()]` — dedupe all rows | O(N) |
| ~785 | `_decategorize_for_storage(existing)` | O(N) |
| 801 | `store.remove(key)` — drop table | O(N) |
| 804 | `store.append(..., data_columns=True)` — rewrite + index **every** column | O(N·cols) |
| 846–883 | same read/remove/rewrite in the column-delete path | O(N) |

**Amplification:** ~5 KB of changed signals per flush → ~200 MB written,
roughly **40,000×**. At `ledger_flush_interval=3.0s` vs ~1.5 s steps, that is a
full-table rewrite about every 2 steps.

Fix direction: append new rows; modify existing rows in place
(`select_as_coordinates` + `table.modify_rows`). Backup incrementally, not per
upsert. No `data_columns=True` — no `store.select()` in this file uses `where=`,
so those per-column indexes are built and never read.

*(A previous attempt to narrow `data_columns` broke the write path entirely —
678 upsert failures, zero persisted data. Any change here needs a
write→read→assert-contents check, not just a timing check.)*

## B. SERVING — `trainer/services/data_service.py`

`_pull_into_all_data_view_df()` (line 938) runs several full-frame passes.

**Already fixed on dev — do not re-report as wins:**
- the collapse no longer re-enters `get_combined_df()`; the pulled frame is
  passed in, so the frame is copied once, not twice
- `array_proxy` no longer does a per-cell `.apply(convert_to_proxy)` (was
  1,660 ms at 4M rows)

Remaining, to be **re-measured on this branch**:

| line | operation | cost (measured @4M) |
|---|---|---|
| 946 | `get_combined_df()` → `dataframe_manager:2140 self._df.copy()` | 101 ms–1.2 s |
| — | `get_collapse_annotations_to_samples_df(df)` — groupby collapse | 6,326 ms* |
| — | `safe_reset_index(df)` | 1,912 ms |
| — | `set_index([origin, sample_id])` | O(N) |
| 3636 | `updated_df.reindex(target_order)` | 290 ms |

Callers — each one is a full O(N) rebuild: lines **440, 851, 3593, 4605, 4626**,
reached from `GetDataSamples`, `GetMetaData`, `EditDataSample`, `GetDataSplits`.

Held under `_update_lock`, which the trainer also needs → measured lock holds of
39–126 s and the 3.7× training penalty while browsing.

**The collapse is provably a no-op when `annotation_id.max() == 0`** (UltraEdit
is exactly 1:1) yet still costs 6.3 s per rebuild.

Fix direction: serve a page from the source frame by index (O(page)); rebuild
the full view only for genuinely global operations (histogram, global sort);
apply deltas rather than rebuilding; never hold the writer lock across a
rebuild — build off-lock and swap the reference.

## C. OTHER FULL SCANS

| location | note |
|---|---|
| `dataframe_manager:1875` `data_snapshot.iterrows()` | input is O(change), but row-wise Python per flush |
| `dataframe_manager:2400` `.apply(lambda …)` | per cell |
| `data_service:1200` `_compute_natural_sort_stats` | builds a list of one Series per row (4M objects). Gated off (`compute_natural_sort=False`) — latent |
| `data_service:538` PreviewCache | bounded by `WL_MAX_PREVIEW_CACHE_SIZE` — OK |

## D. ALREADY O(change) — keep

- `self._pending` dirty-row set (`dataframe_manager:95, 751, 761`)
- flush work set: `work = list(self._pending)` (`:1827`)
- `_origin_revisions` per-origin version counters (`:94, 1235`)

The bookkeeping needed for differential updates already exists; the storage and
view layers just don't use it.

\* measured on the stale copy; re-measure on dev.

## Measurement protocol

Fixed workload: **1,000 train samples** = 41 steps at batch 24. Every change is
reported as:

1. wall-clock for the 41 steps, vs the bare-torch floor
2. bytes written to H5 for those steps
3. grid-page latency (64 images) and training throughput **while** serving
4. **ledger contents verified** — signal columns present, measured-row count

(4) is not optional: a previous "10× win" was writes silently failing.
