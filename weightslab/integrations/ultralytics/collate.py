"""Collate fn that bridges WL's per-sample tuples → UL's batch dict, and
stashes the wrapper-canonical uids under `batch['ids']`."""
from __future__ import annotations

import torch as th

from ultralytics.data.dataset import YOLODataset


def wl_ul_dict_collate(batchs):
    """Collate WL tracking-wrapper per-sample tuples into a UL batch dict.

    Each item is `(image, uid, label, metadata)` from `WLAwareDataset` after
    being wrapped by `DataSampleTrackingWrapper` (so `uid` is the wrapper's
    globally-unique uid, not the positional index). We unpack the UL native
    sample dict from `metadata['batch']`, call UL's own collate on those
    dicts, then stash the per-image uids under `batch['ids']` so train/val
    hooks can key per-sample signals by canonical uid without colliding with
    train rows in the shared ledger."""
    ul_samples = [item_meta["batch"] for (_img, _uid, _label, item_meta) in batchs]
    uids = [int(_uid) for (_img, _uid, _label, _meta) in batchs]
    result = YOLODataset.collate_fn(ul_samples)
    result["ids"] = th.tensor(uids, dtype=th.long)
    return result
