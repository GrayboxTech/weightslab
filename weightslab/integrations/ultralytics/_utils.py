"""UL-version compat shim — `normalize_post_nms_preds`."""
from __future__ import annotations

import torch as th


def normalize_post_nms_preds(preds_list):
    """UL 8.4.51 returns post-NMS as `list[dict]` per image with keys
    `bboxes` (Ni, 4), `conf` (Ni,), `cls` (Ni,), `extra` (ignored). Older UL
    returns `list[Tensor(Ni, 6)] = [x1,y1,x2,y2,conf,cls]`. Normalize to the
    older shape so downstream stays version-agnostic."""
    out = []
    for p in preds_list:
        if isinstance(p, dict):
            bb = p.get("bboxes")
            cf = p.get("conf")
            cl = p.get("cls")
            if bb is None or bb.numel() == 0:
                out.append(th.zeros((0, 6)))
                continue
            out.append(th.cat([bb.float(), cf.view(-1, 1).float(), cl.view(-1, 1).float()], dim=-1))
        else:
            out.append(p if p.numel() > 0 else th.zeros((0, 6)))
    return out
