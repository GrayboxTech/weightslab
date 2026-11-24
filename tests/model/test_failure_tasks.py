"""Failure analysis beyond classification: how each task scores a sample
(segmentation IoU, detection F1, next-token accuracy, a registered per-sample
loss), which task a model is, and full sessions on small planted failures:
objects drawn dim in the test split, so a model trained on bright ones
misses them (segmentation, detection), and a language model that never saw
a pattern."""
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from weightslab.components import failure_analysis as fa
from weightslab.components import failure_tasks as ft


def setUpModule():
    global _lock_patch
    # A model an earlier test module left registered (ModelInterface registers
    # itself even with register=False) may sit in audit mode, which turns
    # Tensor.backward into a no-op. pytest's conftest undoes that between
    # modules; the release job runs plain unittest, so start clean here too.
    from weightslab.backend import ledgers
    ledgers.GLOBAL_LEDGER.unregister_model()
    _lock_patch = mock.patch(
        "weightslab.components.global_monitoring.weightslab_rlock", threading.RLock())
    _lock_patch.start()


def tearDownModule():
    _lock_patch.stop()


# =============================================================================
# Fixtures: tracked datasets / loaders the session reads
# =============================================================================
class _Tracked:
    def __init__(self, items, split, first_id):
        self.items = items                      # (input, target)
        self._dataset_split = split
        self._ids = [str(first_id + i) for i in range(len(items))]
        self.unique_ids = np.array(self._ids)

    def get_index_from_sample_id(self, sid):
        return self._ids.index(str(sid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y = self.items[i]
        return x, self._ids[i], y


class _DL:
    def __init__(self, collate):
        self.collate_fn = collate
        self.batch_size = 16


class _Loader:
    def __init__(self, items, split, first_id, collate=None):
        self.tracked_dataset = _Tracked(items, split, first_id)
        self._dl = _DL(collate) if collate else None

    def get_dataloader(self):
        return self._dl


def _loaders(train, val, test, collate=None):
    return {"train_loader": _Loader(train, "train", 0, collate),
            "val_loader": _Loader(val, "val", 10000, collate),
            "test_loader": _Loader(test, "test", 20000, collate)}


FAST = {"auto_delay": 0.0, "repair_max_steps": 60, "repair_min_steps": 30, "shuffles": 60,
        "repair_lr": 5e-3, "thumb": 16, "max_thumbs": 8, "pool": 600, "ref_samples": 128}


def _run(net, loaders, ids, options=None, timeout=300):
    manager = fa.FailureAnalysisManager()
    session = manager.start(net, loaders, sample_ids=ids, automatic=True,
                            options={**FAST, **(options or {})})
    session.thread.join(timeout=timeout)
    return session


def _captions(session):
    """Every caption on the session's cards."""
    import json
    out = []
    for m in session.messages:
        card = json.loads(m["card_json"]) if m["card_json"] else {}
        for item in card.get("items") or []:
            out.append(item.get("caption", ""))
        for row in card.get("rows") or []:
            if isinstance(row, dict):
                for item in [row.get("failure")] + list(row.get("lookalikes") or []) + list(row.get("items") or []):
                    if isinstance(item, dict):
                        out.append(item.get("caption", ""))
    return out


def _transcript(session):
    return session.message + "\n" + "\n".join(f"[{m['step']}/{m['role']}] {m['text']}" for m in session.messages)


def _train(net, xs, ys, loss_fn, steps=150, lr=1e-2, seed=0):
    g = th.Generator().manual_seed(seed)
    opt = th.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        idx = th.randint(0, len(xs), (32,), generator=g)
        loss = loss_fn(net(xs[idx]), [ys[i] for i in idx] if isinstance(ys, list) else ys[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


# =============================================================================
# Scoring, task by task
# =============================================================================
class TestScores(unittest.TestCase):
    def test_segmentation_iou_ignores_void_and_reads_onehot(self):
        task = ft.SegmentationTask()
        y = th.zeros(1, 4, 4, dtype=th.long)
        y[0, :2] = 1                                          # top half class 1
        logits = th.zeros(1, 2, 4, 4)
        logits[0, 1] = -5.0
        logits[0, 1, :1] = 5.0                                # predicts only the first row as 1
        ev = task.evaluate(logits, y, 1)[0]
        # class 1: inter 4, union 8 -> 0.5; class 0: inter 8, union 12 -> 0.667
        self.assertAlmostEqual(ev["score"], (0.5 + 8 / 12) / 2, places=5)
        self.assertEqual(ev["parts"][1], (4.0, 8.0))
        y2 = y.clone()
        y2[0, 1] = 255                                        # the wrong row is void
        self.assertAlmostEqual(task.evaluate(logits, y2, 1)[0]["score"], 1.0, places=5)
        onehot = F.one_hot(y, 2).permute(0, 3, 1, 2).float()
        self.assertAlmostEqual(task.evaluate(logits, onehot, 1)[0]["score"], ev["score"], places=5)
        # A target twice the output's size: the logits are resized to it.
        big = y.repeat_interleave(2, 1).repeat_interleave(2, 2)
        self.assertAlmostEqual(task.evaluate(logits, big, 1)[0]["score"], ev["score"], places=2)
        mark = task.marks(logits, y, 1)[0]
        self.assertEqual(mark["kind"], "mask")
        self.assertEqual(int(mark["error"].sum()), 4)

    def test_language_model_counts_tokens_and_shifts_hf_labels(self):
        task = ft.LanguageModelTask()
        y = th.tensor([[1, 2, 3, 0]])
        logits = F.one_hot(th.tensor([[1, 2, 0, 0]]), 4).float() * 5
        ev = task.evaluate(logits, y, 1, th.tensor([[0, 1, 2, 3]]))[0]
        self.assertAlmostEqual(ev["score"], 0.75)
        self.assertEqual(ev["miss"], ("#3", "#0"))
        self.assertIn("75% tokens right", ev["caption"])
        # -100 is ignored.
        y_ign = th.tensor([[1, 2, -100, 0]])
        self.assertAlmostEqual(ft.LanguageModelTask().evaluate(logits, y_ign, 1, None)[0]["score"], 1.0)
        # Labels equal to the inputs (Hugging Face): position t predicts t + 1.
        hf = ft.LanguageModelTask()
        ids = th.tensor([[0, 1, 2, 3]])
        logits_hf = F.one_hot(th.tensor([[1, 2, 3, 3]]), 4).float() * 5
        self.assertAlmostEqual(hf.evaluate(logits_hf, ids, 1, ids)[0]["score"], 1.0)
        self.assertTrue(hf.shift)
        loss = hf.loss(logits_hf.requires_grad_(), ids, ids)
        self.assertLess(float(loss.detach()), 0.1)

    def test_detection_f1_matches_class_and_scale(self):
        task = ft.DetectionTask(iou=0.5, conf=0.25)
        gt = [th.tensor([[0.0, 0.0, 0.5, 0.5, 0, 1], [0.5, 0.5, 1.0, 1.0, 1, 1]])]
        pred = [th.tensor([[0.0, 0.0, 0.5, 0.5, 0, 0.9],      # found
                           [0.5, 0.5, 1.0, 1.0, 0, 0.8],      # wrong class: extra + missed
                           [0.2, 0.2, 0.3, 0.3, 0, 0.1]])]    # below the confidence: ignored
        ev = task.evaluate(pred, gt, 1)[0]
        self.assertEqual((ev["tp"], ev["fp"], ev["fn"]), (1, 1, 1))
        self.assertAlmostEqual(ev["score"], 0.5)
        self.assertEqual(ev["parts"], {0: (1.0, 1.0), 1: (0.0, 1.0)})
        self.assertIn("1 missed", ev["caption"])
        # Pixel predictions against normalised targets (a 32 x 32 image).
        pix = [th.tensor([[0.0, 0.0, 16.0, 16.0, 0, 0.9], [16.0, 16.0, 32.0, 32.0, 1, 0.9]])]
        ev = task.evaluate(pix, gt, 1, th.zeros(1, 3, 32, 32))[0]
        self.assertAlmostEqual(ev["score"], 1.0)
        # torchvision-style dicts on both sides.
        tv = [{"boxes": th.tensor([[0.0, 0.0, 0.5, 0.5]]), "labels": th.tensor([2]), "scores": th.tensor([0.7])}]
        tgt = [{"boxes": th.tensor([[0.0, 0.0, 0.5, 0.5]]), "labels": th.tensor([2])}]
        self.assertAlmostEqual(task.evaluate(tv, tgt, 1)[0]["score"], 1.0)
        # An Ultralytics batch: normalised xywh boxes, cls, batch_idx.
        ul = {"bboxes": th.tensor([[0.25, 0.25, 0.5, 0.5]]), "cls": th.tensor([[2.0]]),
              "batch_idx": th.tensor([0.0])}
        self.assertAlmostEqual(task.evaluate(tv, ul, 1)[0]["score"], 1.0)
        mark = task.marks(pred, gt, 1)[0]
        self.assertEqual(len(mark["gt"]), 2)
        self.assertEqual(len(mark["pred"]), 2)

    def test_loss_task_ranks_against_the_reference(self):
        task = ft.LossTask(None, lambda out, y: (out - y).abs().mean(dim=1), "a loss")
        task.set_reference([0.1, 0.2, 0.3, 0.4])
        evs = task.evaluate(th.tensor([[0.0], [0.5]]), th.tensor([[0.05], [0.0]]), 2)
        self.assertAlmostEqual(evs[0]["score"], 1.0)                  # lower than every reference
        self.assertAlmostEqual(evs[1]["score"], 0.0)                  # higher than all
        self.assertTrue(evs[0]["ok"])
        self.assertFalse(evs[1]["ok"])
        self.assertIn("higher than 100% of train / val", evs[1]["caption"])
        with self.assertRaises(ft.NotSupported):
            ft.LossTask(None, lambda out, y: (out - y).abs().mean(), "a loss").evaluate(
                th.zeros(2, 1), th.zeros(2, 1), 2)

    def test_better_and_worse_need_a_real_change(self):
        task = ft.SegmentationTask()
        mk = lambda s: task.finalize({"score": s})
        self.assertTrue(task.better(mk(0.30), mk(0.45)))       # +0.15
        self.assertTrue(task.better(mk(0.47), mk(0.51)))       # crosses the line by enough
        self.assertFalse(task.better(mk(0.495), mk(0.505)))    # noise on the line
        self.assertTrue(task.worse(mk(0.80), mk(0.65)))
        self.assertFalse(task.worse(mk(0.505), mk(0.495)))
        cls = ft.ClassificationTask()
        right = cls.finalize({"label": 1, "pred": 1, "cls_margin": 0.4, "conf": 0.7, "p_true": 0.7, "score": 1.0})
        wrong = cls.finalize({"label": 1, "pred": 0, "cls_margin": -0.4, "conf": 0.7, "p_true": 0.3, "score": 0.0})
        self.assertTrue(cls.better(wrong, right))
        self.assertTrue(cls.worse(right, wrong))
        self.assertFalse(cls.worse(right, right))

    def test_weights_restore_despite_extra_state_dict_entries(self):
        # WeightsLab layers add tracker entries to state_dict() that
        # load_state_dict() refuses (seen live on the NanoGPT backend).
        class _Extra(nn.Linear):
            def _save_to_state_dict(self, destination, prefix, keep_vars):
                super()._save_to_state_dict(destination, prefix, keep_vars)
                destination[prefix + "tracker.count"] = th.zeros(3)

        net = nn.Sequential(_Extra(4, 3), nn.BatchNorm1d(3))
        opt = th.optim.SGD(net.parameters(), lr=0.1)
        with self.assertRaises(RuntimeError):
            net.load_state_dict(net.state_dict())
        snap = fa._weights_of(net)
        weight = net[0].weight
        with th.no_grad():
            weight.add_(1.0)
            net[1].running_mean.add_(2.0)
        fa._load_weights(net, snap)
        self.assertTrue(th.equal(net[0].weight, snap["0.weight"]))
        self.assertTrue(th.equal(net[1].running_mean, snap["1.running_mean"]))
        self.assertIs(net[0].weight, weight)                     # same tensor: the optimizer still holds it
        self.assertIs(opt.param_groups[0]["params"][0], weight)


class TestWhichTask(unittest.TestCase):
    def tearDown(self):
        ft.set_failure_task(None)
        ft.set_failure_boxes_fn(None)
        ft.set_failure_loss_fn(None)

    def _task(self, output, targets, inputs=None, inner=None, datasets=()):
        n = int(ft.logits_of(output).shape[0]) if ft.logits_of(output) is not None else len(targets)
        return ft.detect_task(output, targets, n, inputs, inner, datasets)

    def test_from_the_shapes(self):
        self.assertEqual(self._task(th.zeros(4, 3), th.tensor([0, 1, 2, 0])).name, "classification")
        self.assertEqual(self._task(th.zeros(2, 3, 8, 8), th.zeros(2, 8, 8, dtype=th.long)).name,
                         "segmentation")
        self.assertEqual(self._task(th.zeros(2, 3, 8, 8), th.zeros(2, 1, 16, 16, dtype=th.long)).name,
                         "segmentation")
        tokens = th.zeros(2, 5, dtype=th.long)
        self.assertEqual(self._task(th.zeros(2, 5, 30), tokens, tokens).name, "language_model")

    def test_hints_and_overrides(self):
        class Seg(nn.Module):
            task_type = "segmentation"
        self.assertEqual(self._task(th.zeros(2, 3, 6), th.zeros(2, 6, dtype=th.long), inner=Seg()).name,
                         "segmentation")
        # A wrong hint falls through to the shapes.
        class Cls(nn.Module):
            task_type = "classification"
        self.assertEqual(self._task(th.zeros(2, 3, 8, 8), th.zeros(2, 8, 8, dtype=th.long), inner=Cls()).name,
                         "segmentation")
        ft.set_failure_task(ft.SegmentationTask(ignore_index=0))
        self.assertEqual(self._task(th.zeros(2, 3), th.zeros(2, dtype=th.long)).ignore_index, 0)

    def test_detection_needs_boxes_or_falls_back_to_the_loss(self):
        targets = [th.tensor([[0.0, 0.0, 0.5, 0.5, 0, 1]]), th.zeros(0, 6)]
        raw = th.zeros(2, 1, 4, 4)
        with self.assertRaises(ft.NotSupported):
            self._task(raw, targets)
        ft.set_failure_loss_fn(lambda out, y: out.flatten(1).mean(1))
        task = self._task(raw, targets)
        self.assertEqual(task.name, "loss")
        self.assertTrue(task.boxes)
        ft.set_failure_boxes_fn(lambda out: [th.tensor([[0.0, 0.0, 0.5, 0.5, 0, 0.9]]), th.zeros(0, 6)])
        task = self._task(raw, targets)
        self.assertEqual(task.name, "detection")
        self.assertTrue(task.can_train)

    def test_registered_loss_is_called_without_logging(self):
        calls = []

        class Crit(nn.Module):
            def forward(self, out, y):
                return (out - y).abs().flatten(1).mean(1)

        crit = Crit()
        original = crit.forward

        def wrapped(*a, **kw):
            calls.append(1)
            return original(*a, **kw)
        wrapped.__wrapped__ = original
        crit.forward = wrapped
        with mock.patch("weightslab.backend.ledgers.list_signals", return_value=["train-loss"]), \
                mock.patch("weightslab.backend.ledgers.get_signal", return_value=crit):
            fn, name = ft.user_loss(None)
        self.assertIs(fn, original)
        self.assertIn("train-loss", name)
        self.assertEqual(fn(th.ones(2, 3), th.zeros(2, 3)).tolist(), [1.0, 1.0])
        self.assertEqual(calls, [])


# =============================================================================
# Full sessions on planted failures
# =============================================================================
def _seg_set(n, contrast, g):
    items = []
    for _ in range(n):
        x = th.randn(1, 12, 12, generator=g) * 0.3
        y = th.zeros(12, 12, dtype=th.long)
        r = int(th.randint(1, 9, (1,), generator=g))
        x[0, r:r + 2, 2:10] += contrast
        y[r:r + 2, 2:10] = 1
        items.append((x, y))
    return items


class _SegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.head(self.relu2(self.conv2(self.relu1(self.conv1(x)))))


class TestSegmentationSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.manual_seed(0)
        g = th.Generator().manual_seed(0)
        train, val = _seg_set(400, 1.2, g), _seg_set(100, 1.2, g)
        test = _seg_set(40, 1.2, g) + _seg_set(30, 0.35, g)
        cls.net = _train(_SegNet(), th.stack([x for x, _ in train]), th.stack([y for _, y in train]),
                         lambda out, y: F.cross_entropy(out, y), steps=150)
        cls.state = {k: v.clone() for k, v in cls.net.state_dict().items()}
        cls.loaders = _loaders(train, val, test)
        cls.dim_ids = [str(20040 + i) for i in range(30)]

    def setUp(self):
        self.net.load_state_dict(self.state)

    def test_dim_bars_are_found_and_the_flow_runs_to_the_end(self):
        net, state, loaders, dim_ids = self.net, self.state, self.loaders, self.dim_ids
        session = _run(net, loaders, dim_ids)
        log = _transcript(session)
        self.assertEqual(session.state, "done", log)
        self.assertEqual(session.task.name, "segmentation")
        self.assertGreaterEqual(len(session.failures), 15, log)
        for key in session.failures:
            self.assertLess(session.info[key]["score"], session.task.threshold)
        self.assertTrue(session.lookalikes)
        for key in session.lookalikes:
            self.assertTrue(session.info[key]["ok"])
        steps = {m["step"] for m in session.messages}
        for key in ("check", "lookalikes", "compare", "test", "explain", "fix"):
            self.assertIn(key, steps, log)
        self.assertTrue(any(c.startswith("IoU") for c in _captions(session)), _captions(session))
        # The threshold question was asked (and answered automatically).
        self.assertTrue(any("count as failing" in (m["text"] or "") or m["role"] == "auto"
                            for m in session.messages))
        # Thumbnails carry the wrong pixels.
        import json
        cards = [json.loads(m["card_json"]) for m in session.messages if m["card_json"]]
        first = next(c for c in cards if c["kind"] == "samples")
        self.assertTrue(first["items"][0]["thumb"].startswith("data:image/png;base64,"))
        # The repair table: the gate is IoU, and a repair helps the held-out dim bars more than
        # the control does.
        repair = next(c for c in cards if c["kind"] == "repair")
        self.assertIn("gate IoU", repair["columns"])
        results = {r["name"]: r for r in session.repair_results}
        self.assertIn("control", results)
        best = max((r for r in session.repair_results if r["name"] != "control"),
                   key=lambda r: r["probe_fixed"])
        self.assertGreater(best["probe_fixed"], results["control"]["probe_fixed"], log)
        if not any(r["pass"] for r in session.repair_results):
            # Automatic never applies a repair that fails the gate.
            for k, v in net.state_dict().items():
                self.assertTrue(th.equal(v, state[k]), k)

    def test_the_closest_repair_can_be_applied_anyway_by_hand(self):
        manager = fa.FailureAnalysisManager()
        session = manager.start(self.net, self.loaders, sample_ids=self.dim_ids, automatic=False,
                                options={**FAST, "gate_acc_drop": 0.0, "gate_class_drop": 0.0})
        deadline = time.monotonic() + 300
        asked = []
        while session.thread.is_alive() and time.monotonic() < deadline:
            q = session.question
            if session.state == "waiting" and q:
                key = q["id"].rsplit("-", 1)[0]
                asked.append((key, q["text"]))
                if q["multiple"]:
                    session.answer(q["id"], "", q["defaults"])
                elif key == "accept":
                    session.answer(q["id"], "accept", [])
                else:
                    primary = next((c["id"] for c in q["choices"] if c.get("primary")), q["choices"][0]["id"])
                    session.answer(q["id"], primary, [])
            time.sleep(0.02)
        session.thread.join(timeout=5)
        log = _transcript(session)
        self.assertEqual(session.state, "done", log)
        self.assertIn("threshold", [k for k, _ in asked])
        if any(r["pass"] for r in session.repair_results):
            self.skipTest("a repair passed the strict gate: nothing to apply anyway")
        self.assertTrue(any("Apply it anyway" in t for k, t in asked if k == "accept"), asked)
        self.assertIn("Applied.", log)
        changed = any(not th.equal(v, self.state[k]) for k, v in self.net.state_dict().items())
        self.assertTrue(changed)


def _det_set(n, contrast, g):
    items = []
    for _ in range(n):
        x = th.randn(1, 16, 16, generator=g) * 0.3
        i, j = (int(v) for v in th.randint(0, 4, (2,), generator=g))
        x[0, 4 * i:4 * i + 4, 4 * j:4 * j + 4] += contrast
        box = th.tensor([[j / 4, i / 4, (j + 1) / 4, (i + 1) / 4, 0.0, 1.0]])
        items.append((x, box))
    return items


def _det_collate(items):
    return th.stack([it[0] for it in items]), [it[1] for it in items], [it[2] for it in items]


class _DetNet(nn.Module):
    """Objectness per 4 x 4 cell of a 16 x 16 image."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AvgPool2d(4)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        return self.head(self.pool(self.relu2(self.conv2(self.relu1(self.conv1(x))))))

    @staticmethod
    def cell_targets(targets, device):
        grid = th.zeros(len(targets), 1, 4, 4, device=device)
        for b, t in enumerate(targets):
            for row in t:
                grid[b, 0, int(round(float(row[1]) * 4)), int(round(float(row[0]) * 4))] = 1.0
        return grid

    def wl_failure_loss(self, out, targets):
        grid = self.cell_targets(targets, out.device)
        return F.binary_cross_entropy_with_logits(out, grid, reduction="none").flatten(1).mean(1) * 16


class _DetNetWithBoxes(_DetNet):
    def wl_failure_boxes(self, out):
        prob = th.sigmoid(out.detach())[:, 0]
        boxes = []
        for b in range(prob.shape[0]):
            rows = [[j / 4, i / 4, (j + 1) / 4, (i + 1) / 4, 0.0, float(prob[b, i, j])]
                    for i in range(4) for j in range(4) if prob[b, i, j] >= 0.05]
            boxes.append(th.tensor(rows).reshape(-1, 6))
        return boxes


class TestDetectionSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        th.manual_seed(0)
        g = th.Generator().manual_seed(1)
        cls.train, cls.val = _det_set(400, 1.2, g), _det_set(100, 1.2, g)
        cls.test = _det_set(40, 1.2, g) + _det_set(30, 0.3, g)
        cls.net = _DetNetWithBoxes()
        _train(cls.net, th.stack([x for x, _ in cls.train]), [y for _, y in cls.train],
               lambda out, y: cls.net.wl_failure_loss(out, y).mean(), steps=200)
        cls.state = {k: v.clone() for k, v in cls.net.state_dict().items()}
        cls.dim_ids = [str(20040 + i) for i in range(30)]

    def setUp(self):
        self.net.load_state_dict(self.state)

    def test_missed_boxes_are_found_and_scored_by_f1(self):
        session = _run(self.net, _loaders(self.train, self.val, self.test, _det_collate), self.dim_ids)
        log = _transcript(session)
        self.assertEqual(session.state, "done", log)
        self.assertEqual(session.task.name, "detection")
        self.assertGreaterEqual(len(session.failures), 15, log)
        self.assertTrue(any("missed" in c for c in _captions(session)), _captions(session))
        self.assertTrue(session.repair_results, log)
        import json
        repair = next(json.loads(m["card_json"]) for m in session.messages
                      if m["card_json"] and json.loads(m["card_json"])["kind"] == "repair")
        self.assertIn("gate F1", repair["columns"])

    def test_without_a_box_decoder_it_ranks_the_loss(self):
        net = _DetNet()
        net.load_state_dict(self.state)
        session = _run(net, _loaders(self.train, self.val, self.test, _det_collate), self.dim_ids,
                       options={"repair_max_steps": 30, "repair_min_steps": 20})
        log = _transcript(session)
        self.assertEqual(session.state, "done", log)
        self.assertIsInstance(session.task, ft.LossTask)
        self.assertGreaterEqual(len(session.failures), 15, log)
        self.assertTrue(any("higher than" in c for c in _captions(session)), _captions(session))
        self.assertTrue(session.repair_results, log)


class _CharLM(nn.Module):
    """A bigram model: the next character from the current one."""

    def __init__(self, vocab):
        super().__init__()
        self.itos = list(vocab)
        self.embed = nn.Embedding(len(vocab), 16)
        self.fc1 = nn.Linear(16, 32)
        self.relu = nn.ReLU()
        self.lm_head = nn.Linear(32, len(vocab))

    def forward(self, ids):
        return self.lm_head(self.relu(self.fc1(self.embed(ids))))


def _windows(text, stoi, count, start, stride, block=8):
    ids = th.tensor([stoi[c] for c in text])
    out = []
    for k in range(count):
        p = (start + k * stride) % (len(ids) - block - 1)
        out.append((ids[p:p + block], ids[p + 1:p + block + 1]))
    return out


class TestLanguageModelSession(unittest.TestCase):
    def test_unseen_continuations_are_found_and_the_flow_runs(self):
        th.manual_seed(0)
        vocab = "abcdefgh"
        stoi = {c: i for i, c in enumerate(vocab)}
        common = "abcdefgh" * 40
        rare = "aceg" * 40                                  # a -> c never seen in training
        train = _windows(common, stoi, 300, 0, 3)
        val = _windows(common, stoi, 80, 1, 5)
        test = _windows(common, stoi, 30, 2, 7) + _windows(rare, stoi, 20, 0, 3)
        net = _CharLM(vocab)
        _train(net, th.stack([x for x, _ in train]), th.stack([y for _, y in train]),
               lambda out, y: F.cross_entropy(out.transpose(1, 2), y), steps=150)
        rare_ids = [str(20030 + i) for i in range(20)]
        session = _run(net, _loaders(train, val, test), rare_ids)
        log = _transcript(session)
        self.assertEqual(session.state, "done", log)
        self.assertEqual(session.task.name, "language_model")
        self.assertGreaterEqual(len(session.failures), 15, log)
        self.assertTrue(any("tokens right" in c for c in _captions(session)), _captions(session))
        self.assertIn("→", " ".join(session.info[k]["caption"] for k in session.failures))
        import json
        cards = [json.loads(m["card_json"]) for m in session.messages if m["card_json"]]
        first = next(c for c in cards if c["kind"] == "samples")
        self.assertTrue(first["items"][0]["text"])          # the window as text, no thumbnail
        self.assertFalse(first["items"][0]["thumb"])
        self.assertTrue(any(c["kind"] == "repair" for c in cards), log)


if __name__ == "__main__":
    unittest.main()
