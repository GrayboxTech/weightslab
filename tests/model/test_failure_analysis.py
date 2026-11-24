"""Failure analysis ("Why do these fail?"): look-alikes, suspects beating a
shuffled baseline, the silencing test, and a gated repair -- on a planted
failure whose cause is known: training class 1 always carries a corner patch,
so the model leans on it, and test class-0 samples WITH the patch fail."""
import threading
import time
import unittest
from unittest import mock

import numpy as np
import torch as th
import torch.nn as nn

from weightslab.components import failure_analysis as fa


def setUpModule():
    global _lock_patch
    _lock_patch = mock.patch(
        "weightslab.components.global_monitoring.weightslab_rlock", threading.RLock())
    _lock_patch.start()


def tearDownModule():
    _lock_patch.stop()


def _make(n, cls, patch, g, bar=0.3, noise=0.3):
    x = th.randn(n, 1, 10, 10, generator=g) * noise
    rows = (2, 3) if cls == 0 else (6, 7)
    x[:, 0, rows[0]:rows[1] + 1, 1:9] += bar
    if patch:
        x[:, 0, 0:2, 8:10] += 1.5
    return x


class _Tracked:
    def __init__(self, xs, ys, split, first_id):
        self.xs, self.ys = xs, ys
        self._dataset_split = split
        # A numpy array, as in a real tracked dataset (never `ids or []`).
        self._ids = [str(first_id + i) for i in range(len(xs))]
        self.unique_ids = np.array(self._ids)

    def get_index_from_sample_id(self, sid):
        return self._ids.index(str(sid))

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], self._ids[i], int(self.ys[i])


class _Loader:
    def __init__(self, xs, ys, split, first_id):
        self.tracked_dataset = _Tracked(xs, ys, split, first_id)

    def get_dataloader(self):
        return None


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc(self.pool(self.relu2(self.conv2(self.relu1(self.conv1(x))))).flatten(1))


def _planted():
    """(trained model, loaders, failing test ids)."""
    th.manual_seed(0)
    g = th.Generator().manual_seed(0)
    xtr = th.cat([_make(300, 0, False, g), _make(300, 1, True, g)])
    ytr = th.cat([th.zeros(300), th.ones(300)]).long()
    net = _Net()
    opt = th.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(200):
        idx = th.randint(0, len(xtr), (32,), generator=g)
        loss = nn.functional.cross_entropy(net(xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    xva = th.cat([_make(100, 0, False, g), _make(100, 1, True, g)])
    yva = th.cat([th.zeros(100), th.ones(100)]).long()
    xte = th.cat([_make(60, 0, False, g), _make(60, 1, True, g), _make(40, 0, True, g)])
    yte = th.cat([th.zeros(60), th.ones(60), th.zeros(40)]).long()
    loaders = {
        "train_loader": _Loader(list(xtr), ytr, "train", 0),
        "val_loader": _Loader(list(xva), yva, "val", 10000),
        "test_loader": _Loader(list(xte), yte, "test", 20000),
    }
    with th.no_grad():
        pred = net(xte).argmax(1)
    failing = [str(20000 + i) for i in range(len(xte)) if pred[i] != yte[i]]
    return net, loaders, failing


FAST = {"auto_delay": 0.0, "repair_max_steps": 80, "repair_min_steps": 40, "shuffles": 100, "repair_lr": 5e-3,
        "thumb": 16, "max_thumbs": 8}


def _run(manager, net, loaders, ids, automatic=True, options=None, answers=None, timeout=240):
    session = manager.start(net, loaders, sample_ids=ids, automatic=automatic,
                            options={**FAST, **(options or {})})
    deadline = time.monotonic() + timeout
    while session.thread.is_alive() and time.monotonic() < deadline:
        if not automatic and session.state == "waiting" and session.question:
            q = session.question
            pick = (answers or {}).get(q["id"].rsplit("-", 1)[0])
            if q["multiple"]:
                session.answer(q["id"], "", pick if pick is not None else q["defaults"])
            else:
                primary = next((c["id"] for c in q["choices"] if c.get("primary")), q["choices"][0]["id"])
                session.answer(q["id"], pick or primary, [])
        time.sleep(0.05)
    session.thread.join(timeout=5)
    return session


class TestPlantedFailure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net, cls.loaders, cls.failing = _planted()
        cls._state = {k: v.clone() for k, v in cls.net.state_dict().items()}

    def setUp(self):
        self.net.load_state_dict(self._state)

    def test_finds_the_patch_and_repairs_it(self):
        self.assertGreaterEqual(len(self.failing), 20, "the planted failure must fail")
        manager = fa.FailureAnalysisManager()
        session = _run(manager, self.net, self.loaders, self.failing)
        self.assertEqual(session.state, "done", session.message + "\n" +
                         "\n".join(m["text"] for m in session.messages))
        # Look-alikes: class 0, predicted right, from train / val.
        self.assertTrue(session.lookalikes)
        for key in session.lookalikes:
            info = session.info[key]
            self.assertEqual(info["label"], 0)
            self.assertEqual(info["pred"], 0)
            self.assertIn(key[1], ("train", "val"))
        # The mechanism: silencing a few conv neurons fixes the failures.
        self.assertIsNotNone(session.confirmed, "\n".join(m["text"] for m in session.messages))
        self.assertIn(session.confirmed["cand"]["node"], ("conv1", "conv2"))
        result = session.confirmed["result"]
        self.assertGreaterEqual(len(result["fixed"]), len(session.analysed) // 2)
        # Repair: the best candidate passes the gate and was applied.
        best = [r for r in session.repair_results if r["pass"]]
        self.assertTrue(best, session.repair_results)
        test = self.loaders["test_loader"].tracked_dataset
        idx = [test.get_index_from_sample_id(s) for s in self.failing]
        with th.no_grad():
            pred = self.net(th.stack([test.xs[i] for i in idx])).argmax(1)
        self.assertGreater(int((pred == 0).sum()), len(idx) // 2)
        val = self.loaders["val_loader"].tracked_dataset
        with th.no_grad():
            acc = float((self.net(th.stack(val.xs)).argmax(1) == val.ys).float().mean())
        self.assertGreaterEqual(acc, 0.97)
        # The conversation: every step spoke, automatic answers are marked.
        steps = {m["step"] for m in session.messages}
        for key in ("check", "lookalikes", "compare", "test", "explain", "fix"):
            self.assertIn(key, steps)
        self.assertTrue(any(m["role"] == "auto" for m in session.messages))

    def test_manual_mode_waits_and_stops(self):
        manager = fa.FailureAnalysisManager()
        session = _run(manager, self.net, self.loaders, self.failing, automatic=False,
                       answers={"lookalikes": "stop"})
        self.assertEqual(session.state, "done")
        self.assertEqual(session.step, "lookalikes")
        self.assertTrue(any(m["role"] == "user" for m in session.messages))
        # The model is untouched.
        for k, v in self.net.state_dict().items():
            self.assertTrue(th.equal(v, self._state[k]), k)

    def test_cancel_during_the_repair_rolls_back(self):
        manager = fa.FailureAnalysisManager()
        session = manager.start(self.net, self.loaders, sample_ids=self.failing, automatic=True,
                                options={**FAST, "repair_max_steps": 400, "repair_min_steps": 400})
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline and not session.step_label.startswith("Repair 2"):
            time.sleep(0.05)
        session.cancel.set()
        session.thread.join(timeout=30)
        self.assertEqual(session.state, "cancelled")
        for k, v in self.net.state_dict().items():
            self.assertTrue(th.equal(v, self._state[k]), k)
        from weightslab.components.global_monitoring import pause_controller
        self.assertEqual(pause_controller.held_reason(), "")


class TestHelpers(unittest.TestCase):
    def test_classify_reads_logits_onehot_and_binary(self):
        out = th.tensor([[2.0, 0.0], [0.0, 3.0]])
        res = fa.classify(out, th.tensor([[1.0, 0.0], [1.0, 0.0]]), 2)
        self.assertEqual(res["label"].tolist(), [0, 0])
        self.assertEqual(res["pred"].tolist(), [0, 1])
        self.assertLess(res["margin"][1], 0)
        res = fa.classify(th.tensor([[2.0], [-2.0]]), th.tensor([1, 0]), 2)
        self.assertEqual(res["pred"].tolist(), [1, 0])

    def test_a_segmentation_output_is_refused(self):
        with self.assertRaises(fa.NotClassification):
            fa.classify(th.zeros(2, 3, 8, 8), th.zeros(2, 8, 8), 2)

    def test_first_jump_is_where_the_difference_starts(self):
        self.assertEqual(fa.first_jump([0.1, 0.2, 1.1, 1.5, 1.4]), 2)
        self.assertEqual(fa.first_jump([0.05, 0.1]), -1)

    def test_shuffled_baseline_rejects_chance(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 500))
        b = rng.normal(size=(200, 500))
        t = fa.null_threshold(a, b, 100, 0.95, np.random.default_rng(1))
        d = fa._cohen_d(a, b)
        self.assertGreater(t, 0.5)                        # chance alone gives large |d|
        self.assertLess(float(np.mean(np.abs(d) > t)), 0.01)
        a[:, 7] += 3.0
        self.assertGreater(abs(fa._cohen_d(a, b)[7]), t)  # a real difference beats it

    def test_service_converts_a_question(self):
        from weightslab.trainer.services.model_service import ModelService
        status = {"session_id": "s", "state": "waiting", "message": "", "step": "check",
                  "step_label": "Check", "done": 1, "total": 2, "automatic": False,
                  "messages": [{"seq": 1, "role": "agent", "step": "check", "text": "hi", "card_json": ""}],
                  "question": {"id": "check-1", "text": "Go?", "multiple": False, "defaults": [],
                               "confirm_label": "", "choices": [{"id": "go", "label": "Go", "primary": True}]},
                  "last_seq": 1, "title": "t", "steps": ["check"], "step_titles": ["Check"], "step_index": 0}
        pb = ModelService._failure_state_pb(status)
        self.assertEqual(pb.state, "waiting")
        self.assertEqual(pb.question.choices[0].id, "go")
        self.assertTrue(pb.question.choices[0].primary)
        self.assertEqual(pb.messages[0].text, "hi")


if __name__ == "__main__":
    unittest.main()
