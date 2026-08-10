"""Tests for the four standalone integrations of the four-way SDK approach.

Docs: ``docs/four_way_approach.rst`` and its four subpages promise that each level
runs on its own, driven from the CLI. The bundled examples that back those pages
live in ``weightslab/examples/PyTorch/wl-standalone-{model,data,config,logger}``.

Two things are checked here:

``TestStandaloneExampleFiles``
    every example is present, imports, parses its arguments, and — by walking its
    AST for ``wl.watch_or_edit(..., flag=...)`` calls — registers *only* its own
    level. That is the machine-checkable form of "the levels are independent".

``TestStandaloneCliSurfaces``
    each level is registered in-process exactly as its example does, a real CLI
    server is started, and the commands the docs list for that level are sent over
    the socket and their answers asserted.

The CLI tests use a small MNIST-shaped synthetic dataset instead of torchvision's
MNIST so they stay hermetic (no download) while exercising the same code paths;
the examples themselves run on real MNIST.
"""

import ast
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import Dataset

import weightslab as wl
import weightslab.backend.cli as cli_backend
from weightslab.backend.ledgers import GLOBAL_LEDGER, resolve_hp_name


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "weightslab" / "examples" / "PyTorch"

# level key -> (example directory, flags that level is allowed to register)
STANDALONE_EXAMPLES = {
    "model": ("wl-standalone-model", {"model", "optimizer"}),
    "data": ("wl-standalone-data", {"data"}),
    "config": ("wl-standalone-config", {"hyperparameters"}),
    "logger": ("wl-standalone-logger", {"loss", "metric"}),
}


def _load_example(dir_name: str):
    """Import an example's main.py by path (the directories are not packages)."""
    import importlib.util

    main_py = EXAMPLES_DIR / dir_name / "main.py"
    spec = importlib.util.spec_from_file_location(f"wl_example_{dir_name}", main_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registered_flags(dir_name: str) -> set:
    """The ``flag=`` literals of every ``watch_or_edit`` call in an example.

    Read from the AST, so the ``flag="data"`` mentions inside docstrings and
    comments (which describe what the example does *not* register) don't count.
    """
    source = (EXAMPLES_DIR / dir_name / "main.py").read_text(encoding="utf-8")
    flags = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "watch_or_edit":
            continue
        for keyword in node.keywords:
            if keyword.arg == "flag" and isinstance(keyword.value, ast.Constant):
                flags.add(keyword.value.value)
    return flags


class MnistShapedDataset(Dataset):
    """``(1, 28, 28)`` tensors with deterministic labels — MNIST's shape, no download."""

    def __init__(self, size: int = 32, num_classes: int = 10):
        generator = torch.Generator().manual_seed(0)
        self.images = torch.rand(size, 1, 28, 28, generator=generator)
        self.labels = torch.arange(size) % num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], int(self.labels[index])


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x):
        return self.fc1(self.flatten(self.pool1(self.relu1(self.conv1(x)))))


class TestStandaloneExampleFiles(unittest.TestCase):
    """The four bundled examples exist, import, and stay single-level."""

    def test_every_example_is_present(self):
        for level, (dir_name, _) in STANDALONE_EXAMPLES.items():
            main_py = EXAMPLES_DIR / dir_name / "main.py"
            self.assertTrue(main_py.is_file(), f"missing {level} example: {main_py}")

    def test_examples_import_and_parse_default_args(self):
        for level, (dir_name, _) in STANDALONE_EXAMPLES.items():
            with self.subTest(level=level):
                module = _load_example(dir_name)
                self.assertTrue(hasattr(module, "main"))
                args = module.parse_args([])
                # Every example must be runnable with no arguments at all
                # (`weightslab start example ...` passes none) and must expose the
                # switches the docs use to keep a run headless/bounded.
                self.assertFalse(args.no_cli)
                self.assertFalse(args.no_grpc)
                self.assertIsNone(args.serve_timeout)

    def test_each_example_registers_only_its_own_level(self):
        for level, (dir_name, allowed) in STANDALONE_EXAMPLES.items():
            with self.subTest(level=level):
                flags = _registered_flags(dir_name)
                self.assertTrue(flags, f"{dir_name} registers nothing")
                self.assertEqual(
                    flags, allowed,
                    f"{dir_name} should register exactly {sorted(allowed)}, got {sorted(flags)}",
                )

    def test_levels_do_not_overlap(self):
        """No two standalone examples claim the same registration flag."""
        seen = {}
        for level, (_, allowed) in STANDALONE_EXAMPLES.items():
            for flag in allowed:
                self.assertNotIn(flag, seen,
                                 f"flag={flag!r} is claimed by both {seen.get(flag)} and {level}")
                seen[flag] = level


class StandaloneCliTestCase(unittest.TestCase):
    """Base: a clean ledger plus a live CLI server on an ephemeral port."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._saved_env = {
            key: os.environ.get(key)
            for key in ("WEIGHTSLAB_CLI_FILE", "WEIGHTSLAB_ROOT_LOG_DIR", "CLI_PORT", "CLI_HOST")
        }
        # Keep the discovery file and experiment artifacts out of the real HOME.
        os.environ["WEIGHTSLAB_CLI_FILE"] = os.path.join(self._tmpdir, "cli.json")
        os.environ["WEIGHTSLAB_ROOT_LOG_DIR"] = self._tmpdir
        os.environ.pop("CLI_PORT", None)
        os.environ.pop("CLI_HOST", None)

        self._clear_ledger()

        result = cli_backend.cli_serve(cli_host="127.0.0.1", cli_port=0, spawn_client=False)
        self.assertTrue(result["ok"], f"CLI server did not start: {result}")
        self._host, self._port = result["host"], result["port"]

    def tearDown(self):
        sock = getattr(cli_backend, "_server_sock", None)
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        thread = getattr(cli_backend, "_server_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        cli_backend._server_sock = None
        cli_backend._server_thread = None

        self._clear_ledger()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _clear_ledger(self):
        for registry in ("_models", "_dataloaders", "_optimizers", "_hyperparams",
                         "_dataframes", "_checkpoint_managers", "_loggers", "_signals",
                         # The weak registries are listed by list_*() too, so a
                         # leftover there would still advertise another test's level.
                         "_models_weak", "_dataloaders_weak", "_optimizers_weak",
                         "_dataframes_weak", "_checkpoint_managers_weak",
                         "_hyperparams_weak", "_loggers_weak", "_signals_weak"):
            store = getattr(GLOBAL_LEDGER, registry, None)
            if store is not None and hasattr(store, "clear"):
                store.clear()
        for registry in ("_proxies_models", "_proxies_dataloaders", "_proxies_optimizers",
                         "_proxies_hyperparams", "_proxies_dataframes",
                         "_proxies_checkpoint_managers", "_proxies_loggers"):
            store = getattr(GLOBAL_LEDGER, registry, None)
            if isinstance(store, dict):
                store.clear()
        for name, watcher in list(getattr(GLOBAL_LEDGER, "_hp_watchers", {}).items()):
            try:
                watcher["stop_event"].set()
                watcher["thread"].join(timeout=1.0)
            except Exception:
                pass
            GLOBAL_LEDGER._hp_watchers.pop(name, None)

    def cli(self, command: str) -> dict:
        """Send one command to the CLI server and return its JSON answer."""
        with socket.create_connection((self._host, self._port), timeout=10) as sock:
            stream = sock.makefile("rwb")

            # The server greets every new connection before answering commands.
            greeting = json.loads(stream.readline().decode("utf8"))
            self.assertEqual(greeting.get("welcome"), "weightslab-cli", greeting)

            stream.write((command + "\n").encode("utf8"))
            stream.flush()
            line = stream.readline()
            self.assertTrue(line, f"no answer from the CLI server for {command!r}")
            return json.loads(line.decode("utf8"))


class TestModelLevelCli(StandaloneCliTestCase):
    """Level 1 — model interaction, nothing else registered."""

    def setUp(self):
        super().setUp()
        self.model = wl.watch_or_edit(SmallCNN(), flag="model", device="cpu")
        self.optimizer = wl.watch_or_edit(
            optim.Adam(self.model.parameters(), lr=1e-3), flag="optimizer")

    def test_status_reports_the_model_and_no_other_level(self):
        answer = self.cli("status")
        self.assertTrue(answer["ok"], answer)
        snapshot = answer["snapshot"]
        self.assertTrue(snapshot["models"], "the wrapped model should be listed")
        self.assertTrue(snapshot["optimizers"], "the wrapped optimizer should be listed")
        self.assertEqual(snapshot["dataloaders"], [],
                         "model-only integration must not register loaders")

    def test_list_models(self):
        answer = self.cli("list_models")
        self.assertTrue(answer["ok"], answer)
        self.assertTrue(answer["models"])

    def test_list_optimizers(self):
        answer = self.cli("list_optimizers")
        self.assertTrue(answer["ok"], answer)
        self.assertTrue(answer["optimizers"])

    def test_plot_model_returns_the_architecture(self):
        answer = self.cli("plot_model")
        self.assertTrue(answer["ok"], answer)
        self.assertIn("conv1", answer["plot"])
        self.assertGreater(answer["line_count"], 1)

    def test_pause_and_resume(self):
        paused = self.cli("pause")
        self.assertTrue(paused["ok"], paused)
        self.assertEqual(paused["action"], "paused")

        resumed = self.cli("resume")
        self.assertTrue(resumed["ok"], resumed)
        self.assertEqual(resumed["action"], "resumed")

    def _train_one_step(self):
        criterion = nn.CrossEntropyLoss()
        dataset = MnistShapedDataset(size=4)
        images = torch.stack([dataset[i][0] for i in range(4)])
        labels = torch.tensor([dataset[i][1] for i in range(4)])
        with wl.guard_training_context:
            self.optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
        return loss

    def test_training_writes_history_without_a_wrapped_loss(self):
        """The model level logs its own signals, so a model-only run is not silent.

        Nothing else writes to the logger until a loss/metric is wrapped (the logger
        level), so without this the UI, the report and write_history() would all be
        empty for a model-only integration.
        """
        self.cli("resume")
        for _ in range(3):
            self._train_one_step()

        history = wl.write_history(
            path=os.path.join(self._tmpdir, "history.csv"), format="csv")
        self.assertTrue(os.path.exists(history), history)

        rows = [line for line in Path(history).read_text(encoding="utf-8").splitlines()[1:]
                if line.strip()]
        self.assertTrue(any("model/grad_norm" in row for row in rows),
                        f"model/grad_norm missing from history:\n{rows}")
        self.assertTrue(any("model/parameters" in row for row in rows),
                        f"model/parameters missing from history:\n{rows}")

        steps = {row.split(",")[3] for row in rows if "model/grad_norm" in row}
        self.assertGreater(len(steps), 1,
                           f"every model signal landed on the same step: {steps}")

    def test_model_signal_logging_can_be_switched_off(self):
        """``log_model_signals=False`` is the opt-out for the per-step overhead."""
        # self.model is the ledger proxy; the flag lives on the wrapper it holds.
        wrapper = self.model.get() if hasattr(self.model, "get") else self.model
        wrapper._log_model_signals = False

        self.cli("resume")
        for _ in range(3):
            self._train_one_step()

        history = wl.write_history(
            path=os.path.join(self._tmpdir, "history_off.csv"), format="csv")
        content = Path(history).read_text(encoding="utf-8") if os.path.exists(history) else ""
        self.assertNotIn("model/grad_norm", content)

    def test_a_guarded_training_step_runs_without_the_other_levels(self):
        """The model level alone must be able to leave the paused state.

        resume() only waits on hashes of levels that were actually registered, so
        a model-only run reaches its first guarded step instead of blocking.
        """
        self.cli("resume")
        self.assertFalse(
            wl.guard_training_context.pause_controller.is_paused()
            if hasattr(wl.guard_training_context, "pause_controller")
            else False,
            "training must not stay paused",
        )

        criterion = nn.CrossEntropyLoss()
        images, labels = MnistShapedDataset(size=4)[0]
        images = images.unsqueeze(0)
        labels = torch.tensor([labels])
        with wl.guard_training_context:
            self.optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            loss.backward()
            self.optimizer.step()
        self.assertTrue(torch.isfinite(loss.detach()))


class TestDataLevelCli(StandaloneCliTestCase):
    """Level 2 — data exploration, no model/optimizer/signal registered."""

    def setUp(self):
        super().setUp()
        self.train_loader = wl.watch_or_edit(
            MnistShapedDataset(size=16),
            flag="data",
            loader_name="train_loader",
            batch_size=4,
            shuffle=False,
            is_training=True,
            compute_hash=False,
            preload_labels=True,
            preload_metadata=False,
            root_log_dir=self._tmpdir,
        )
        self.uids = [str(uid) for uid in self.train_loader.wrapped_dataset.unique_ids]

    def test_list_loaders_reports_the_split(self):
        answer = self.cli("list_loaders")
        self.assertTrue(answer["ok"], answer)
        self.assertIn("train_loader", answer["loaders"])

    def test_status_has_no_model_registered(self):
        answer = self.cli("status")
        self.assertTrue(answer["ok"], answer)
        self.assertEqual(answer["snapshot"]["models"], [],
                         "data-only integration must not register a model")
        self.assertIn("train_loader", answer["snapshot"]["dataloaders"])

    def test_list_uids_returns_real_sample_ids(self):
        answer = self.cli("list_uids train_loader --limit 5")
        self.assertTrue(answer["ok"], answer)
        rows = answer["uids"]["train_loader"]
        self.assertEqual(len(rows), 5)
        listed = {str(row["uid"]) for row in rows}
        self.assertTrue(listed.issubset(set(self.uids)),
                        f"list_uids returned ids outside the dataset: {listed}")
        for row in rows:
            self.assertIn("discarded", row)
            self.assertIn("tags", row)

    def test_add_tag_then_query_it_back(self):
        target = self.uids[1]
        answer = self.cli(f"add_tag {target} hard_examples")
        self.assertTrue(answer["ok"], answer)

        tagged = {str(sid) for sid in wl.get_samples_by_tag("hard_examples",
                                                            origin="train_loader")}
        self.assertIn(str(target), tagged)

    def test_discard_and_undiscard_round_trip(self):
        target = self.uids[0]

        answer = self.cli(f"discard {target}")
        self.assertTrue(answer["ok"], answer)
        discarded = {str(sid) for sid in wl.get_discarded_samples(origin="train_loader")}
        self.assertIn(str(target), discarded)

        listed = self.cli("list_uids train_loader --discarded")
        self.assertTrue(listed["ok"], listed)
        self.assertIn(str(target),
                      {str(row["uid"]) for row in listed["uids"]["train_loader"]})

        answer = self.cli(f"undiscard {target}")
        self.assertTrue(answer["ok"], answer)
        discarded = {str(sid) for sid in wl.get_discarded_samples(origin="train_loader")}
        self.assertNotIn(str(target), discarded)

    def test_discarded_samples_leave_the_training_batches(self):
        target = self.uids[0]
        self.cli(f"discard {target}")

        sampled = set()
        for _, ids, _ in self.train_loader:
            sampled.update(str(int(i)) for i in ids)
        self.assertNotIn(str(target), sampled)

    def test_dump_returns_the_loader(self):
        answer = self.cli("dump")
        self.assertTrue(answer["ok"], answer)
        self.assertIn("train_loader", answer["ledger"]["dataloaders"])

    def test_labels_are_readable_for_a_two_tuple_dataset(self):
        """What the studio does when it renders the data grid.

        ``load_label`` used to index ``data[2]`` for any tuple of length <= 3, so a
        plain ``(image, label)`` dataset raised "tuple index out of range" once per
        row and the grid came up label-less.
        """
        from weightslab.data.data_utils import load_label

        tracked = self.train_loader.wrapped_dataset
        for uid in list(tracked.unique_ids)[:5]:
            label = load_label(tracked, uid)
            self.assertIsNotNone(label, f"no label read back for sample {uid}")
            self.assertIn(int(label), range(10))

    def test_write_dataframe_exports_the_curated_subset(self):
        self.cli(f"add_tag {self.uids[2]} hard_examples")
        export = os.path.join(self._tmpdir, "curated.csv")
        written = wl.write_dataframe(path=export, format="csv",
                                    columns=["discarded", "tag:hard_examples"])
        self.assertTrue(os.path.exists(written), written)
        content = Path(written).read_text(encoding="utf-8")
        self.assertIn("tag:hard_examples", content)


class TestConfigLevelCli(StandaloneCliTestCase):
    """Level 3 — config management, nothing else registered."""

    def setUp(self):
        super().setUp()
        self.config_path = Path(self._tmpdir) / "config.yaml"
        self.defaults = {
            "experiment_name": "standalone_config",
            "root_log_dir": self._tmpdir,
            "is_training": True,
            "optimizer": {"lr": 1e-3},
            "data": {"train_loader": {"batch_size": 16}},
        }
        self.config_path.write_text(yaml.safe_dump(self.defaults, sort_keys=False),
                                    encoding="utf-8")
        self.hp = wl.watch_or_edit(str(self.config_path), flag="hyperparameters",
                                   defaults=dict(self.defaults), poll_interval=0.2)

    def test_yaml_path_registration_exposes_the_config(self):
        self.assertEqual(self.hp["optimizer"]["lr"], 1e-3)
        self.assertEqual(self.hp["data"]["train_loader"]["batch_size"], 16)

    def test_hp_lists_and_shows(self):
        listing = self.cli("hp")
        self.assertTrue(listing["ok"], listing)
        self.assertTrue(listing["hyperparams"], "at least one config set is registered")

        # Ask for the set the CLI itself resolves (`hp`'s listing can hold more
        # than one name, and the first is not necessarily the live one).
        name = resolve_hp_name()
        self.assertIn(name, listing["hyperparams"])

        shown = self.cli(f"hp {name}")
        self.assertTrue(shown["ok"], shown)
        self.assertEqual(shown["name"], name)
        self.assertEqual(shown["hyperparams"]["experiment_name"], "standalone_config")

    def test_set_hp_updates_the_live_config(self):
        answer = self.cli("set_hp optimizer.lr 0.0005")
        self.assertTrue(answer["ok"], answer)
        self.assertEqual(answer["key"], "optimizer.lr")
        self.assertEqual(answer["value"], 0.0005)
        self.assertEqual(self.hp["optimizer"]["lr"], 0.0005)

    def test_set_hp_updates_a_nested_data_key(self):
        answer = self.cli("set_hp data.train_loader.batch_size 32")
        self.assertTrue(answer["ok"], answer)
        self.assertEqual(self.hp["data"]["train_loader"]["batch_size"], 32)

    def test_status_shows_config_and_no_other_level(self):
        answer = self.cli("status")
        self.assertTrue(answer["ok"], answer)
        self.assertTrue(answer["snapshot"]["hyperparams"])
        self.assertEqual(answer["snapshot"]["models"], [])
        self.assertEqual(answer["snapshot"]["dataloaders"], [])


class TestLoggerLevelCli(StandaloneCliTestCase):
    """Level 4 — logger and signals, no model/data/config registered."""

    def setUp(self):
        super().setUp()
        self.train_loss = wl.watch_or_edit(
            nn.CrossEntropyLoss(reduction="none"),
            flag="loss", signal_name="train/loss", log=True)
        self.eval_loss = wl.watch_or_edit(
            nn.CrossEntropyLoss(reduction="none"),
            flag="loss", signal_name="eval/loss", log=True)

        # A plain, unwrapped model/loader: the logger level owns the signals only.
        self.model = SmallCNN()
        self.dataset = MnistShapedDataset(size=8)

    def _log_steps(self, steps: int = 4):
        images = torch.stack([self.dataset[i][0] for i in range(4)])
        labels = torch.tensor([self.dataset[i][1] for i in range(4)])
        for step in range(1, steps + 1):
            with wl.guard_training_context:
                # step= is what places the point on the x-axis with no wrapped model.
                self.train_loss(self.model(images), labels, step=step)
        return steps

    def test_status_has_only_the_logger_level(self):
        answer = self.cli("status")
        self.assertTrue(answer["ok"], answer)
        self.assertEqual(answer["snapshot"]["models"], [])
        self.assertEqual(answer["snapshot"]["dataloaders"], [])
        self.assertTrue(answer["snapshot"]["loggers"], "a logger must be registered")

    def test_wrapped_loss_logs_one_point_per_step(self):
        steps = self._log_steps(4)
        history = wl.write_history(
            path=os.path.join(self._tmpdir, "history.csv"), format="csv")
        self.assertTrue(os.path.exists(history), history)

        rows = [line for line in Path(history).read_text(encoding="utf-8").splitlines()[1:]
                if line.strip()]
        self.assertGreaterEqual(
            len(rows), steps - 1,
            f"expected about one row per step, got {len(rows)}:\n{rows}")
        self.assertTrue(any("train/loss" in row for row in rows), rows)

        steps_logged = {row.split(",")[3] for row in rows if "train/loss" in row}
        self.assertGreater(len(steps_logged), 1,
                           f"every point landed on the same step: {steps_logged}")

    def test_report_command_renders_the_history(self):
        self._log_steps(3)
        answer = self.cli("report --no-agent")
        self.assertTrue(answer["ok"], answer)
        self.assertTrue(os.path.exists(answer["path"]), answer)
        self.assertGreaterEqual(answer["signals"], 1, answer)

    def test_pause_and_resume_without_the_other_levels(self):
        self.assertTrue(self.cli("pause")["ok"])
        self.assertTrue(self.cli("resume")["ok"])


def run_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for case in (TestStandaloneExampleFiles, TestModelLevelCli, TestDataLevelCli,
                 TestConfigLevelCli, TestLoggerLevelCli):
        suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
