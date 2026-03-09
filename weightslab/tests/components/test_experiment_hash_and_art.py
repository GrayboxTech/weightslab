import unittest
from unittest.mock import patch

import torch as th

from weightslab.components.experiment_hash import ExperimentHashGenerator
from weightslab.data.sample_stats import SampleStatsEx
from weightslab import art


class _TinyModel(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = th.nn.Linear(4, 2)
        self.fc.in_neurons = 4
        self.fc.out_neurons = 2
        self.fc.operation_age = {"prune": 1}

    def forward(self, x):
        return self.fc(x)


class TestExperimentHashGenerator(unittest.TestCase):
    def test_generate_hash_defaults_to_zero_segments(self):
        gen = ExperimentHashGenerator()
        out = gen.generate_hash(model=None, config=None, data_state=None)

        self.assertEqual(out, "000000000000000000000000")
        self.assertEqual(gen.get_last_hash(), out)

    def test_hash_config_ignores_runtime_keys(self):
        gen = ExperimentHashGenerator()
        c1 = {"lr": 1e-3, "root_log_dir": "a", "is_training": True}
        c2 = {"lr": 1e-3, "root_log_dir": "b", "is_training": False}

        self.assertEqual(gen._hash_config(c1), gen._hash_config(c2))

    def test_restore_hashes_from_combined_and_components(self):
        gen = ExperimentHashGenerator()
        combined = "aaaabbbbccccdddd11112222"

        gen.restore_hashes(combined_hash=combined)
        parts = gen.get_component_hashes()
        self.assertEqual(parts["hp"], "aaaabbbb")
        self.assertEqual(parts["model"], "ccccdddd")
        self.assertEqual(parts["data"], "11112222")
        self.assertEqual(parts["combined"], combined)

        gen.restore_hashes(component_hashes={"hp": "12345678", "model": "87654321", "data": "abcdabcd"})
        parts2 = gen.get_component_hashes()
        self.assertEqual(parts2["combined"], "1234567887654321abcdabcd")

    def test_compare_hashes_detects_segment_changes(self):
        gen = ExperimentHashGenerator()
        h1 = "111111112222222233333333"
        h2 = "11111111aaaaaaaabbbbbbbb"
        self.assertEqual(gen.compare_hashes(h1, h2), {"model", "data"})
        self.assertEqual(gen.compare_hashes("short", h2), set())

    def test_has_changed_reports_hp_model_data(self):
        gen = ExperimentHashGenerator()
        model = _TinyModel()
        config = {"lr": 0.01}
        data_state = {
            SampleStatsEx.DISCARDED.value: {"1": False, "2": True},
            SampleStatsEx.TAG.value: {"1": ["a"], "2": ["b"]},
        }

        gen.generate_hash(model=model, config=config, data_state=data_state, model_init_step=0, _last_time_loaded=0.0)
        changed, components = gen.has_changed(
            model=model,
            config=config,
            data_state=data_state,
            model_init_step=0,
            _last_time_loaded=0.0,
        )
        self.assertFalse(changed)
        self.assertEqual(components, set())

        changed2, components2 = gen.has_changed(
            model=model,
            config=config,
            data_state=data_state,
            model_init_step=0,
            force=True,
            _last_time_loaded=0.0,
        )
        self.assertTrue(changed2)
        self.assertEqual(components2, {"hp", "model", "data"})


class TestArtGitInfo(unittest.TestCase):
    def test_get_git_info_returns_values_when_git_found(self):
        with patch("weightslab.art.os.path.abspath", return_value="C:/repo/weightslab/art.py"), \
             patch("weightslab.art.os.path.isdir", side_effect=lambda p: p.replace('\\', '/').endswith('/.git')), \
             patch("weightslab.art.os.path.dirname", side_effect=lambda p: p.rsplit("/", 1)[0] if "/" in p else p), \
             patch("weightslab.art.subprocess.check_output") as check_output:
            check_output.side_effect = [b"main\n", b"deadbeef\n", b"v1.2.3\n"]
            branch, version, commit = art.get_git_info()

        self.assertEqual(branch, "main")
        self.assertEqual(version, "v1.2.3")
        self.assertEqual(commit, "deadbeef")

    def test_get_git_info_returns_none_tuple_on_failure(self):
        with patch("weightslab.art.subprocess.check_output", side_effect=FileNotFoundError()):
            result = art.get_git_info()
        self.assertEqual(result, (None, None, None))


if __name__ == "__main__":
    unittest.main()
