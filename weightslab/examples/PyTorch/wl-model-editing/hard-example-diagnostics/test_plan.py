"""Standard-library checks for the planning scaffold; no training or downloads."""

import copy
import json
import unittest
from pathlib import Path

from plan import build_plan


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path(__file__).with_name("experiment.json").read_text())

    def test_matrix_is_paired_and_not_executed(self):
        plan = build_plan(self.config)
        self.assertEqual(len(plan["jobs"]), 12)
        self.assertEqual(len({job["run_id"] for job in plan["jobs"]}), 12)
        for seed in self.config["seeds"]:
            jobs = [job for job in plan["jobs"] if job["seed"] == seed]
            self.assertEqual(len(jobs), 4)
            self.assertEqual(len({job["parent_checkpoint_id"] for job in jobs}), 1)
            self.assertEqual({job["training_steps_to_do"] for job in jobs}, {250})
            for job in jobs:
                self.assertEqual(job["status"], "planned_not_executed")
                self.assertIsNone(job["parent_checkpoint_sha256"])
                self.assertIsNone(job["case_manifest_sha256"])

    def test_fingerprint_is_stable_and_tracks_config_changes(self):
        original = copy.deepcopy(self.config)
        first = build_plan(self.config)
        self.assertEqual(first, build_plan(dict(reversed(list(self.config.items())))))
        self.assertEqual(self.config, original)
        self.config["training_steps_to_do"] += 1
        changed = build_plan(self.config)
        self.assertNotEqual(first["config_sha256"], changed["config_sha256"])
        self.assertNotEqual(first["jobs"][0]["run_id"], changed["jobs"][0]["run_id"])

    def test_invalid_seeds(self):
        for seeds in ([], [17, 17], [True], [-1], [1.5], "17"):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                build_plan({**self.config, "seeds": seeds})

    def test_invalid_budgets(self):
        for key in ("baseline_training_steps", "training_steps_to_do"):
            for value in (0, -1, 1.5, True, None):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    build_plan({**self.config, key: value})

    def test_invalid_schema(self):
        for value in (0, 2, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_plan({**self.config, "schema_version": value})

    def test_missing_or_duplicate_control(self):
        arms = self.config["arms"]
        for invalid in ([], arms[1:], [*arms, arms[0]], None):
            with self.subTest(arms=invalid), self.assertRaises(ValueError):
                build_plan({**self.config, "arms": invalid})

    def test_invalid_arm_fields(self):
        for field, value in (("name", ""), ("name", 12), ("add_neurons", -1),
                             ("add_neurons", True), ("add_neurons", 1.5),
                             ("sampling", "test_set_balanced")):
            config = copy.deepcopy(self.config)
            config["arms"][1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                build_plan(config)

    def test_control_cannot_be_an_intervention(self):
        for field, value in (("add_neurons", 16), ("sampling", "group_balanced")):
            config = copy.deepcopy(self.config)
            config["arms"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_plan(config)

    def test_optimizer_policy_must_match(self):
        with self.assertRaises(ValueError):
            build_plan({**self.config, "optimizer_at_fork": "reset_only_edited_arm"})

    def test_nonfinite_config_is_rejected(self):
        self.config["optimizer"]["lr"] = float("nan")
        with self.assertRaises(ValueError):
            build_plan(self.config)


if __name__ == "__main__":
    unittest.main()
