import json
import unittest

import torch
import torch.nn as nn

from weightslab.backend import ledgers
from weightslab.backend.model_interface import ModelInterface


class TestModelGraphApi(unittest.TestCase):
    def tearDown(self):
        ledgers.clear_all()

    @staticmethod
    def _wrapped_model():
        model = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        wrapped = ModelInterface(
            model,
            dummy_input=torch.randn(1, 4),
            compute_dependencies=True,
            register=False,
            skip_previous_auto_load=True,
        )
        # Architecture-change hooks update registered optimizers. These focused
        # API tests do not register one, so avoid unrelated ledger work.
        wrapped._architecture_change_hook_fns = []
        return wrapped

    def test_get_model_graph_returns_serializable_structure(self):
        model = self._wrapped_model()

        graph = model.get_model_graph()

        self.assertEqual(graph["schema_version"], 1)
        self.assertEqual([layer["name"] for layer in graph["layers"]], ["0", "1", "2"])
        self.assertEqual(
            graph["dependencies"],
            [
                {"source_layer_id": 0, "target_layer_id": 1, "type": "SAME"},
                {"source_layer_id": 1, "target_layer_id": 2, "type": "INCOMING"},
            ],
        )
        self.assertNotIn("neurons", graph["layers"][0])
        json.dumps(graph)

    def test_layer_info_and_modifiers_update_the_live_model(self):
        model = self._wrapped_model()

        layer = model.get_layer_info(0)
        self.assertEqual(layer["type"], "Linear")
        self.assertEqual(layer["input_neurons"], 4)
        self.assertEqual(layer["output_neurons"], 3)
        self.assertEqual(len(layer["neurons"]), 3)

        model.freeze_neurons(0, [0])
        self.assertTrue(model.get_layer_info(0)["neurons"][0]["frozen"])

        # Freezing twice is idempotent rather than toggling the state.
        model.freeze_neurons(0, [0])
        self.assertTrue(model.get_layer_info(0)["neurons"][0]["frozen"])
        self.assertEqual(model.get_layer_info(0)["operation_counts"]["FREEZE"], 1)

        model.unfreeze_neurons(0, [0])
        self.assertFalse(model.get_layer_info(0)["neurons"][0]["frozen"])

        model.add_neurons(0, count=2)
        self.assertEqual(model.get_layer_info(0)["output_neurons"], 5)
        self.assertEqual(model.get_layer_info(2)["input_neurons"], 5)
        self.assertEqual(tuple(model(torch.randn(1, 4)).shape), (1, 2))

    def test_modelling_api_rejects_ambiguous_or_invalid_targets(self):
        model = self._wrapped_model()

        with self.assertRaisesRegex(ValueError, "Unknown layer id"):
            model.get_layer_info(99)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            model.prune_neurons(0, [])
        with self.assertRaisesRegex(ValueError, "outside layer 0"):
            model.reset_neurons(0, [3])
        with self.assertRaisesRegex(ValueError, "no learnable weights"):
            model.freeze_neurons(1, [0])
        with self.assertRaisesRegex(ValueError, "strictly between 0 and 1"):
            model.perturb_neurons(0, [0], ratio=1.0)

    def test_freeze_and_unfreeze_reject_untracked_weight(self):
        raw_model = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
        )
        raw_model[0].weight.requires_grad_(False)
        model = ModelInterface(
            raw_model,
            dummy_input=torch.randn(1, 4),
            compute_dependencies=True,
            register=False,
            skip_previous_auto_load=True,
        )
        model._architecture_change_hook_fns = []

        for operation in (model.freeze_neurons, model.unfreeze_neurons):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(
                    ValueError, "trainable, per-neuron tracked weight"
                ):
                    operation(0, [0])

        self.assertEqual(model.get_layer_info(0)["operation_counts"]["FREEZE"], 0)


if __name__ == "__main__":
    unittest.main()
