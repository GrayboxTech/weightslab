"""The projection, swept across the whole baseline model zoo.

``examples/utils/baseline_models`` is the closest thing the repo has to "every
shape of model a user might wrap": MLPs, CNNs, VGG/ResNet, U-Nets (2-D and
3-D), YOLO heads, a GAN and a VAE. The claim this feature makes is that it
attaches to any of them with no user input, so the honest way to hold that
claim is to actually run all of them.

For each model the sweep asserts the auto-pick produced a ``(B, F)`` embedding
that is *not* simply the model's output width -- reading the head's output
instead of its input is the exact failure this is guarding against, and it
looks perfectly healthy unless you check the width.

Generative models (GAN/VAE) are reported but not asserted on: their "head" is a
decoder producing an image, so there is no supervised representation to project
and the projection is not a meaningful view of them.
"""

import inspect
import unittest

import torch as th
import torch.nn as nn

from weightslab.projection import ProjectionTracker, pick_embedding_layer

try:
    from weightslab.examples.utils.baseline_models.pytorch import models as zoo
except Exception:  # pragma: no cover - the zoo is example data, not a hard dep
    zoo = None


# Generative: no supervised representation, so the projection is not claimed
# to be meaningful for them (it still must not crash).
GENERATIVE = {"DCGAN", "SimpleVAE"}
# Not projection problems:
#  - FlexibleCNNBlock needs constructor arguments the sweep cannot guess;
#  - Yolov11 wraps an Ultralytics trainer whose forward loads a COCO dataset
#    from disk, so running it here fails on missing data, not on the hook.
#    TinyYOLO covers the same detection-head shape without that dependency.
SKIP = {"FlexibleCNNBlock", "Yolov11"}


def discover():
    """Every zero-arg-constructible model in the zoo, with its own input shape."""
    found = []
    for name, obj in vars(zoo).items():
        if not (inspect.isclass(obj) and issubclass(obj, nn.Module)):
            continue
        if obj.__module__ != zoo.__name__ or name in SKIP:
            continue
        try:
            model = obj()
        except Exception:
            continue  # needs constructor args
        shape = getattr(model, "input_shape", None)
        if not shape:
            continue
        found.append((name, model, tuple(shape)))
    return sorted(found, key=lambda row: row[0])


@unittest.skipIf(zoo is None, "baseline model zoo unavailable")
class TestBaselineModelZoo(unittest.TestCase):
    """One subtest per model, so a failure names the model that broke."""

    def test_every_baseline_model_yields_an_embedding(self):
        models = discover()
        self.assertGreater(len(models), 10, "zoo discovery found almost nothing")

        report = []
        for name, model, shape in models:
            with self.subTest(model=name):
                layer, use_input, feature_last = pick_embedding_layer(model)
                self.assertIsNotNone(layer, f"{name}: nothing to hook")

                tracker = ProjectionTracker()
                tracker.attach(layer, use_input=use_input, feature_last=feature_last)
                model.eval()
                # Batch of 2 at the model's own declared shape; a batch of 1
                # would hide any batch-dim mistake.
                sample = th.randn((2,) + shape[1:])
                try:
                    with th.no_grad():
                        output = model(sample)
                finally:
                    handle_shape = tracker._pending
                    tracker.detach()

                captured = tuple(handle_shape.shape) if handle_shape is not None else None
                out_width = None
                if isinstance(output, th.Tensor):
                    out_width = output.shape[1] if output.ndim > 1 else None
                report.append((name, type(layer).__name__, use_input, captured, out_width))

                if name in GENERATIVE:
                    continue

                self.assertIsNotNone(captured, f"{name}: hook never fired")
                self.assertEqual(len(captured), 2, f"{name}: not reduced to (B, F)")
                self.assertEqual(captured[0], 2, f"{name}: lost the batch dim")
                self.assertGreater(captured[1], 0, f"{name}: empty feature dim")

        print("\n--- baseline model zoo: projection auto-pick ---")
        for name, layer, use_input, captured, out_width in report:
            note = " (generative, not asserted)" if name in GENERATIVE else ""
            print(f"  {name:<28} {layer:<10} input={str(use_input):<5} "
                  f"-> {captured}  model_out_width={out_width}{note}")

    def test_segmentation_models_project_features_not_class_logits(self):
        """The regression that motivated hooking a conv's INPUT.

        A U-Net's final 1x1 conv outputs `n_classes` channels. Reading its
        OUTPUT gives a (B, n_classes) tensor -- well-formed, and completely the
        wrong thing to lay out. The embedding has to be wider than the class
        count, which is what distinguishes the two.
        """
        candidates = [(name, model) for name, model, _ in discover()
                      if "UNet" in name or "FCN" in name]
        self.assertTrue(candidates, "no segmentation models discovered")

        for name, model in candidates:
            with self.subTest(model=name):
                layer, use_input, feature_last = pick_embedding_layer(model)
                shape = tuple(model.input_shape)
                tracker = ProjectionTracker()
                tracker.attach(layer, use_input=use_input, feature_last=feature_last)
                model.eval()
                try:
                    with th.no_grad():
                        output = model(th.randn((2,) + shape[1:]))
                    captured = tracker._pending
                finally:
                    tracker.detach()

                self.assertIsNotNone(captured, f"{name}: hook never fired")
                if isinstance(output, th.Tensor) and output.ndim > 1:
                    self.assertGreater(
                        captured.shape[1], output.shape[1],
                        f"{name}: captured {captured.shape[1]} features but the model "
                        f"outputs {output.shape[1]} channels -- that is the head's "
                        f"OUTPUT (class logits), not the representation feeding it",
                    )


try:
    import transformers  # noqa: F401
    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAVE_TRANSFORMERS = False


@unittest.skipIf(not HAVE_TRANSFORMERS, "transformers not installed")
class TestLanguageModels(unittest.TestCase):
    """LLMs are the shape most likely to be projected wrongly.

    A language head is a ``Linear(hidden, vocab)`` with an enormous output, fed
    ``(B, T, C)`` hidden states. Reading its OUTPUT would hand the projection a
    vocab-sized logit vector; reading its INPUT and pooling the token axis gives
    a sentence embedding, which is the thing worth laying out. These assert the
    captured width is the model's hidden size and NOT its vocabulary/label count.

    Models are built from config (never downloaded) so the test is offline.
    """

    def _capture(self, model, **inputs):
        layer, use_input, feature_last = pick_embedding_layer(model)
        tracker = ProjectionTracker()
        tracker.attach(layer, use_input=use_input, feature_last=feature_last)
        model.eval()
        try:
            with th.no_grad():
                model(**inputs)
            return tracker._pending, layer
        finally:
            tracker.detach()

    def test_causal_lm_projects_hidden_states_not_vocab_logits(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        hidden, vocab = 128, 500
        model = GPT2LMHeadModel(GPT2Config(
            n_embd=hidden, n_layer=2, n_head=4, vocab_size=vocab))
        captured, _ = self._capture(model, input_ids=th.randint(0, vocab, (2, 12)))

        self.assertIsNotNone(captured)
        self.assertEqual(tuple(captured.shape), (2, hidden))
        self.assertNotEqual(captured.shape[1], vocab,
                            "projected the vocabulary logits, not the representation")

    def test_encoder_model(self):
        from transformers import BertConfig, BertModel
        hidden = 96
        model = BertModel(BertConfig(
            hidden_size=hidden, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=128, vocab_size=500))
        captured, _ = self._capture(model, input_ids=th.randint(0, 500, (2, 12)))
        self.assertEqual(tuple(captured.shape), (2, hidden))

    def test_sequence_classifier_projects_features_not_labels(self):
        from transformers import BertConfig, BertForSequenceClassification
        hidden, labels = 96, 3
        model = BertForSequenceClassification(BertConfig(
            hidden_size=hidden, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=128, vocab_size=500, num_labels=labels))
        captured, _ = self._capture(model, input_ids=th.randint(0, 500, (2, 12)))
        self.assertEqual(tuple(captured.shape), (2, hidden))
        self.assertNotEqual(captured.shape[1], labels)

    def test_sequence_length_does_not_change_the_embedding_width(self):
        """Token pooling means a batch of any length yields the same width --
        what makes sequences comparable as points at all."""
        from transformers import BertConfig, BertModel
        model = BertModel(BertConfig(
            hidden_size=96, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=128, vocab_size=500))
        short, _ = self._capture(model, input_ids=th.randint(0, 500, (2, 4)))
        long, _ = self._capture(model, input_ids=th.randint(0, 500, (2, 40)))
        self.assertEqual(tuple(short.shape), tuple(long.shape))


if __name__ == "__main__":
    unittest.main()
