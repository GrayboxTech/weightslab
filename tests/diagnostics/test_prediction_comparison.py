"""The comparison must never silently pair different evaluation populations."""

import copy

import pytest

import weightslab as wl


@pytest.fixture
def pair():
    control = {"run_id": "control", "parent_checkpoint_sha256": "a" * 64,
               "case_manifest_sha256": "b" * 64, "preprocessing_sha256": "c" * 64,
               "evaluation_split": "test", "training_steps": 250, "optimizer_policy": "fresh_sgd",
               "cases": [{"sample_id": "a", "group": "rare", "label": 1, "prediction": 0,
                          "true_label_margin": -0.2},
                         {"sample_id": "b", "group": "common", "label": 0, "prediction": 0}]}
    intervention = copy.deepcopy(control)
    intervention["run_id"] = "widen"
    intervention["cases"][0].update(prediction=1, true_label_margin=0.3)
    intervention["cases"][1]["prediction"] = 1
    return control, intervention


def test_corrected_regressed_and_order_independence(pair):
    original = copy.deepcopy(pair)
    result = wl.compare_predictions(*pair)
    assert result["corrected_ids"] == ["a"]
    assert result["regressed_ids"] == ["b"]
    assert result["overall"]["accuracy_delta"] == 0
    assert result["groups"]["rare"]["accuracy_delta"] == 1
    assert result["groups"]["common"]["accuracy_delta"] == -1
    assert result["cases"][0]["margin_delta"] == pytest.approx(0.5)
    assert result["cases"][1]["margin_delta"] is None
    assert pair == original
    pair[1]["cases"].reverse()
    assert wl.compare_predictions(*pair) == result


@pytest.mark.parametrize("key,value", [("parent_checkpoint_sha256", "d" * 64),
                                     ("case_manifest_sha256", "d" * 64),
                                     ("preprocessing_sha256", "d" * 64),
                                     ("evaluation_split", "validation"),
                                     ("training_steps", 251), ("optimizer_policy", "keep_state")])
def test_unpaired_provenance(pair, key, value):
    pair[1][key] = value
    with pytest.raises(ValueError, match="Unpaired"):
        wl.compare_predictions(*pair)


@pytest.mark.parametrize("key", ["parent_checkpoint_sha256", "case_manifest_sha256", "run_id",
                               "preprocessing_sha256", "evaluation_split", "training_steps", "optimizer_policy"])
def test_missing_provenance(pair, key):
    del pair[1][key]
    with pytest.raises(ValueError):
        wl.compare_predictions(*pair)


@pytest.mark.parametrize("key,value", [("label", 0), ("group", "different"), ("sample_id", "new")])
def test_changed_cohort(pair, key, value):
    pair[1]["cases"][0][key] = value
    with pytest.raises(ValueError):
        wl.compare_predictions(*pair)


@pytest.mark.parametrize("key,value", [("prediction", True), ("label", -1), ("sample_id", ""),
                                     ("group", None), ("true_label_margin", float("nan")),
                                     ("true_label_margin", float("inf"))])
def test_invalid_case(pair, key, value):
    pair[1]["cases"][0][key] = value
    with pytest.raises(ValueError):
        wl.compare_predictions(*pair)


def test_empty_missing_and_duplicate_cases(pair):
    for cases in ([], pair[1]["cases"][:1], [pair[1]["cases"][0]] * 2):
        changed = copy.deepcopy(pair[1])
        changed["cases"] = cases
        with pytest.raises(ValueError):
            wl.compare_predictions(pair[0], changed)


def test_same_run_rejected(pair):
    with pytest.raises(ValueError, match="distinct"):
        wl.compare_predictions(pair[0], pair[0])


def test_margin_overflow_rejected(pair):
    pair[0]["cases"][0]["true_label_margin"] = -1e308
    pair[1]["cases"][0]["true_label_margin"] = 1e308
    with pytest.raises(ValueError, match="overflowed"):
        wl.compare_predictions(*pair)
