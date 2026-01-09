import pandas as pd
import pytest
import os
import sys
sys.path.append(f"{os.path.dirname(__file__)}")

from agent import DataManipulationAgent
from dataclasses import dataclass
from typing import Optional, Set


# Test DataFrame
def make_test_df() -> pd.DataFrame:
    """Create a small DataFrame matching the real schema for testing."""
    return pd.DataFrame(
        [
            # sample_id, label, image, origin, prediction_age, prediction_loss,
            # prediction_raw, target, deny_listed, encountered, tags, pred, loss/combined
            (1, 0, None, "train", 20, 0.5, None, None, False, 10, {}, None, 0.6),
            (2, 1, None, "train", 25, 1.5, None, None, True,  20, {}, None, 1.6),
            (3, 4, None, "eval",  30, 2.5, None, None, False, 30, {}, None, 2.6),
            (4, 4, None, "train", 35, 3.5, None, None, True,  40, {}, None, 3.6),
            (5, 7, None, "test",  40, 0.9, None, None, False, 50, {}, None, 0.95),
        ],
        columns=[
            "sample_id",
            "label",
            "image",
            "origin",
            "prediction_age",
            "prediction_loss",
            "prediction_raw",
            "target",
            "deny_listed",
            "encountered",
            "tags",
            "pred",
            "loss/combined",
        ],
    )


# Test case spec
@dataclass
class AgentTestCase:
    name: str
    instruction: str

    # structural expectations
    expect_noop: bool = False
    expected_len: Optional[int] = None

    # label expectations
    expected_labels_exact: Optional[Set[int]] = None   # exact set
    allowed_labels: Optional[Set[int]] = None          # subset of
    forbidden_labels: Optional[Set[int]] = None        # must not appear

    # origin expectations
    expected_origin_exact: Optional[str] = None

    # loss expectations
    min_prediction_loss: Optional[float] = None        # strictly greater than this
    max_prediction_loss: Optional[float] = None        # strictly less than or equal

    # sorting expectations
    expect_sorted_by: Optional[str] = None             # column expected to be monotonic


# Easy / Medium / Hard cases
EASY_CASES = [
    AgentTestCase(
        name="keep_label_4",
        instruction="keep only samples with label 4",
        expected_labels_exact={4},
    ),
    AgentTestCase(
        name="keep_label_0_or_1",
        instruction="keep samples where label is 0 or 1",
        allowed_labels={0, 1},
    ),
    AgentTestCase(
        name="keep_loss_gt_2",
        instruction="keep samples with loss greater than 2",
        min_prediction_loss=2.0,
    ),
    AgentTestCase(
        name="keep_origin_train",
        instruction="keep only samples from origin 'train'",
        expected_origin_exact="train",
    ),
    AgentTestCase(
        name="sort_by_label",
        instruction="sort by label",
        expect_sorted_by="label",
    ),
    AgentTestCase(
        name="sort_by_loss_desc",
        instruction="sort by loss descending",
        expect_sorted_by="prediction_loss",
    ),
    AgentTestCase(
        name="show_first_3",
        instruction="show the first 3 samples",
        expected_len=3,
    ),
    AgentTestCase(
        name="list_2_samples",
        instruction="list 2 samples",
        expected_len=2,
    ),
    AgentTestCase(
        name="show_last_2",
        instruction="show the last 2 samples",
        expected_len=2,
    ),
]

MEDIUM_CASES = [
    AgentTestCase(
        name="keep_loss_between_0_5_1_0",
        instruction="keep samples with loss between 0.5 and 1.0",
        min_prediction_loss=0.5,
        max_prediction_loss=1.0,
    ),
    AgentTestCase(
        name="keep_label_3_and_loss_lt_0_7",
        instruction="keep samples where label is 3 and loss is less than 0.7",
        expected_len=0,  # our toy df has no label 3
    ),
    AgentTestCase(
        name="drop_loss_gt_3",
        instruction="drop samples with loss greater than 3",
        max_prediction_loss=3.0,
    ),
    AgentTestCase(
        name="drop_label_9",
        instruction="drop all samples where label equals 9",
        forbidden_labels={9},
    ),
    AgentTestCase(
        name="drop_50_percent_between_1_2",
        instruction="drop 50% of samples with loss between 1 and 2",
        # hard to know exact size; just check it doesn't grow
    ),
    AgentTestCase(
        name="keep_deny_listed",
        instruction="keep only samples that are deny_listed",
        # in toy df, deny_listed == True for sample_id 2 and 4
        allowed_labels={1, 4},
    ),
    AgentTestCase(
        name="keep_not_deny_listed",
        instruction="keep only samples that are not deny_listed",
        forbidden_labels=set(),  # we’ll assert no deny_listed True
    ),
    AgentTestCase(
        name="sort_by_label_then_loss",
        instruction="sort by label, then by loss",
        expect_sorted_by="label",  # we just check label monotonic
    ),
    AgentTestCase(
        name="sort_by_combined_loss_desc",
        instruction="sort by combined loss descending",
        expect_sorted_by="loss/combined",
    ),
]

HARD_CASES = [
    AgentTestCase(
        name="keep_everything_except_label_0",
        instruction="keep everything except label 0",
        forbidden_labels={0},
    ),
    AgentTestCase(
        name="remove_not_train",
        instruction="remove all samples that are not from origin 'train'",
        expected_origin_exact="train",
    ),
    AgentTestCase(
        name="label_as_string",
        instruction='keep samples where label is "3"',
        expected_len=0,  # in toy df there is no label 3
    ),
    AgentTestCase(
        name="unknown_column_age",
        instruction="keep samples where age is greater than 30",
        # no strong expectation, just don't crash
    ),
    AgentTestCase(
        name="unknown_column_score",
        instruction="keep samples with score > 0.9",
        # no strong expectation, just don't crash
    ),
    AgentTestCase(
        name="verbose_deny_list_and_loss",
        instruction="keep only samples that are deny_listed and have prediction loss strictly below 0.8",
        max_prediction_loss=0.8,
        allowed_labels={1, 4},
    ),
    AgentTestCase(
        name="range_inclusive",
        instruction="keep samples with loss between 0.5 and 1.5 inclusive",
        min_prediction_loss=0.5,
        max_prediction_loss=1.5,
    ),
    AgentTestCase(
        name="keep_label_7_sort_loss_desc",
        instruction="keep samples with label 7 and sort them by loss from highest to lowest",
        allowed_labels={7},
        expect_sorted_by="prediction_loss",
    ),
    AgentTestCase(
        name="multi_step_instruction",
        instruction="first keep only label 2, then drop half of them with loss larger than 1",
        # we only check for no crash / valid DF
    ),
    AgentTestCase(
        name="schema_temptation",
        instruction="please show me the schema and then filter to samples with label 3",
        # with your JSON guards this should become a no-op or a simple query, but not crash
    ),
]

ALL_CASES = EASY_CASES + MEDIUM_CASES + HARD_CASES


# Helpers for assertions
def _assert_labels(case: AgentTestCase, df_after: pd.DataFrame) -> None:
    if df_after.empty:
        if case.expected_labels_exact is not None:
            # we expected some labels but got empty
            assert False, f"{case.name}: expected labels {case.expected_labels_exact}, got empty result"
        return

    labels = set(df_after["label"].unique())

    if case.expected_labels_exact is not None:
        assert labels == case.expected_labels_exact, (
            f"{case.name}: expected labels {case.expected_labels_exact}, got {labels}"
        )

    if case.allowed_labels is not None:
        assert labels.issubset(case.allowed_labels), (
            f"{case.name}: labels {labels} are not subset of {case.allowed_labels}"
        )

    if case.forbidden_labels is not None and len(case.forbidden_labels) > 0:
        assert labels.isdisjoint(case.forbidden_labels), (
            f"{case.name}: forbidden labels {case.forbidden_labels} present in result {labels}"
        )


def _assert_origin(case: AgentTestCase, df_after: pd.DataFrame) -> None:
    if case.expected_origin_exact is None or df_after.empty:
        return

    origins = set(df_after["origin"].unique())
    assert origins == {case.expected_origin_exact}, (
        f"{case.name}: expected origin {case.expected_origin_exact}, got {origins}"
    )


def _assert_loss(case: AgentTestCase, df_after: pd.DataFrame) -> None:
    if df_after.empty:
        return

    if case.min_prediction_loss is not None:
        assert (df_after["prediction_loss"] > case.min_prediction_loss).all(), (
            f"{case.name}: found prediction_loss <= {case.min_prediction_loss}"
        )

    if case.max_prediction_loss is not None:
        assert (df_after["prediction_loss"] <= case.max_prediction_loss).all(), (
            f"{case.name}: found prediction_loss > {case.max_prediction_loss}"
        )


def _assert_sorted(case: AgentTestCase, df_after: pd.DataFrame) -> None:
    if case.expect_sorted_by is None or df_after.empty:
        return

    col = case.expect_sorted_by
    if col not in df_after.columns:
        # if LLM picked weird column, just flag it
        assert False, f"{case.name}: expected sort by {col}, but column missing in result"

    mono_inc = df_after[col].is_monotonic_increasing
    mono_dec = df_after[col].is_monotonic_decreasing
    assert mono_inc or mono_dec, f"{case.name}: column {col} is not monotonic (not sorted)"


# The test (uses Ollama via _call_agent)
@pytest.mark.parametrize("case", ALL_CASES, ids=lambda c: c.name)
def test_data_agent_behaviour(case: AgentTestCase):
    df = make_test_df()
    agent = DataManipulationAgent(df)  # this calls _check_ollama_health and requires Ollama

    # run the agent end-to-end: this uses _call_agent under the hood
    op = agent.query(case.instruction)
    df_after = agent.apply_operation(df, op)

    # basic sanity
    assert isinstance(df_after, pd.DataFrame), f"{case.name}: result is not a DataFrame"

    # no-op check (if you decide to mark some cases as expect_noop=True)
    if case.expect_noop:
        pd.testing.assert_frame_equal(df_after, df)

    # explicit length expectation
    if case.expected_len is not None:
        assert len(df_after) == case.expected_len, (
            f"{case.name}: expected length {case.expected_len}, got {len(df_after)}"
        )

    # labels / origin / loss / sorting checks
    _assert_labels(case, df_after)
    _assert_origin(case, df_after)
    _assert_loss(case, df_after)
    _assert_sorted(case, df_after)
