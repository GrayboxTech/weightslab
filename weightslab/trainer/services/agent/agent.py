import ast
import os
import re
import json
import time
import yaml
import logging
import threading
import pandas as pd
from urllib.parse import urlparse, urlunparse

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Literal, Callable, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Ensure intent_prompt is accessible
from .intent_prompt import INTENT_PROMPT
from .notebook_prompt import NOTEBOOK_CODE_PROMPT
from weightslab.data.sample_stats import SampleStatsEx
from weightslab.trainer.trainer_tools import get_layer_representations


# Set up logging
_LOGGER = logging.getLogger(__name__)


# Try to find .env in weightslab/ or parent root
env_path = Path(__file__).resolve().parents[3] / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()


# ==========================================
# 1. PYDANTIC MODELS (Robust & Flexible)
# ==========================================

class Condition(BaseModel):
    column: str = Field(description="The column name to filter/check")
    op: Literal["==", "=", "equals", "!=", ">", "<", ">=", "<=", "between", "contains", "in", "not in", "is null", "is not null", "max", "min"] = Field(description="The operator")
    value: Optional[Union[float, int, str, List[Union[int, float, str]]]] = Field(default=None, description="The primary value")
    value2: Optional[Union[float, int]] = Field(default=None, description="The secondary value for 'between'")

class AtomicIntent(BaseModel):
    kind: Literal["keep", "drop", "sort", "group", "head", "tail", "reset", "analysis", "transform", "action", "model_info", "model_action", "noop", "clarify"] = Field(
        description="The type of operation. Use 'action' for external tasks like saving or plotting."
    )
    conditions: Optional[List[Condition]] = Field(default=None, description="Conditions for keep/drop")
    sort_by: Optional[List[str]] = Field(default=None, description="Exact column names to sort by")
    # Robustness: Accepts bool OR list of bools for multi-column sorting
    ascending: Optional[Union[bool, List[bool]]] = Field(default=None, description="True for ASC, False for DESC")
    # Robustness: Accepts (int) count OR (str) expression like "10%"
    n: Optional[Union[int, str]] = Field(default=None, description="Number of rows for head/tail (int or '10%')")
    drop_frac: Optional[float] = Field(default=None, description="Fraction of rows to drop (0.0 to 1.0)")
    keep_frac: Optional[float] = Field(default=None, description="Fraction of rows to keep (0.0 to 1.0)")
    analysis_expression: Optional[str] = Field(default=None, description="Pandas expression string for analysis queries")

    # Transformation (Column Modification)
    transform_code: Optional[str] = Field(default=None, description="Pandas expression for the new value (e.g. df['col'] * 2)")
    target_column: Optional[str] = Field(default=None, description="The column to create or modify")
    is_temporary: Optional[bool] = Field(
        default=False,
        description="True if target_column is scratch state used only to help compute a LATER step's result within this same request (e.g. two intermediate tags combined into a final one). It is dropped automatically once the whole request finishes executing. Never set this on the column the user actually asked for.",
    )

    # Future-proofing for Actions
    action_name: Optional[str] = Field(default=None, description="Name of the action (e.g. 'save_dataset')")
    action_params: Optional[Dict[str, Any]] = Field(default=None, description="Parameters for the action")

    # Model introspection / architecture management (kind="model_info"/"model_action")
    layer_query: Optional[List[Condition]] = Field(
        default=None,
        description="Conditions over layer attributes (layer_id, layer_name, layer_type, neurons_count, incoming_neurons_count, frozen) to select which layer(s) to inspect or act on. Omit to target every layer.",
    )
    model_query_expression: Optional[str] = Field(
        default=None,
        description="Pandas expression evaluated against `layers_df` for aggregate model questions (e.g. 'layers_df[\"neurons_count\"].sum()'). Use layer_query instead for simple filters.",
    )
    model_action_name: Optional[Literal["freeze", "reset", "unfreeze"]] = Field(
        default=None,
        description="Architecture operation to apply to the layer(s)/neuron(s) selected by layer_query. 'unfreeze' only ever touches currently-frozen layers/neurons.",
    )
    neuron_indices: Optional[List[int]] = Field(
        default=None,
        description="Specific neuron indices within the selected layer(s) for model_action. Omit to target the whole layer.",
    )

class Intent(BaseModel):
    reasoning: str = Field(description="The thought process or clarification question.")
    primary_goal: Literal["ui_manipulation", "data_analysis", "action", "model_management", "out_of_scope"] = Field(
        description="Whether the user wants to change the grid view, get an answer, perform an action, or inspect/manage the model's architecture."
    )
    steps: List[AtomicIntent] = Field(description="A sequence of atomic operations to execute in order.")


# ==========================================
# 2. HANDLER STRATEGY (The Logic Layer)
# ==========================================

class IntentHandler(ABC):
    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        """Converts an atomic step into an execution dictionary."""
        pass

class SortHandler(IntentHandler):
    """Handles sorting logic. Maps 'group' intents to sort operations."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        # 1. Resolve columns (fuzzy match)
        raw_cols = step.sort_by or [c.column for c in step.conditions or []]
        resolved_cols = [self.agent._resolve_column(c) for c in raw_cols if c]
        resolved_cols = [c for c in resolved_cols if c] # Remove Nones

        if not resolved_cols: return None

        # 2. Determine Direction (Default: Ascending for sort, Descending for group)
        ascending = step.ascending if step.ascending is not None else (step.kind != "group")

        # 3. Safety Net: Check textual cues to override default Ascending
        if ascending is True:
            reasoning = context.reasoning.lower()
            desc_triggers = {"desc", "highest", "largest", "decreasing", "down", "worst"}

            # Force DESC for 'group' unless 'asc' is explicitly requested
            if step.kind == "group" and "asc" not in reasoning:
                ascending = False
            # Force DESC if reasoning contains trigger words
            elif any(t in reasoning for t in desc_triggers):
                ascending = False

        # 4. Broadcast boolean to match number of columns (Pandas requirement)
        if isinstance(ascending, bool):
            ascending = [ascending] * len(resolved_cols)
        elif len(ascending) < len(resolved_cols):
            ascending.extend([True] * (len(resolved_cols) - len(ascending)))

        return {
            "function": "df.sort_values",
            "params": {"by": resolved_cols, "ascending": ascending}
        }

class FilterHandler(IntentHandler):
    """Handles keep/drop logic and query construction."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        kind = step.kind

        # Case A: Structured Conditions
        if step.conditions:
            # Sequential Filtering Strategy: Apply filters before extrema (for "Highest X in subset Y")
            extremes = [c for c in step.conditions if c.op.lower() in ('max', 'min')]
            filters = [c for c in step.conditions if c.op.lower() not in ('max', 'min')]

            if kind == "keep" and extremes and filters:
                ops = []
                # 1. Apply standard filters
                expr_f = self.agent._build_python_mask(filters, n=None)
                if expr_f:
                    ops.append({"function": "df.apply_mask", "params": {"code": expr_f}})

                # 2. Apply extrema on the subset
                expr_e = self.agent._build_python_mask(extremes, n=step.n)
                if expr_e:
                    if step.keep_frac:
                        ops.append({"function": "df.drop", "params": {"index": f"df.index.difference(df[{expr_e}].sample(frac={step.keep_frac}).index)"}})
                    else:
                        ops.append({"function": "df.apply_mask", "params": {"code": expr_e}})
                return ops

            # Standard Logic (Single Step)
            expr = self.agent._build_python_mask(step.conditions, n=step.n)
            if not expr: return None

            if kind == "keep":
                # Handle sampling on keep result
                if step.keep_frac:
                    return {"function": "df.drop", "params": {"index": f"df.index.difference(df[{expr}].sample(frac={step.keep_frac}).index)"}}

                return {"function": "df.apply_mask", "params": {"code": expr}}

            else: # drop
                # If frac provided: Drop frac of matches.
                if step.drop_frac:
                     return {"function": "df.drop", "params": {"index": f"df[{expr}].sample(frac={step.drop_frac}).index"}}

                # Standard drop
                return {"function": "df.drop", "params": {"index": f"df[{expr}].index"}}

        # Case B: Free-form Expression (Top N, etc)
        elif step.analysis_expression:
            expr = self.agent._rewrite_origin_literals(self.agent._clean_code(step.analysis_expression))
            if kind == "keep":
                return {"function": "df.apply_mask", "params": {"code": expr}}
            else: # drop
                return {"function": "df.drop", "params": {"index": f"df.query({expr}).index"}}
        return None

class AnalysisHandler(IntentHandler):
    """Handles read-only analysis requests with robust column auto-correction."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        if not step.analysis_expression: return None

        raw_code = self.agent._clean_code(step.analysis_expression)

        # 1. Broad Regex: Matches any ['string'] or ["string"] pattern. This catches df['col'], df.loc['col'], and df[...]['col']
        pattern = r"(\[\s*['\"])(.*?)(['\"]\s*\])"

        def replace_col(match):
            prefix = match.group(1) # e.g. ['
            content = match.group(2) # e.g. signals//train_loss
            suffix = match.group(3) # e.g. ']

            resolved = self.agent._resolve_column(content) # Try to resolve the content to a real column

            if resolved:
                return f"{prefix}{resolved}{suffix}" # If we found a better match in the schema, use it

            return match.group(0) # If it's a value (e.g., 'bug' in tags), leave it alone

        fixed_code = re.sub(pattern, replace_col, raw_code)
        fixed_code = self.agent._rewrite_origin_literals(fixed_code)

        return {
            "function": "df.analyze",
            "params": {"code": fixed_code}
        }

class TransformHandler(IntentHandler):
    """Handles column creation and modification with column resolution."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        if not step.target_column or not step.transform_code: return None

        raw_code = self.agent._clean_code(step.transform_code)

        # Reuse robust column resolution logic (similar to AnalysisHandler)
        pattern = r"(\[\s*['\"])(.*?)(['\"]\s*\])"

        def replace_col(match):
            prefix = match.group(1)
            content = match.group(2)
            suffix = match.group(3)
            resolved = self.agent._resolve_column(content)
            if resolved:
                return f"{prefix}{resolved}{suffix}"
            return match.group(0)

        fixed_code = re.sub(pattern, replace_col, raw_code)
        fixed_code = self.agent._rewrite_origin_literals(fixed_code)

        return {
            "function": "df.modify",
            "params": {
                "col": step.target_column,
                "code": fixed_code,
                "temporary": bool(step.is_temporary),
            }
        }

class ViewHandler(IntentHandler):
    """Handles view resets and head/tail slicing."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        if step.kind == "reset":
            return {"function": "df.reset_view", "params": {"__agent_reset__": True}}
        elif step.kind in ["head", "tail"]:
            return {"function": f"df.{step.kind}", "params": {"n": step.n or 5}}
        return None

class ClarifyHandler(IntentHandler):
    """Handles ambiguity by returning the reasoning as a question."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        return {
            "function": "clarify",
            "params": {"reason": context.reasoning}
        }

class ActionHandler(IntentHandler):
    """Extensible handler for future actions (Save, Plot, etc)."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        if not step.action_name: return None
        return {
            "function": f"action.{step.action_name}",
            "params": step.action_params or {}
        }

class ModelInfoHandler(IntentHandler):
    """Read-only architecture questions (layer/neuron counts, frozen state, full dump)."""
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        agent = self.agent
        if not agent.model_available:
            return {"function": "model.info", "params": {"text": agent._format_layers_table()}}

        if step.model_query_expression:
            code = agent._clean_code(step.model_query_expression)
            try:
                result = eval(code, {"layers_df": agent.model_layers_df, "pd": pd})
                text = str(result)
            except Exception as e:
                text = f"Could not evaluate model query: {e}"
            return {"function": "model.info", "params": {"text": text}}

        selected = agent._select_layers(step.layer_query)
        return {"function": "model.info", "params": {"text": agent._format_layers_table(selected)}}

class ModelActionHandler(IntentHandler):
    """Architecture management: freeze/reset/unfreeze layers or specific neurons.

    `unfreeze` reuses the same `model.freeze` execution path: freezing a
    neuron TOGGLES its learning rate (new_lr = 1.0 - current_lr), so applying
    it again to an already-frozen neuron restores it — no separate backend
    primitive exists or is needed. To keep this safe, an unfreeze request is
    always constrained to layers/neurons that are ALREADY frozen, so it can
    never accidentally freeze something that wasn't frozen yet.
    """
    def build_op(self, step: AtomicIntent, context: Intent) -> Optional[dict]:
        agent = self.agent
        action = step.model_action_name
        if not action:
            return None

        if not agent.model_available:
            return {"function": "model.error", "params": {"reason": "No model is currently registered; there is no architecture to modify."}}

        if action == "unfreeze":
            return self._build_unfreeze_op(agent, step)

        selected = agent._select_layers(step.layer_query)
        layer_ids = [] if selected is None else selected["layer_id"].tolist()
        if not layer_ids:
            return {"function": "model.error", "params": {"reason": "No layers matched the given criteria; nothing was changed."}}

        return {
            "function": f"model.{action}",
            "params": {"layer_ids": layer_ids, "neuron_ids": step.neuron_indices or []},
        }

    @staticmethod
    def _build_unfreeze_op(agent, step: AtomicIntent) -> dict:
        selected = agent._select_layers(step.layer_query)
        layer_ids = [] if selected is None else selected["layer_id"].tolist()
        if not layer_ids:
            return {"function": "model.error", "params": {"reason": "No layers matched the given criteria; nothing to unfreeze."}}

        # Neuron-scoped unfreeze on a single resolved layer: only toggle the
        # subset of the requested neurons that are actually frozen.
        if step.neuron_indices and len(layer_ids) == 1:
            layer_id = layer_ids[0]
            neurons_df = agent.model_neurons_df
            frozen_ids = []
            if neurons_df is not None and not neurons_df.empty:
                mask = (
                    (neurons_df["layer_id"] == layer_id)
                    & (neurons_df["neuron_id"].isin(step.neuron_indices))
                    & (neurons_df["frozen"] == True)  # noqa: E712
                )
                frozen_ids = neurons_df.loc[mask, "neuron_id"].tolist()
            if not frozen_ids:
                return {"function": "model.error", "params": {"reason": f"None of the requested neurons in layer {layer_id} are currently frozen."}}
            return {"function": "model.freeze", "params": {"layer_ids": [layer_id], "neuron_ids": frozen_ids}}

        # Layer-scoped unfreeze: constrain to layers that are ACTUALLY frozen
        # (freeze toggles, so applying it to an unfrozen layer would freeze it).
        frozen_selected = selected[selected["frozen"] == True] if "frozen" in selected.columns else selected.iloc[0:0]  # noqa: E712
        frozen_layer_ids = frozen_selected["layer_id"].tolist()
        if not frozen_layer_ids:
            return {"function": "model.error", "params": {"reason": "None of the selected layers are currently frozen; nothing to unfreeze."}}

        return {"function": "model.freeze", "params": {"layer_ids": frozen_layer_ids, "neuron_ids": []}}


# ==========================================
# 3. THE AGENT (Orchestrator)
# ==========================================

class DataManipulationAgent:
    def __init__(self, context):
        """Initializes the agent with context and builds the column schema/index."""
        _LOGGER.info("Initializing DataManipulationAgent")
        self.ctx = context

        self._setup_schema()
        self._setup_model_schema()
        self._build_column_index()
        self._load_config()
        self._setup_providers()
        self._verify_startup_providers()
        self.history = []

        # --- HANDLER REGISTRY ---
        self.handlers = {
            "sort": SortHandler(self),
            "group": SortHandler(self),
            "keep": FilterHandler(self),
            "drop": FilterHandler(self),
            "transform": TransformHandler(self),
            "analysis": AnalysisHandler(self),
            "head": ViewHandler(self),
            "tail": ViewHandler(self),
            "reset": ViewHandler(self),
            "clarify": ClarifyHandler(self),
            "action": ActionHandler(self),
            "model_info": ModelInfoHandler(self),
            "model_action": ModelActionHandler(self),
            "noop": None
        }

    @staticmethod
    def _format_samples_for_prompt(samples, max_items: int = 6, max_len: int = 60) -> str:
        """Render sample values compactly for the LLM prompt.

        The raw schema keeps full values (needed by _resolve_categorical_value),
        but object columns can hold huge reprs — absolute file paths, ArrayH5Proxy
        objects — that bloat the prompt by thousands of tokens for zero planning
        value. Truncate each value's length and cap the count for display only.
        """
        if not samples:
            return "[]"
        shown = []
        for v in samples[:max_items]:
            s = str(v)
            if len(s) > max_len:
                s = s[:max_len] + "…"
            shown.append(s)
        suffix = f", … (+{len(samples) - max_items} more)" if len(samples) > max_items else ""
        return "[" + ", ".join(repr(x) for x in shown) + "]" + suffix

    def _dataframe_fingerprint(self):
        """Cheap fingerprint of the shared dataframe, used to decide whether the
        (expensive) per-column stats schema needs rebuilding.

        Every mutation path in DataService surfaces as either a reassignment of
        ``_all_datasets_df`` (id changes) or a shape/column/index change, so
        this tuple is a correct invalidation signal without hashing the data.
        Numeric ranges/means can still drift during training without changing
        the fingerprint, but those only feed the LLM's numeric *hints*, never
        correctness (column resolution, origin values), so staleness is fine.
        """
        df = getattr(self.ctx, "_all_datasets_df", None)
        if df is None:
            return None
        try:
            return (id(df), df.shape, tuple(df.columns), tuple(df.index.names))
        except Exception:
            return None

    def _setup_schema(self):
        """Builds a rich column schema with statistical context for the LLM."""
        # Skip the expensive rebuild when the dataframe is unchanged since the
        # last build (same call is made on every query()); see plan Phase B.3.
        fp = self._dataframe_fingerprint()
        if (
            fp is not None
            and fp == getattr(self, "_schema_fp", None)
            and getattr(self, "df_schema", None) is not None
        ):
            return

        df = self.ctx._all_datasets_df

        # Internal bookkeeping columns/levels that the user should never query
        # against. ``annotation_id`` is the per-instance multi-index level and
        # ``_instance_signals`` is the nested per-instance signal dict produced
        # when the (sample_id, annotation_id) view is collapsed to one row per
        # sample. Hide both from the LLM-facing schema.
        INTERNAL_COLUMNS = {"annotation_id", "_instance_signals"}

        all_columns = [c for c in df.columns.tolist() if c not in INTERNAL_COLUMNS]
        index_columns = []

        if isinstance(df.index, pd.MultiIndex):
            for name in df.index.names:
                if name is not None and name not in INTERNAL_COLUMNS:
                    index_columns.append(name)
                    if name not in all_columns:
                        all_columns.append(name)
        elif df.index.name is not None and df.index.name not in INTERNAL_COLUMNS:
            index_columns.append(df.index.name)
            if df.index.name not in all_columns:
                all_columns.append(df.index.name)

        # Report the number of distinct samples, not raw rows. The shared
        # dataframe may still be annotation-expanded (one row per instance), so
        # falling back to len(df) would overstate the dataset size to the LLM.
        sample_level = SampleStatsEx.SAMPLE_ID.value
        try:
            if isinstance(df.index, pd.MultiIndex) and sample_level in (df.index.names or []):
                row_count = int(df.index.get_level_values(sample_level).nunique())
            elif sample_level in df.columns:
                row_count = int(df[sample_level].nunique())
            else:
                row_count = len(df)
        except Exception:
            row_count = len(df)

        column_metadata = {}
        for col in all_columns:
            try:
                # Get a slice to avoid giant overhead
                s = None
                if col in df.columns: s = df[col]
                elif col in index_columns: s = df.index.get_level_values(col)

                if s is not None:
                    dtype = str(s.dtype)
                    meta = {"dtype": dtype}

                    if "float" in dtype or "int" in dtype:
                        meta["range"] = [float(s.min()), float(s.max())]
                        meta["mean"] = float(s.mean()) if len(s) > 0 else 0
                    elif "object" in dtype or "category" in dtype or "string" in dtype:
                        # Categorical/String context
                        unique_vals = s.dropna().unique().tolist()
                        meta["samples"] = unique_vals[:10]
                        meta["unique_count"] = len(unique_vals)

                    column_metadata[col] = meta
            except:
                column_metadata[col] = {"dtype": "unknown"}

        self.df_schema = {
            'columns': all_columns,
            'index_columns': index_columns,
            'metadata': column_metadata,
            'row_count': row_count
        }
        self._build_column_index()
        self._schema_fp = fp

    def _setup_model_schema(self):
        """
        Builds small layer- and neuron-level tables describing the live
        model's architecture, mirroring `_setup_schema` but for `model.*`
        requests (introspection + freeze/reset/unfreeze) instead of
        dataframe requests.
        """
        layer_columns = [
            "layer_id", "layer_name", "layer_type", "neurons_count",
            "incoming_neurons_count", "kernel_size", "stride", "frozen",
        ]
        neuron_columns = ["layer_id", "neuron_id", "learning_rate", "frozen"]

        # Resolve the live model cheaply first so we can decide whether the
        # (potentially expensive) per-neuron walk below needs to rerun.
        model = None
        try:
            exp_ctx = getattr(self.ctx, "_ctx", None)
            if exp_ctx is None:
                _LOGGER.info("[Agent] No experiment context (_ctx) available on DataService; skipping model schema.")
            else:
                exp_ctx.ensure_components()
                model = exp_ctx.components.get("model")
                if model is None:
                    try:
                        from weightslab.backend.ledgers import list_models, get_model
                        registered = list_models()
                        model = get_model(registered[0]) if registered else None
                    except Exception:
                        registered = "<could not list>"
                    if model is None:
                        _LOGGER.info(
                            "[Agent] components.get('model') returned None (ledger-registered model names: %s). "
                            "Model-related agent requests will report 'no model registered'. If a model IS "
                            "registered under a name other than the experiment name / 'experiment' / 'main', "
                            "ExperimentContext.ensure_components()'s model-resolution heuristic won't find it.",
                            registered,
                        )
        except Exception as e:
            _LOGGER.warning(f"[Agent] Failed to resolve model for schema: {e}", exc_info=True)
            model = None

        # Cache guard: same model object and not explicitly invalidated (frozen
        # state is mutated by model_action, which keeps the same object, so
        # invalidate_model_schema() is called after any model.* op) -> reuse the
        # already-built tables and skip the neuron walk. See plan Phase B.3.
        fp = (id(model),) if model is not None else None
        if (
            fp is not None
            and fp == getattr(self, "_model_schema_fp", None)
            and not getattr(self, "_model_schema_dirty", False)
            and getattr(self, "model_layers_df", None) is not None
        ):
            return

        self.model_layers_df = pd.DataFrame(columns=layer_columns)
        self.model_neurons_df = pd.DataFrame(columns=neuron_columns)
        self.model_available = False
        self._model_schema_fp = fp
        self._model_schema_dirty = False

        if model is None:
            return

        try:
            layer_rows = []
            neuron_rows = []
            for rep in get_layer_representations(model):
                lrs = [ns.learning_rate for ns in rep.neurons_statistics]
                # A layer is considered frozen when every neuron's learning
                # rate has been zeroed out (how `FREEZE` is implemented).
                frozen = bool(lrs) and all(lr == 0 for lr in lrs)
                layer_rows.append({
                    "layer_id": rep.layer_id,
                    "layer_name": rep.layer_name,
                    "layer_type": rep.layer_type,
                    "neurons_count": rep.neurons_count,
                    "incoming_neurons_count": rep.incoming_neurons_count,
                    "kernel_size": rep.kernel_size,
                    "stride": rep.stride,
                    "frozen": frozen,
                })
                for ns in rep.neurons_statistics:
                    neuron_rows.append({
                        "layer_id": rep.layer_id,
                        "neuron_id": ns.neuron_id.neuron_id,
                        "learning_rate": ns.learning_rate,
                        "frozen": ns.learning_rate == 0,
                    })
            if layer_rows:
                self.model_layers_df = pd.DataFrame(layer_rows, columns=layer_columns)
                self.model_available = True
            if neuron_rows:
                self.model_neurons_df = pd.DataFrame(neuron_rows, columns=neuron_columns)
        except Exception as e:
            _LOGGER.warning(f"[Agent] Failed to build model schema: {e}", exc_info=True)

    def invalidate_model_schema(self) -> None:
        """Force the next `_setup_model_schema()` to rebuild the layer/neuron
        tables even though the model object is unchanged.

        Freeze/reset/unfreeze mutate per-neuron learning rates (hence `frozen`
        flags) in place on the same model object, so the cheap identity
        fingerprint can't detect them. DataService calls this right after
        applying any `model.*` op so frozen state never goes stale."""
        self._model_schema_dirty = True

    def _build_layer_mask(self, conditions: List["Condition"]) -> Optional[str]:
        """Builds a boolean mask expression over `layers_df` (small in-memory
        layer table), analogous to `_build_python_mask` for the dataframe."""
        if not conditions or self.model_layers_df is None or self.model_layers_df.empty:
            return None

        parts = []
        for cond in conditions:
            col = cond.column
            if col not in self.model_layers_df.columns:
                continue

            op = cond.op.lower()
            if op in ("=", "equals"):
                op = "=="

            val = cond.value
            if col not in ("layer_name", "layer_type") and isinstance(val, str):
                low = val.strip().lower()
                if low in ("true", "false"):
                    val = (low == "true")
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

            col_ref = f"layers_df['{col}']"
            val_repr = repr(val)

            if op == "==": parts.append(f"({col_ref} == {val_repr})")
            elif op == "!=": parts.append(f"({col_ref} != {val_repr})")
            elif op == ">": parts.append(f"({col_ref} > {val_repr})")
            elif op == "<": parts.append(f"({col_ref} < {val_repr})")
            elif op == ">=": parts.append(f"({col_ref} >= {val_repr})")
            elif op == "<=": parts.append(f"({col_ref} <= {val_repr})")
            elif op == "contains": parts.append(f"({col_ref}.astype(str).str.contains({val_repr}, na=False, regex=False))")
            elif op == "in":
                v = val if isinstance(val, list) else [val]
                parts.append(f"({col_ref}.isin({v!r}))")
            elif op == "not in":
                v = val if isinstance(val, list) else [val]
                parts.append(f"(~{col_ref}.isin({v!r}))")

        return " & ".join(parts) if parts else None

    def _select_layers(self, conditions: Optional[List["Condition"]]) -> pd.DataFrame:
        """Returns the subset of `model_layers_df` matching `conditions` (all layers if none given)."""
        df = self.model_layers_df
        if df is None or df.empty or not conditions:
            return df
        expr = self._build_layer_mask(conditions)
        if not expr:
            return df
        try:
            return df[eval(expr, {"layers_df": df})]
        except Exception as e:
            _LOGGER.warning(f"[Agent] Failed to filter layers with {expr!r}: {e}")
            return df

    def _format_layers_table(self, df: Optional[pd.DataFrame] = None) -> str:
        target = df if df is not None else self.model_layers_df
        if not self.model_available or target is None or target.empty:
            return "No model is currently registered, so there is no architecture information available."

        lines = [f"Model has {len(target)} layer(s):"]
        for _, row in target.iterrows():
            lines.append(
                f"- Layer {row['layer_id']} ({row['layer_name']}/{row['layer_type']}): "
                f"\t{row['neurons_count']} neurons, incoming={row['incoming_neurons_count']}, "
                f"\tfrozen={row['frozen']}"
            )
        return "\n".join(lines)

    def _load_config(self):
        self.preferred_provider = os.environ.get("PREFERRED_PROVIDER", "openrouter") # Default to OpenRouter if API key is provided, otherwise fallback to local Ollama. This can be overridden by config file or env variable.

        # Cloud provider settings with sensible defaults. OpenRouter is the default cloud provider if API key is provided.
        # Default to a fast flash-class model: the intent-planning task is
        # simple JSON generation, and a 70B model added ~15-30s of latency for
        # no accuracy benefit (see plan Phase B.1). Override with OPENROUTER_MODEL
        # (or agent_config.yaml) to use a larger model.
        self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "~google/gemini-flash-latest")
        self.openrouter_base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", None)
        self.openrouter_request_timeout = float(os.environ.get("OPENROUTER_REQUEST_TIMEOUT", "15.0"))
        # Bias OpenRouter's upstream provider selection ("throughput"/"latency"/
        # "price"); empty string lets OpenRouter choose freely (see Phase B.2).
        self.openrouter_provider_sort = os.environ.get("OPENROUTER_PROVIDER_SORT", "throughput")
        # Ask the model for a schema-validated Intent object directly instead of
        # free-form JSON + regex repair. More reliable, but only works on models
        # whose OpenRouter route supports structured/JSON-schema output (e.g.
        # Gemini, GPT-4o). Default OFF so an unsupported model can't break; flip
        # on with OPENROUTER_STRUCTURED_OUTPUT=1 or the config key.
        self.openrouter_structured_output = os.environ.get(
            "OPENROUTER_STRUCTURED_OUTPUT", ""
        ).strip().lower() in ("1", "true", "yes", "on")

        # Local fallback if no cloud (OpenRouter) is available or if the user prefers it. Ollama is the default local provider.
        self.fallback_to_local = True # Default to allowing fallback to local Ollama if OpenRouter fails
        self.ollama_host = "localhost"
        self.ollama_port = "11435"
        self.ollama_model = "llama3.2:3b"

        repo_root = Path(__file__).resolve().parents[4] # weightslab/ root
        inner_pkg = Path(__file__).resolve().parents[3]

        env_paths = [repo_root / ".env", inner_pkg / ".env"]
        for ep in env_paths:
            if ep.exists():
                load_dotenv(dotenv_path=ep)
                _LOGGER.info(f"Loaded credentials from {ep}")
                break

        config_paths = [
            Path(os.environ.get("AGENT_CONFIG_PATH", repo_root)),
            Path(os.environ.get("AGENT_CONFIG_PATH", repo_root)) / ".agent_config.yaml",
            Path(os.environ.get("AGENT_CONFIG_PATH", repo_root)) / "agent_config.yaml",
            inner_pkg / "agent_config.yaml",
            Path.cwd() / "agent_config.yaml"
        ]
        for path in config_paths:
            if not path.exists(): continue
            try:
                with open(path, 'r') as f:
                    cfg = yaml.safe_load(f)
                if not cfg or "agent" not in cfg: continue
                a_cfg = cfg["agent"]

                # Agents settings
                self.preferred_provider = a_cfg.get("provider", self.preferred_provider).lower()
                self.fallback_to_local = a_cfg.get("fallback_to_local", self.fallback_to_local)

                # OPENROUTER
                self.openrouter_model = a_cfg.get("openrouter_model", self.openrouter_model)
                self.openrouter_base_url = a_cfg.get("openrouter_base_url", self.openrouter_base_url)
                self.openrouter_api_key = a_cfg.get("openrouter_api_key", self.openrouter_api_key)
                self.openrouter_request_timeout = float(a_cfg.get("openrouter_request_timeout", self.openrouter_request_timeout))
                self.openrouter_provider_sort = a_cfg.get("openrouter_provider_sort", self.openrouter_provider_sort)
                self.openrouter_structured_output = bool(a_cfg.get("openrouter_structured_output", self.openrouter_structured_output))

                # OLLAMA
                self.ollama_host = a_cfg.get("ollama_host", self.ollama_host)
                self.ollama_port = a_cfg.get("ollama_port", self.ollama_port)
                self.ollama_model = a_cfg.get("ollama_model", self.ollama_model)

                _LOGGER.info(f"Applied agent configuration from {path}")
                _LOGGER.debug(f"Agent Config: {cfg}")
                break
            except Exception as e:
                _LOGGER.warning(f"Error loading config from {path}: {e}")

        # Log the final configuration for transparency
        _LOGGER.info(
            "" + "\n" +
            "\n# #######################################" + "\n" +
            "# #######################################" + "\n" +
            f"Agent initialized from configuration {path}: " + "\n" +
            f"\tFinal Agent Configuration: Preferred Provider={self.preferred_provider}, " + "\n" +
            f"\tFallback to Local={self.fallback_to_local}, " + "\n" +
            f"\tOpenRouter Model={self.openrouter_model} with:" + "\n" +
            f"\t\tAPI Key={f'{self.openrouter_api_key[:4]}****{self.openrouter_api_key[-4:]}' if self.openrouter_api_key else 'None'}" + "\n" +
            f"\t\tBase URL={self.openrouter_base_url}, " + "\n" +
            f"\tOllama Model={self.ollama_model}" + "\n" +
            "# #######################################" + "\n" +
            "# #######################################" + "\n" + ""
        )

    @staticmethod
    def _effective_http_port(parsed_url, explicit_port: Optional[str]) -> int:
        if explicit_port and explicit_port.isdigit():
            return int(explicit_port)
        if parsed_url.port is not None:
            return parsed_url.port
        return 443 if parsed_url.scheme == "https" else 80

    @staticmethod
    def _normalize_openrouter_base_url(raw_url: str, explicit_port: Optional[str]) -> str:
        url = (raw_url or "https://openrouter.ai/api/v1").strip()
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f"https://{url}")

        host = parsed.hostname or "openrouter.ai"
        port = DataManipulationAgent._effective_http_port(parsed, explicit_port)
        netloc = host if ((parsed.scheme == "https" and port == 443) or (parsed.scheme == "http" and port == 80)) else f"{host}:{port}"
        path = parsed.path if parsed.path else "/api/v1"

        return urlunparse((parsed.scheme, netloc, path, "", "", ""))

    def _setup_providers(self):
        self.chain_ollama = None
        self.chain_openrouter = None
        initialized = False

        # Determine which providers to initialize
        active_providers = {self.preferred_provider}
        if self.fallback_to_local:
            active_providers.add("ollama")

        # OPEN ROUTER
        if "openrouter" in active_providers and self.openrouter_api_key:
            _LOGGER.info(f"Setting up OpenRouter with model {self.openrouter_model}")
            try:
                if ChatOpenAI is None:
                    _LOGGER.warning("langchain_openai is not installed, skipping OpenRouter provider")
                else:
                    explicit_openrouter_port = os.environ.get("OPENROUTER_PORT", "").strip()
                    openrouter_base_url = self._normalize_openrouter_base_url(self.openrouter_base_url, explicit_openrouter_port)
                    parsed = urlparse(openrouter_base_url)
                    effective_port = self._effective_http_port(parsed, explicit_openrouter_port)

                    # Bias OpenRouter's upstream routing so it doesn't pick a
                    # slow/cold provider for the model — a major source of the
                    # query tail latency (see plan Phase B.2). Configurable via
                    # `openrouter_provider_sort` ("throughput"/"latency"/"price");
                    # set to "" to let OpenRouter choose freely.
                    extra_body = None
                    sort = getattr(self, "openrouter_provider_sort", "throughput")
                    if sort:
                        extra_body = {"provider": {"sort": sort}}

                    llm = ChatOpenAI(
                        model=self.openrouter_model, temperature=0,
                        api_key=self.openrouter_api_key,
                        base_url=openrouter_base_url,
                        streaming=False, max_retries=1, request_timeout=self.openrouter_request_timeout,
                        extra_body=extra_body,
                    )
                    self.chain_openrouter = llm
                    initialized = True
                    _LOGGER.info(
                        f"[Agent] OpenRouter enabled: {self.openrouter_model} via {parsed.hostname}:{effective_port} "
                        f"(provider sort={sort or 'default'})"
                    )
            except Exception as e: _LOGGER.error(f"OpenRouter error: {e}")

        # LOCAL
        if "ollama" in active_providers:
            try:
                if ChatOllama is None:
                    _LOGGER.warning("langchain_ollama is not installed, skipping Ollama provider")
                else:
                    _LOGGER.info(f"Setting up Ollama with model {self.ollama_model}")
                    host = self.ollama_host.split(':')[0]
                    port = self.ollama_port
                    llm = ChatOllama(base_url=f"http://{host}:{port}", model=self.ollama_model, temperature=0, timeout=15)
                    self.chain_ollama = llm
                    initialized = True
                    _LOGGER.info(f"[Agent] Ollama enabled: {self.ollama_model}")
            except Exception as e: _LOGGER.error(f"Ollama error: {e}")

        return initialized

    def _verify_startup_providers(self) -> None:
        """
        A provider configured at construction time (agent_config.yaml / env
        vars, e.g. OPENROUTER_API_KEY) never goes through the connectivity
        check the `/init` UI flow runs (see `initialize_with_cloud_key`) --
        `_setup_providers()` only builds a client object from whatever key
        string it was given, it never confirms the key is actually accepted.
        `is_available()` treats "a client object exists" as "ready", so a
        bad startup key would otherwise report available=True indefinitely
        (CheckAgentHealth says "ready to help you") until the first real
        query 401s. Probe once here instead, so that mismatch can't happen.

        Only runs for the OpenRouter chain built in `__init__`: Ollama's
        `is_available()` already does a live reachability check every call,
        and `initialize_with_cloud_key`/`change_model` already run their own
        explicit post-`_setup_providers()` check.
        """
        if self.chain_openrouter is None:
            return
        ok, message = self._check_chat_provider("openrouter")
        if not ok:
            _LOGGER.warning(f"[Agent] Startup OpenRouter connectivity check failed, disabling it: {message}")
            self.chain_openrouter = None

    def _check_chat_provider(self, provider: str) -> "tuple[bool, str]":
        """Run a minimal chat request to verify the provider is actually usable."""
        chain = getattr(self, f"chain_{provider}", None)
        if chain is None:
            return False, f"{provider} client was not initialized."

        try:
            response = chain.invoke("Reply with OK.")
        except Exception as e:
            _LOGGER.warning(f"[{provider}] connectivity check failed: {e}")
            return False, f"{provider} connectivity check failed: {e}"

        text = response.content if hasattr(response, "content") else str(response)
        if not str(text).strip():
            return False, f"{provider} connectivity check returned an empty response."

        return True, f"{provider} connectivity check succeeded."

    def is_ollama_available(self) -> bool:
        return self.chain_ollama is not None

    def is_available(self) -> bool:
        """
        Return True if any LLM provider is actually ready to serve requests.

        OpenRouter is considered ready as soon as its chain is set up.

        Ollama requires an active HTTP connection check because the ChatOllama
        constructor succeeds even when the daemon is not running.
        """
        if self.chain_openrouter is not None:
            return True
        if self.chain_ollama is not None:
            return self._is_ollama_reachable()
        return False

    def _is_ollama_reachable(self) -> bool:
        """Ping the Ollama HTTP endpoint to verify the daemon is actually running."""
        try:
            import urllib.request as _ur
            host = self.ollama_host.split(':')[0]
            url = f"http://{host}:{self.ollama_port}/api/version"
            with _ur.urlopen(_ur.Request(url), timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def initialize_with_cloud_key(self, api_key: str, provider: str, model: Optional[str] = None) -> "tuple[bool, str]":
        """
        Initialize (or reinitialize) the OpenRouter cloud provider.

        Args:
            api_key: The API key obtained from the provider's website.
            provider: Must be ``"openrouter"``.
            model: OpenRouter model identifier chosen by the user.

        Returns:
            ``(True, success_message)`` or ``(False, error_message)``.
        """
        if not api_key or not api_key.strip():
            return False, "API key cannot be empty."

        if provider.lower() != "openrouter":
            return False, "Only OpenRouter cloud onboarding is supported."

        if model is not None and not model.strip():
            return False, "Model cannot be empty."

        self.openrouter_api_key = api_key.strip()
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.openrouter_model = model.strip() if model and model.strip() else self.openrouter_model
        self.preferred_provider = "openrouter"

        success = self._setup_providers()

        if self.chain_openrouter is None or not success:
            return False, "Provider client initialization failed. Please verify your API key, base URL, and model."

        chat_ok, chat_message = self._check_chat_provider("openrouter")
        if not chat_ok:
            self.chain_openrouter = None
            return False, chat_message

        return True, "Agent initialized successfully. Ready to help you."

    def change_model(self, model: str) -> "tuple[bool, str]":
        """
        Switch the active OpenRouter model without re-entering the API key.

        Args:
            model: OpenRouter model identifier (e.g. ``"openai/gpt-4o"``).

        Returns:
            ``(True, success_message)`` or ``(False, error_message)``.
        """
        if not model or not model.strip():
            return False, "Model cannot be empty."

        if not getattr(self, "openrouter_api_key", None):
            return False, "No API key configured. Please initialize the agent first (/init)."

        self.openrouter_model = model.strip()
        success = self._setup_providers()

        if self.chain_openrouter is None or not success:
            return False, "Provider reinitialization failed. Please verify the model name."

        chat_ok, chat_message = self._check_chat_provider("openrouter")
        if not chat_ok:
            self.chain_openrouter = None
            return False, chat_message

        return True, f"Model switched to {self.openrouter_model}. Ready to help you."

    def get_available_models(self) -> "tuple[bool, list[str], str]":
        """
        Fetch the list of models available via the configured OpenRouter API key.

        Returns:
            ``(True, model_ids, "")`` on success, or ``(False, [], error_message)``.
        """
        import urllib.request as _ur
        import json as _json

        api_key = getattr(self, "openrouter_api_key", None)
        if not api_key:
            return False, [], "No API key configured. Please initialize the agent first (/init)."

        try:
            url = "https://openrouter.ai/api/v1/models"
            req = _ur.Request(url, headers={"Authorization": f"Bearer {api_key}"})
            with _ur.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode())
            models = sorted(entry["id"] for entry in data.get("data", []) if "id" in entry)
            return True, models, ""
        except Exception as exc:
            _LOGGER.warning("get_available_models error: %s", exc)
            return False, [], f"Could not fetch models: {exc}"

    def reset_connection(self) -> "tuple[bool, str]":
        """Clear the active cloud connection and revert the agent to the uninitialized state."""
        self.chain_ollama = None
        self.chain_openrouter = None
        self.openrouter_api_key = None
        self.openrouter_model = os.environ.get("OPENROUTER_MODEL", "~google/gemini-flash-latest")
        self.preferred_provider = "openrouter"

        return True, "Agent connection reset. Type /init to set up again."

    def _build_column_index(self):
        """Builds normalized token indexes and lightweight synonyms for column resolution."""
        self._cols = list(self.df_schema['columns'])
        self._col_tokens = {c: set(t for t in re.split(r"[ _/:\.]+", str(c).lower()) if t) for c in self._cols}
        self._column_synonyms = {
            "loss": {"loss", "error", "score"}, "score": {"score", "loss", "error"},
            "age": {"age"}, "label": {"label", "class", "target"},
            "origin": {"origin", "split", "dataset"}, "sample_id": {"sample_id", "id", "sample", "index"},
            "prediction": {"prediction", "predicted", "output", "predicted_class", "target_class"}
        }

    def _resolve_column(self, user_name: str) -> Optional[str]:
        """Robust column resolution handling synonyms and nested signals (//)."""
        if not user_name: return None

        # Normalize Input: lowercase, replace spaces AND SLASHES with underscores
        user_lower = user_name.strip().lower()
        user_clean = re.sub(r"[ /_]+", "_", user_lower) # "signals//train_loss" -> "signals_train_loss"

        # 1. Exact Match (Fast path)
        if user_name in self._cols: return user_name

        # Fuzzy matching (substring/token) must not accidentally resolve a
        # generic word (e.g. "loss") to a `tag:*` control column just because
        # the tag's user-chosen name happens to contain that word as a
        # substring (e.g. a request for "loss" wrongly matching
        # `tag:high_train_loss`, a boolean column, causing a numpy "boolean
        # subtract"/"ambiguous truth value" crash downstream). Control
        # columns should only be reachable via fuzzy match when the user is
        # explicitly talking about tags.
        user_mentions_tag = "tag" in user_lower
        fuzzy_candidates = self._cols if user_mentions_tag else [c for c in self._cols if not str(c).startswith("tag:")]

        # 2. Substring / Normalized Match (The Fix) - matches "train loss" OR "signals//train_loss" to "signals//train_loss/mlt_loss"
        for c in fuzzy_candidates:
            c_lower = c.lower()
            c_clean = re.sub(r"[ /_]+", "_", c_lower) # Normalize candidate: "signals//train_loss/mlt_loss" -> "signals_train_loss_mlt_loss"

            # Check if input is a subset of the column (e.g. "train_loss" in "signals_train_loss_mlt_loss") OR if input is the start of the column (more precise)
            if user_clean in c_clean:
                return c

        # 3. Token Set Ratio (Synonyms)
        # Use a more aggressive split that eats slashes
        user_tokens = set(re.split(r"[ _/]+", user_lower))
        for t in list(user_tokens):
            for base, syns in self._column_synonyms.items():
                if t == base or t in syns:
                    user_tokens.update(syns)

        best_col, best_score = None, 0.0
        for c in fuzzy_candidates:
            c_tokens = self._col_tokens.get(c)
            if not c_tokens: continue
            score = len(user_tokens & c_tokens) / len(user_tokens | c_tokens)
            if score > best_score:
                best_score, best_col = score, c

        return best_col if best_score > 0.3 else None

    # Semantic families for dataset split naming. Any dataset may store its
    # origin values under different spellings ("test", "test_split",
    # "test_loader", "inf_split", "holdout", ...); this lets a user's natural
    # word ("test", "inference") deterministically resolve to whatever the
    # ACTUAL stored value is, instead of relying on the LLM to guess the
    # exact spelling or on prompt-only regex heuristics.
    _SPLIT_VALUE_FAMILIES = {
        "train": {"train", "training", "tr"},
        "val": {"val", "valid", "validation", "dev"},
        "test": {
            "test", "testing", "eval", "evaluation", "inference", "inf",
            "holdout", "hold",
        },
    }

    @classmethod
    def _split_family(cls, token: str) -> Optional[str]:
        for family, words in cls._SPLIT_VALUE_FAMILIES.items():
            if token in words:
                return family
        return None

    def _resolve_categorical_value(self, column: str, value: str):
        """
        Maps a user-provided literal (e.g. "test") to the actual value present
        in `column`'s data (e.g. "inf_split"), so origin/split filters never
        silently return zero rows because of a naming mismatch. Falls back to
        the original literal if no confident match is found.
        """
        if not isinstance(value, str):
            return value

        candidates = self.df_schema['metadata'].get(column, {}).get('samples')
        if not candidates:
            return value

        raw_lower = value.strip().lower()

        # 1. Exact match (case-insensitive) -> use the dataset's own casing.
        for c in candidates:
            if str(c).lower() == raw_lower:
                return c

        # 2. Substring containment either way (e.g. "test" <-> "test_split").
        for c in candidates:
            c_lower = str(c).lower()
            if raw_lower in c_lower or c_lower in raw_lower:
                return c

        # 3. Split-family match (train/val/test/inference/holdout synonyms),
        # tokenizing both the user value and each candidate on separators.
        user_tokens = set(re.split(r"[ _/\-]+", raw_lower)) - {""}
        user_families = {self._split_family(t) for t in user_tokens} - {None}
        if user_families:
            for c in candidates:
                c_tokens = set(re.split(r"[ _/\-]+", str(c).lower())) - {""}
                c_families = {self._split_family(t) for t in c_tokens} - {None}
                if user_families & c_families:
                    return c

        # 4. Generic token-overlap fallback (same spirit as _resolve_column).
        best_c, best_score = None, 0.0
        for c in candidates:
            c_tokens = set(re.split(r"[ _/\-]+", str(c).lower())) - {""}
            if not c_tokens or not user_tokens:
                continue
            score = len(user_tokens & c_tokens) / len(user_tokens | c_tokens)
            if score > best_score:
                best_score, best_c = score, c

        return best_c if best_score > 0.3 else value

    # Matches plain integers/decimals and scientific notation (e.g. "2e-4",
    # "1.5E+10", "-0.003"), used as a last-resort numeric coercion for
    # ordering comparisons when a column's dtype couldn't be reliably
    # classified (e.g. a numeric signal column stored as pandas `object`).
    _NUMERIC_LITERAL_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

    @classmethod
    def _looks_numeric(cls, v) -> bool:
        return isinstance(v, str) and bool(cls._NUMERIC_LITERAL_RE.match(v.strip()))

    @classmethod
    def _coerce_numeric_literal(cls, v):
        """
        Best-effort numeric coercion (handles scientific notation like
        "2e-4"). Used as a fallback for ordering comparisons (>,<,>=,<=,
        between) when the target column's dtype metadata is missing or
        wrongly classified as categorical, so a numeric literal never
        survives as a string into a comparison and crashes at execution time
        (e.g. "'>' not supported between instances of 'float' and 'str'").
        Returns `v` unchanged (same object) if it doesn't look numeric.
        """
        if isinstance(v, (int, float)):
            return v
        if cls._looks_numeric(v):
            s = v.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    pass
        return v

    def _coalesce_same_column_equality(self, conditions: List[Condition]) -> List[Condition]:
        """
        All entries of a `conditions` list are AND-ed together when compiled
        into a mask. Two (or more) equality conditions on the SAME column are
        therefore always contradictory — a column can't equal two different
        literals at once — so that always-empty result is virtually never
        what was intended; it's almost always a mis-planned OR (e.g. "keep
        validation or test samples" emitted as `origin=='val'` AND
        `origin=='test'`). Deterministically collapse such groups into a
        single `in` condition so the OR semantics the user actually asked for
        survive regardless of how the plan phrased it.
        """
        if not conditions or len(conditions) < 2:
            return conditions

        groups: Dict[str, List[Condition]] = {}
        result: List[Optional[Condition]] = []
        slot_of: Dict[str, int] = {}

        for cond in conditions:
            op = (cond.op or "").lower()
            if op in ("==", "=", "equals"):
                key = self._resolve_column(cond.column) or cond.column
                groups.setdefault(key, []).append(cond)
                if key not in slot_of:
                    slot_of[key] = len(result)
                    result.append(None)  # reserved slot, filled in below
            else:
                result.append(cond)

        for key, group in groups.items():
            idx = slot_of[key]
            if len(group) == 1:
                result[idx] = group[0]
            else:
                result[idx] = Condition(column=group[0].column, op="in", value=[c.value for c in group])

        return [c for c in result if c is not None]

    def _rewrite_origin_literals(self, code: str) -> str:
        """
        Post-processes free-form generated code (`transform_code`/
        `analysis_expression`) so any string literal compared against the
        `origin` column goes through the same deterministic split-value
        resolution as structured `conditions` — this path (raw Python code
        the LLM writes directly, e.g. for `transform`/`analysis` kinds) does
        NOT go through `_build_python_mask`, so without this it silently
        matches nothing when the LLM writes `df['origin'] == 'train'` but the
        actual stored value is `'train_loader'`.

        Handles `df['origin'] ==/!= 'X'` and the index-level equivalent
        `df.index.get_level_values('origin') ==/!= 'X'`, plus `.isin([...])`
        on either form. Falls back to the original code unchanged if it
        isn't a single parseable expression.
        """
        origin_col = SampleStatsEx.ORIGIN.value
        if not code or origin_col not in code:
            return code
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError:
            return code

        agent = self

        def _is_origin_ref(node) -> bool:
            if isinstance(node, ast.Subscript):
                key = node.slice
                # Python <3.9 wraps the slice in an ast.Index; unwrap if present.
                if hasattr(ast, "Index") and isinstance(key, getattr(ast, "Index")):
                    key = key.value
                return (
                    isinstance(key, ast.Constant) and key.value == origin_col
                    and isinstance(node.value, ast.Name) and node.value.id == "df"
                )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_level_values":
                return bool(node.args) and isinstance(node.args[0], ast.Constant) and node.args[0].value == origin_col
            return False

        def _resolve_const(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                resolved = agent._resolve_categorical_value(origin_col, node.value)
                if resolved != node.value:
                    return ast.copy_location(ast.Constant(value=resolved), node)
            return node

        class _Rewriter(ast.NodeTransformer):
            def visit_Compare(self, node):
                self.generic_visit(node)
                if _is_origin_ref(node.left):
                    node.comparators = [_resolve_const(c) for c in node.comparators]
                elif node.comparators and _is_origin_ref(node.comparators[0]) and isinstance(node.left, ast.Constant):
                    node.left = _resolve_const(node.left)
                return node

            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "isin" and _is_origin_ref(node.func.value):
                    if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                        node.args[0].elts = [_resolve_const(e) for e in node.args[0].elts]
                return node

        try:
            new_tree = _Rewriter().visit(tree)
            ast.fix_missing_locations(new_tree)
            return ast.unparse(new_tree)
        except Exception:
            return code

    def _build_python_mask(self, conditions: List[Condition], n: Optional[int] = None) -> Optional[str]:
        """Builds an explicit Python boolean mask (df['col'] == val) for Index stability."""
        if not conditions: return None
        conditions = self._coalesce_same_column_equality(conditions)
        parts = []
        for cond in conditions:
            resolved_col = self._resolve_column(cond.column)
            if not resolved_col: continue

            # 1. Determine Accessor
            if resolved_col in self.df_schema['index_columns']:
                col_ref = f"df.index.get_level_values('{resolved_col}')"
            else:
                col_ref = f"df['{resolved_col}']"

            # 2. Normalize Operator
            op = cond.op.lower()
            if op == "=" or op == "equals": op = "==" # Fix "equals"

            val = cond.value

            # 3. Special "Max/Min" Logic (Self-referential filtering)
            if op == "max":
                if resolved_col not in self.df_schema['index_columns']:
                    if n and n > 1:
                        # Top N
                        parts.append(f"(df.index.isin({col_ref}.nlargest({n}).index))")
                    else:
                        # Top 1 (Tie breaker)
                        parts.append(f"(df.index == {col_ref}.idxmax())")
                else:
                    parts.append(f"({col_ref} == {col_ref}.max())")
                continue
            if op == "min":
                if resolved_col not in self.df_schema['index_columns']:
                    if n and n > 1:
                        parts.append(f"(df.index.isin({col_ref}.nsmallest({n}).index))")
                    else:
                        parts.append(f"(df.index == {col_ref}.idxmin())")
                else:
                    parts.append(f"({col_ref} == {col_ref}.min())")
                continue

            # 4. Type Match & Value Resolution
            is_col_ref = False
            if isinstance(val, str):
                # Treat condition values as literals by default.
                # Only convert to a column reference on exact schema-name match,
                # never via fuzzy resolution, to avoid errors like "train" -> "train_loss".
                raw_val = val.strip()
                possible_col = None

                if raw_val in self.df_schema['columns']:
                    possible_col = raw_val
                else:
                    raw_val_lower = raw_val.lower()
                    for schema_col in self.df_schema['columns']:
                        if str(schema_col).lower() == raw_val_lower:
                            possible_col = schema_col
                            break

                if possible_col:
                    if possible_col in self.df_schema['index_columns']:
                        val = f"df.index.get_level_values('{possible_col}')"
                    else:
                        val = f"df['{possible_col}']"
                    is_col_ref = True

            if not is_col_ref:
                # It's a literal. Apply type correction to fix Index mismatches.
                meta = self.df_schema['metadata'].get(resolved_col, {})
                dtype = str(meta.get('dtype', '')).lower()
                is_categorical_dtype = 'str' in dtype or 'object' in dtype or 'category' in dtype
                is_ordering_op = op in (">", "<", ">=", "<=", "between")

                def cast_v(v):
                    try:
                        if isinstance(v, str) and is_categorical_dtype:
                            # Deterministically map the user's wording to the
                            # actual stored value (e.g. "test" -> "inf_split")
                            # instead of relying on an LLM-guessed spelling.
                            v = self._resolve_categorical_value(resolved_col, v)
                        if 'int' in dtype: return int(v)
                        if 'float' in dtype: return float(v)
                        if is_ordering_op:
                            # dtype metadata is missing/mis-detected (e.g. a
                            # numeric signal column stored as pandas `object`).
                            # An ordering comparison against a numeric-looking
                            # literal (even scientific notation like "2e-4")
                            # is always intended numerically, so coerce it
                            # regardless of the (unreliable) dtype string.
                            coerced = self._coerce_numeric_literal(v)
                            if coerced is not v:
                                return coerced
                        if is_categorical_dtype: return str(v)
                    except: pass
                    return v

                if isinstance(val, list):
                    val = [cast_v(v) for v in val]
                else:
                    val = cast_v(val)

                # Prepare val2 if needed for 'between'
                val2 = getattr(cond, 'value2', None)
                if val2 is not None:
                    val2 = repr(cast_v(val2))

                val = repr(val)

            # 5. Build Expression
            if op == "==": parts.append(f"({col_ref} == {val})")
            elif op == "!=": parts.append(f"({col_ref} != {val})")
            elif op == ">": parts.append(f"({col_ref} > {val})")
            elif op == "<": parts.append(f"({col_ref} < {val})")
            elif op == ">=": parts.append(f"({col_ref} >= {val})")
            elif op == "<=": parts.append(f"({col_ref} <= {val})")
            elif op == "between":
                # Ensure we have a val2
                if 'val2' in locals() and val2 is not None:
                    parts.append(f"({col_ref}.between({val}, {val2}))")
                else:
                    parts.append(f"({col_ref} >= {val})") # Fallback to >= if between used improperly
            elif op == "contains": parts.append(f"({col_ref}.astype(str).str.contains({val}, na=False, regex=False))")
            elif op == "in":
                # Ensure val is a list for .isin()
                if not val.startswith('['):
                    val = f"[{val}]"
                parts.append(f"({col_ref}.isin({val}))")
            elif op == "not in":
                if not val.startswith('['):
                    val = f"[{val}]"
                parts.append(f"(~{col_ref}.isin({val}))")
            elif op == "is null": parts.append(f"({col_ref}.isna())")
            elif op == "is not null": parts.append(f"({col_ref}.notna())")

        return " & ".join(parts) if parts else None

    def _clean_code(self, code: str) -> str:
        if not code or not isinstance(code, str): return ""
        code = re.sub(r"^python", "", code, flags=re.IGNORECASE)
        code = re.sub(r"^code:", "", code, flags=re.IGNORECASE)
        if "```" in code:
            match = re.search(r"```(?:python)?\n?(.*?)\n?```", code, re.DOTALL)
            code = match.group(1).strip() if match else code
        return code.strip()

    def _coerce_discard_intent(self, intent: Intent) -> None:
        # Invariant: WL never removes dataframe rows. Any kind="drop" with
        # conditions is rewritten to flip the `discarded` deny-list flag.
        coerced = False
        for step in intent.steps:
            if step.kind != "drop" or not step.conditions:
                continue
            expr = self._build_python_mask(step.conditions, n=step.n)
            if expr:
                step.kind, step.target_column = "transform", "discarded"
                step.transform_code = f"np.where({expr}, True, df['discarded'])"
                coerced = True
        if coerced:
            intent.reasoning += " [Note: agent tried to drop samples from view; rewrote as deny-list flag (discarded=True). Rephrase as 'hide' or 'show only' if you wanted view-only filtering.]"

    def _is_agent_writable_column(self, col: Optional[str]) -> bool:
        """
        Invariant: the agent may create NEW columns freely (derived signals),
        but must never overwrite values already recorded in an existing
        column. The only existing columns it may write to are the control
        columns it owns: `discarded` and any `tag:*` boolean column.
        """
        if not col:
            return False
        if col not in self.df_schema.get('columns', []):
            return True  # Does not exist yet -> creating a new column is always allowed.
        return col == "discarded" or col.startswith("tag:")

    def _coerce_protected_transform_intent(self, intent: Intent) -> None:
        # Invariant: the agent must never mutate values in an existing data
        # column (signals, labels, metadata, sample_id, origin, ...). Steps
        # that target such a column are dropped; the user is told to ask for
        # a new derived column instead.
        blocked = []
        kept_steps = []
        for step in intent.steps:
            if step.kind == "transform" and not self._is_agent_writable_column(step.target_column):
                blocked.append(step.target_column)
                continue
            kept_steps.append(step)
        if blocked:
            intent.steps = kept_steps
            intent.reasoning += (
                f" [Safety: refused to overwrite existing column(s) {blocked} — the agent "
                "may only create new columns or update tag:*/discarded control columns. "
                "Ask for a new derived column (e.g. 'create <name> from ...') instead.]"
            )

    def _resolve_intent_to_ops(self, intent: Intent) -> List[dict]:
        """Applies safety coercions then converts an intent into executable ops.

        Multi-step intents may reference, in a LATER step's conditions, a
        column an EARLIER `transform` step in the same intent is about to
        create (e.g. "Tag X with A and B. Then discard these data." — the
        drop condition targets the tag column created a moment before).
        Temporarily register those pending columns so column resolution treats
        them as valid instead of silently failing to match.
        """
        pending_columns = [
            step.target_column for step in intent.steps
            if step.kind == "transform" and step.target_column and step.target_column not in self._cols
        ]
        self._cols.extend(pending_columns)
        try:
            self._coerce_discard_intent(intent)
            self._coerce_protected_transform_intent(intent)
            return self._intent_to_pandas_op(intent)
        finally:
            for col in pending_columns:
                if col in self._cols:
                    self._cols.remove(col)

    def _intent_to_pandas_op(self, intent: Intent) -> List[dict]:
            """Dispatches structured intent steps to registered handlers."""
            if intent.primary_goal == "out_of_scope":
                return [{"function": "out_of_scope", "params": {"reason": intent.reasoning}}]

            ops = []
            for step in intent.steps:
                handler = self.handlers.get(step.kind)

                # 1. Handle missing handler
                if not handler:
                    if step.kind != "noop":
                        _LOGGER.warning(f"No handler registered for kind: {step.kind}")
                    continue

                # 2. Execute Handler
                try:
                    # Pass step parameters AND intent context (reasoning)
                    result = handler.build_op(step, intent)
                    if not result:
                        continue

                    # 3. Normalize result to a list (removes the list vs dict indentation)
                    new_ops = result if isinstance(result, list) else [result]

                    # 4. Log and Store
                    for op in new_ops:
                        _LOGGER.info(f"[Agent] Generated Op: {op['function']} with params {op.get('params')}")

                    ops.extend(new_ops)

                except Exception as e:
                    _LOGGER.error(f"Handler {step.kind} failed: {e}")

            # Auto-cleanup: scratch columns created only to help compute a
            # LATER step's result (is_temporary=True) never remain in the
            # live dataframe once the whole request has finished executing.
            final_targets = {
                s.target_column for s in intent.steps
                if s.kind == "transform" and not s.is_temporary and s.target_column
            }
            temp_columns = [
                s.target_column for s in intent.steps
                if s.kind == "transform" and s.is_temporary and s.target_column
                and s.target_column not in final_targets
            ]
            for col in dict.fromkeys(temp_columns):  # de-dup, preserve order
                ops.append({"function": "df.drop_column", "params": {"col": col}})

            return ops

    def _parse_intent_from_response(self, name: str, intent) -> Optional[Intent]:
        text = intent.content if hasattr(intent, 'content') else str(intent)
        if not text or not text.strip():
            return None

        # 1. Isolate JSON block
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1:
            # No JSON at all -- the LLM likely refused, asked a clarifying
            # question, or got confused by an ambiguous/malformed prompt.
            # Always surface its own words back to the user (truncated)
            # instead of a generic "Internal Agent Error", regardless of how
            # long the response is.
            _LOGGER.info(f"[{name}] No JSON found in response. Wrapping as out_of_scope.")
            return Intent(reasoning=text[:800].strip(), primary_goal="out_of_scope", steps=[])

        # 2. Extract and pre-clean (avoid breaking valid escaped newlines)
        json_str = text[start:end+1]

        # 3. Robust clean: replace actual newlines with spaces only if they aren't escaped
        # (Very basic heuristic for malformed LLM outputs)
        json_str = json_str.replace('\n', ' ').replace('\r', ' ')
        # Collapse multiple spaces
        json_str = re.sub(r'\s+', ' ', json_str).strip()

        try:
            # Try parsing with standard json
            data = json.loads(json_str)
            return Intent(**data)
        except Exception as e:
            _LOGGER.error(f"[{name}] JSON decode failed: {e}. String: {json_str[:200]}...")

            # 4. Emergency fallback: try to fix missing quotes around keys/values or trailing commas
            try:
                fixed_json = re.sub(r'("\s*:\s*"[^"]*")\s*(")', r'\1, \2', json_str) # Add missing comma between "key":"val" "key"
                fixed_json = re.sub(r'(false|true|null|\d+)\s*(")', r'\1, \2', fixed_json) # Add missing comma after primitives

                # Try parsing again
                data = json.loads(fixed_json)
                return Intent(**data)
            except Exception:
                pass

            # 5. Could not parse or repair the JSON either -- still give the
            # user something actionable instead of a hard "failed to
            # generate a plan" error further up the call chain.
            _LOGGER.warning(f"[{name}] Could not parse or repair JSON; wrapping raw text as out_of_scope.")
            return Intent(
                reasoning=f"The agent's response could not be parsed as a valid plan. Raw response (truncated): {text[:500].strip()}",
                primary_goal="out_of_scope",
                steps=[],
            )

    def _query_langchain(self, name: str, chain, instruction: str, system_prompt: str) -> Optional[dict]:
        try:
            _LOGGER.info(f"[{name}] Invoking chain: '{instruction[:50]}...'")
            # Using double braces for prompt formatting
            escaped_sys = system_prompt.replace("{", "{{").replace("}", "}}").replace("{{instruction}}", "{instruction}")

            # NOTE: If using RAG, add {{examples}} replacement here

            prompt = ChatPromptTemplate.from_messages([("system", escaped_sys), ("human", "{instruction}")])

            # Optionally bind schema-validated structured output (Intent) so the
            # model returns a parsed object directly, skipping the free-form JSON
            # + regex-repair path in _parse_intent_from_response. Gated because
            # not every OpenRouter route supports it (see _load_config).
            runnable_chain = chain
            if name == "openrouter" and getattr(self, "openrouter_structured_output", False):
                try:
                    runnable_chain = chain.with_structured_output(Intent)
                except Exception as e:
                    _LOGGER.warning(f"[{name}] structured output unavailable, falling back to free-form JSON: {e}")
                    runnable_chain = chain

            # Time the pure LLM round-trip (see plan Phase A). This is the
            # dominant cost of a query; ~chars/4 is a rough token estimate.
            prompt_chars = len(escaped_sys) + len(instruction)
            model_name = getattr(self, f"{name}_model", None) or getattr(self, "openrouter_model", name)
            _t0 = time.perf_counter()
            response = (prompt | runnable_chain).invoke({"instruction": instruction})
            _llm_elapsed = time.perf_counter() - _t0

            _LOGGER.info(
                f"[{name}] LLM invoke took {_llm_elapsed:.2f}s "
                f"(model={model_name}, prompt~{prompt_chars} chars / ~{prompt_chars // 4} tokens)"
            )
            _LOGGER.info(f"[{name}] Response received")

            parsed_intent = None
            if isinstance(response, Intent):
                parsed_intent = response
            elif isinstance(response, dict):
                # with_structured_output can hand back a plain dict on some
                # langchain versions instead of the pydantic instance.
                try:
                    parsed_intent = Intent(**response)
                except Exception:
                    parsed_intent = self._parse_intent_from_response(name, response)
            else:
                parsed_intent = self._parse_intent_from_response(name, response)

            if parsed_intent:
                _LOGGER.info(f"[{name}] Reasoning: {parsed_intent.reasoning}")
                ops = self._resolve_intent_to_ops(parsed_intent)
                _LOGGER.info(f"[{name}] Converted to {len(ops)} operations")
                return ops
            return None

        except Exception as e:
            _LOGGER.warning(f"[{name}] Failed: {e}")
            # Remember the failure so query() can surface a specific message
            # (e.g. an auth/401 means the user isn't connected, not a planning bug).
            self._last_query_error = e
            return None

    @staticmethod
    def _is_auth_error(error) -> bool:
        """True when a provider error is an authentication/connection failure (401)."""
        if error is None:
            return False
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if status == 401:
            return True
        text = str(error).lower()
        return (
            "401" in text
            or "unauthorized" in text
            or "user not found" in text
            or "invalid api key" in text
            or "no auth credentials" in text
            or "authentication" in text
        )

    def _try_query_provider(self, provider: str, instruction: str, system_prompt: str) -> Optional[List[dict]]:
            # 1. Dynamically find the chain (chain_openrouter, chain_ollama)
            chain = getattr(self, f"chain_{provider}", None)

            # 2. If it exists, use the standard LangChain method
            if chain:
                return self._query_langchain(provider, chain, instruction, system_prompt)

            return None

    def query(self, instruction: str, abort_event: Optional[threading.Event] = None, status_callback: Optional[Callable[[str], None]] = None) -> List[dict]:
        _query_t0 = time.perf_counter()
        _LOGGER.info(f"[Agent] Query started: '{instruction}'")
        if abort_event and abort_event.is_set(): return []

        # Reset per-query error tracking so a stale failure doesn't leak forward.
        self._last_query_error = None

        # Time the schema build (cheap when cached; see plan Phase A/B.3).
        _schema_t0 = time.perf_counter()
        self._setup_schema()
        self._setup_model_schema()
        _LOGGER.info(f"[Agent] Schema build took {time.perf_counter() - _schema_t0:.3f}s")

        # 1. Format metadata for the prompt
        schema_lines = []
        for col, meta in self.df_schema['metadata'].items():
            if col in self.df_schema['index_columns']:
                tag = "[INDEX]"
            else:
                tag = "[COL]"

            line = f"- {tag} `{col}` ({meta['dtype']})"
            if col == SampleStatsEx.ORIGIN.value and "samples" in meta:
                # Show the actual split values (handles ANY naming scheme,
                # not just train/val/test/inference/holdout), but give an
                # explicit, mechanical matching rule instead of leaving it to
                # freeform guessing — that's what previously caused the LLM to
                # confuse e.g. train_loader/val_loader. If it still can't tell
                # confidently, it should fall back to the user's plain word,
                # which _resolve_categorical_value then resolves deterministically.
                line += (
                    f" | This is the dataset SPLIT column. Actual values: {self._format_samples_for_prompt(meta['samples'], max_items=20)}. "
                    "Match the user's split word to whichever listed value "
                    "TEXTUALLY CONTAINS that word, case-insensitively (e.g. "
                    "'validation'/'val' -> the value containing 'val'; "
                    "'test'/'inference' -> the one containing 'test' or 'inf'; "
                    "'train' -> the one containing 'train'). Never swap two "
                    "listed values for each other. If unsure which one matches, "
                    "write the user's own plain word instead of guessing."
                )
            elif "range" in meta:
                line += f" | Range: {meta['range'][0]:.3f} to {meta['range'][1]:.3f} | Mean: {meta['mean']:.3f}"
            elif "samples" in meta:
                line += f" | Samples: {self._format_samples_for_prompt(meta['samples'])} | Unique: {meta['unique_count']}"
            schema_lines.append(line)

        formatted_schema = "\n".join(schema_lines)

        # 2. Format the live model architecture for the prompt (may be unavailable)
        if self.model_available:
            model_schema_lines = [
                f"- Layer `{row['layer_id']}` (`{row['layer_name']}` / {row['layer_type']}): "
                f"neurons_count={row['neurons_count']}, incoming_neurons_count={row['incoming_neurons_count']}, "
                f"frozen={row['frozen']}"
                for _, row in self.model_layers_df.iterrows()
            ]
            formatted_model_schema = "\n".join(model_schema_lines)
        else:
            formatted_model_schema = "No model is currently registered."

        system_prompt = INTENT_PROMPT.format(
            schema=formatted_schema,
            row_count=self.df_schema['row_count'],
            model_schema=formatted_model_schema,
            history="\\n".join(self.history[-5:]) if self.history else "None"
        )

        order = [self.preferred_provider]
        if self.fallback_to_local and self.preferred_provider != "ollama":
            order.append("ollama")

        for provider in order:
            if abort_event and abort_event.is_set(): return []
            try:
                result = self._try_query_provider(provider, instruction, system_prompt)
                if result:
                    # Update History with Action
                    self.history.append(f"User: {instruction}")
                    self.history.append(f"Action: {len(result)} ops executed")
                    _LOGGER.info(f"[Agent] Query total wall time: {time.perf_counter() - _query_t0:.2f}s (provider={provider})")
                    return result
            except Exception as e:
                _LOGGER.error(f"Provider {provider} failed: {e}")
                continue

        # If we get here, all providers failed
        error_msg = "Internal Agent Error: Failed to generate a plan."
        if self._is_auth_error(self._last_query_error):
            # The provider rejected our credentials — the agent isn't
            # connected. Invalidate the cached client so is_available()/
            # CheckAgentHealth immediately stop reporting "available" for a
            # connection that just proved broken, instead of leaving the
            # health check permanently stale until the process restarts.
            self.chain_openrouter = None
            error_msg = (
                "Agent not connected: the LLM provider rejected the request "
                "(401 Unauthorized). Check your API key and re-initialize the "
                "agent with /init."
            )
        elif not self.is_ollama_available() and not os.environ.get("OPENROUTER_API_KEY"):
            error_msg = "No LLM providers configured. Please check your API keys or local Ollama setup. Initialize the agent with /init."

        _LOGGER.info(f"[Agent] Query total wall time: {time.perf_counter() - _query_t0:.2f}s (no provider succeeded)")
        return [{"function": "out_of_scope", "params": {"reason": error_msg}}]

    # ------------------------------------------------------------------
    # Notebook code generation
    # ------------------------------------------------------------------

    def _compact_schema_for_prompt(self) -> str:
        """Return a short "- `col` (dtype)" listing of the live dataframe columns.

        Reuses the cached schema built for query() so notebook code-gen does not
        pay a second full stats pass.
        """
        try:
            self._setup_schema()
            meta = (self.df_schema or {}).get("metadata", {})
            lines = [f"- `{col}` ({info.get('dtype', '?')})" for col, info in meta.items()]
            return "\n".join(lines) if lines else "(no dataframe columns available yet)"
        except Exception as exc:
            _LOGGER.debug("notebook schema build failed: %s", exc)
            return "(dataframe schema unavailable)"

    @staticmethod
    def _extract_code_and_explanation(text: str):
        """Split an LLM reply into (code, explanation).

        Prefers a fenced ```python block; falls back to the first generic fence,
        then to treating the whole reply as code when no fence is present.
        """
        if text is None:
            return "", ""
        fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence:
            code = fence.group(1).strip()
            explanation = (text[:fence.start()] + text[fence.end():]).strip()
            return code, explanation
        return text.strip(), ""

    def generate_code(self, prompt: str, context_code: str = ""):
        """Propose Python for a notebook cell from a natural-language ``prompt``.

        This does NOT execute anything and does NOT touch the intent pipeline; it
        only asks the active LLM provider for runnable code. Returns a
        ``(code, explanation)`` tuple. Raises RuntimeError when no provider is
        configured so the service can report a clean error.
        """
        if not self.is_available():
            raise RuntimeError(
                "Agent not configured. Initialize a provider with /init (or run a "
                "local Ollama server) before using \">\" notebook cells."
            )

        system_prompt = NOTEBOOK_CODE_PROMPT.format(
            schema=self._compact_schema_for_prompt(),
            context_code=(context_code or "(none)").strip(),
        )
        # Escape braces the same way _query_langchain does so ChatPromptTemplate
        # does not treat stray { } in the schema/context as template variables.
        escaped_sys = system_prompt.replace("{", "{{").replace("}", "}}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", escaped_sys), ("human", "{instruction}")]
        )

        order = [self.preferred_provider]
        if self.fallback_to_local and self.preferred_provider != "ollama":
            order.append("ollama")

        last_error = None
        for provider in order:
            chain = getattr(self, f"chain_{provider}", None)
            if chain is None:
                continue
            try:
                response = (chat_prompt | chain).invoke({"instruction": prompt})
                text = getattr(response, "content", None)
                if text is None:
                    text = str(response)
                code, explanation = self._extract_code_and_explanation(text)
                self.history.append(f"User (notebook): {prompt}")
                self.history.append("Action: proposed notebook code")
                return code, explanation
            except Exception as exc:
                last_error = exc
                _LOGGER.warning("[notebook code-gen] provider %s failed: %s", provider, exc)
                continue

        raise RuntimeError(
            f"Code generation failed: {last_error}" if last_error
            else "Code generation failed: no provider produced a response."
        )
