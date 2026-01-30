import os
import logging
import difflib
import re
import pandas as pd
import threading
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Literal, Callable, Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import yaml

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Ensure intent_prompt is accessible
from .intent_prompt import INTENT_PROMPT

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
    kind: Literal["keep", "drop", "sort", "group", "head", "tail", "reset", "analysis", "transform", "action", "noop", "clarify"] = Field(
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
    
    # Future-proofing for Actions
    action_name: Optional[str] = Field(default=None, description="Name of the action (e.g. 'save_dataset')")
    action_params: Optional[Dict[str, Any]] = Field(default=None, description="Parameters for the action")

class Intent(BaseModel):
    reasoning: str = Field(description="The thought process or clarification question.")
    primary_goal: Literal["ui_manipulation", "data_analysis", "action", "out_of_scope"] = Field(
        description="Whether the user wants to change the grid view, get an answer, or perform an action."
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
            expr = self.agent._clean_code(step.analysis_expression)
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
            prefix = match.group(1)  # e.g. ['
            content = match.group(2) # e.g. signals//train_loss
            suffix = match.group(3)  # e.g. ']
            
            resolved = self.agent._resolve_column(content) # Try to resolve the content to a real column
            
            if resolved:
                return f"{prefix}{resolved}{suffix}" # If we found a better match in the schema, use it
            
            return match.group(0) # If it's a value (e.g., 'bug' in tags), leave it alone

        fixed_code = re.sub(pattern, replace_col, raw_code)
        
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
        
        return {
            "function": "df.modify",
            "params": {
                "col": step.target_column, 
                "code": fixed_code
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


# ==========================================
# 3. THE AGENT (Orchestrator)
# ==========================================

class DataManipulationAgent:
    def __init__(self, context):
        """Initializes the agent with context and builds the column schema/index."""
        _LOGGER.info("Initializing DataManipulationAgent")
        self.ctx = context

        self._setup_schema()
        self._build_column_index()
        self._load_config()
        self._setup_providers()
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
            "noop": None
        }

    def _setup_schema(self):
        """Builds a rich column schema with statistical context for the LLM."""
        df = self.ctx._all_datasets_df
        all_columns = df.columns.tolist()
        index_columns = []

        if isinstance(df.index, pd.MultiIndex):
            for name in df.index.names:
                if name is not None:
                    index_columns.append(name)
                    if name not in all_columns:
                        all_columns.append(name)
        elif df.index.name is not None:
            index_columns.append(df.index.name)
            if df.index.name not in all_columns:
                all_columns.append(df.index.name)

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
            'row_count': len(df)
        }
        self._build_column_index()

    def _load_config(self):
        self.preferred_provider = "ollama"
        self.google_model = "gemini-1.5-flash-latest"
        self.openai_model = "gpt-4o-mini"
        self.openrouter_model = "mistralai/mistral-7b-instruct:free"
        self.fallback_to_local = True
        self.ollama_host = "localhost"
        self.ollama_port = "11435"
        self.ollama_model = "qwen2.5:3b-instruct"

        repo_root = Path(__file__).resolve().parents[4]
        inner_pkg = Path(__file__).resolve().parents[3]

        env_paths = [repo_root / ".env", inner_pkg / ".env"]
        for ep in env_paths:
            if ep.exists():
                load_dotenv(dotenv_path=ep)
                _LOGGER.info(f"Loaded credentials from {ep}")
                break

        config_paths = [repo_root / "agent_config.yaml", inner_pkg / "agent_config.yaml", Path.cwd() / "agent_config.yaml"]
        for path in config_paths:
            if not path.exists(): continue
            try:
                with open(path, 'r') as f:
                    cfg = yaml.safe_load(f)
                if not cfg or "agent" not in cfg: continue
                a_cfg = cfg["agent"]
                self.preferred_provider = a_cfg.get("provider", self.preferred_provider).lower()
                self.google_model = a_cfg.get("google_model", self.google_model)
                self.openai_model = a_cfg.get("openai_model", self.openai_model)
                self.openrouter_model = a_cfg.get("openrouter_model", self.openrouter_model)
                self.fallback_to_local = a_cfg.get("fallback_to_local", self.fallback_to_local)
                self.ollama_host = a_cfg.get("ollama_host", self.ollama_host)
                self.ollama_port = a_cfg.get("ollama_port", self.ollama_port)
                self.ollama_model = a_cfg.get("ollama_model", self.ollama_model)
                _LOGGER.info(f"Applied agent configuration from {path}")
                break
            except Exception as e:
                _LOGGER.warning(f"Error loading config from {path}: {e}")

    def _setup_providers(self):
        self.chain_openai = None
        self.chain_google = None
        self.chain_ollama = None
        self.chain_openrouter = None

        # Determine which providers to initialize
        active_providers = {self.preferred_provider}
        if self.fallback_to_local:
            active_providers.add("ollama")

        # OPEN AI
        if "openai" in active_providers and os.environ.get("OPENAI_API_KEY"):
            try:
                llm = ChatOpenAI(model=self.openai_model, temperature=0)
                self.chain_openai = llm.with_structured_output(Intent)
                _LOGGER.info(f"[Agent] OpenAI enabled: {self.openai_model}")
            except Exception as e: _LOGGER.error(f"OpenAI error: {e}")

        # GOOGLE
        if "google" in active_providers and os.environ.get("GOOGLE_API_KEY"):
            try:
                llm = ChatOpenAI(
                    model=self.google_model,
                    temperature=0,
                    api_key=os.environ.get("GOOGLE_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    max_retries=1
                )
                self.chain_google = llm
                _LOGGER.info(f"[Agent] Google Gemini enabled: {self.google_model}")
            except Exception as e: _LOGGER.error(f"Google error: {e}")

        # OPEN ROUTER
        if "openrouter" in active_providers and os.environ.get("OPENROUTER_API_KEY"):
            try:
                llm = ChatOpenAI(
                    model=self.openrouter_model, temperature=0,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    streaming=False, max_retries=1, request_timeout=15.0,
                )
                self.chain_openrouter = llm
                _LOGGER.info(f"[Agent] OpenRouter enabled: {self.openrouter_model}")
            except Exception as e: _LOGGER.error(f"OpenRouter error: {e}")

        # LOCAL
        if "ollama" in active_providers:
            try:
                host = self.ollama_host.split(':')[0]
                port = self.ollama_port
                llm = ChatOllama(base_url=f"http://{host}:{port}", model=self.ollama_model, temperature=0, timeout=15)
                self.chain_ollama = llm
                _LOGGER.info(f"[Agent] Ollama enabled: {self.ollama_model}")
            except Exception as e: _LOGGER.error(f"Ollama error: {e}")
            
    def is_ollama_available(self) -> bool:
        return self.chain_ollama is not None

    def _build_column_index(self):
        self._cols = list(self.df_schema['columns'])
        self._col_tokens = {c: set(t for t in re.split(r"[ _/\.]+", c.lower()) if t) for c in self._cols}
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
        user_clean = re.sub(r"[ /_]+", "_", user_lower)  # "signals//train_loss" -> "signals_train_loss"

        # 1. Exact Match (Fast path)
        if user_name in self._cols: return user_name

        # 2. Substring / Normalized Match (The Fix) - matches "train loss" OR "signals//train_loss" to "signals//train_loss/mlt_loss"
        for c in self._cols:
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
        for c, c_tokens in self._col_tokens.items():
            if not c_tokens: continue
            score = len(user_tokens & c_tokens) / len(user_tokens | c_tokens)
            if score > best_score:
                best_score, best_col = score, c

        return best_col if best_score > 0.3 else None

    def _build_python_mask(self, conditions: List[Condition], n: Optional[int] = None) -> Optional[str]:
        """Builds an explicit Python boolean mask (df['col'] == val) for Index stability."""
        if not conditions: return None
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
            if op == "=" or op == "equals": op = "=="  # Fix "equals"
            
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
                possible_col = self._resolve_column(val)
                if possible_col and possible_col in self.df_schema['columns']:
                    if possible_col in self.df_schema['index_columns']:
                         val = f"df.index.get_level_values('{possible_col}')"
                    else:
                         val = f"df['{possible_col}']"
                    is_col_ref = True
            
            if not is_col_ref:
                # It's a literal. Apply type correction to fix Index mismatches.
                meta = self.df_schema['metadata'].get(resolved_col, {})
                dtype = str(meta.get('dtype', '')).lower()
                
                def cast_v(v):
                    try:
                        if 'int' in dtype: return int(v)
                        if 'float' in dtype: return float(v)
                        if 'str' in dtype or 'object' in dtype: return str(v)
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

            return ops

    def _parse_intent_from_response(self, name: str, intent) -> Optional[Intent]:
        text = intent.content if hasattr(intent, 'content') else str(intent)
        
        # 1. Isolate JSON block
        start = text.find('{')
        end = text.rfind('}')
        if start == -1 or end == -1: 
            # If no JSON found, check if it's a short text (likely a refusal or out-of-scope reply)
            if len(text) < 500:
                _LOGGER.info(f"[{name}] No JSON found, but text is short. Wrapping as out_of_scope.")
                return Intent(
                    reasoning=text,
                    primary_goal="out_of_scope",
                    steps=[]
                )
            _LOGGER.error(f"[{name}] No JSON braces found in response: {text[:200]}...")
            return None
        
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
            except:
                pass
                
            return None

    def _query_langchain(self, name: str, chain, instruction: str, system_prompt: str) -> Optional[dict]:
        try:
            _LOGGER.info(f"[{name}] Invoking chain: '{instruction[:50]}...'")
            # Using double braces for prompt formatting
            escaped_sys = system_prompt.replace("{", "{{").replace("}", "}}").replace("{{instruction}}", "{instruction}")
            
            # NOTE: If using RAG, add {{examples}} replacement here
            
            prompt = ChatPromptTemplate.from_messages([("system", escaped_sys), ("human", "{instruction}")])
            response = (prompt | chain).invoke({"instruction": instruction})
            
            _LOGGER.info(f"[{name}] Response received")
            
            parsed_intent = None
            if isinstance(response, Intent):
                parsed_intent = response
            else:
                parsed_intent = self._parse_intent_from_response(name, response)
            
            if parsed_intent:
                _LOGGER.info(f"[{name}] Reasoning: {parsed_intent.reasoning}")
                ops = self._intent_to_pandas_op(parsed_intent)
                _LOGGER.info(f"[{name}] Converted to {len(ops)} operations")
                return ops
            return None
            
        except Exception as e:
            _LOGGER.warning(f"[{name}] Failed: {e}")
            return None

    def _try_query_provider(self, provider: str, instruction: str, system_prompt: str) -> Optional[List[dict]]:
            # 1. Dynamically find the chain (chain_openai, chain_google, chain_ollama)
            chain = getattr(self, f"chain_{provider}", None)
            
            # 2. If it exists, use the standard LangChain method
            if chain:
                return self._query_langchain(provider, chain, instruction, system_prompt)
                
            return None

    def query(self, instruction: str, abort_event: Optional[threading.Event] = None, status_callback: Optional[Callable[[str], None]] = None) -> List[dict]:
        _LOGGER.info(f"[Agent] Query started: '{instruction}'")
        if abort_event and abort_event.is_set(): return []

        self._setup_schema()
        
        # 1. Format metadata for the prompt
        schema_lines = []
        for col, meta in self.df_schema['metadata'].items():
            if col in self.df_schema['index_columns']:
                tag = "[INDEX]"
            else:
                tag = "[COL]"
            
            line = f"- {tag} `{col}` ({meta['dtype']})"
            if "range" in meta:
                line += f" | Range: {meta['range'][0]:.3f} to {meta['range'][1]:.3f} | Mean: {meta['mean']:.3f}"
            elif "samples" in meta:
                line += f" | Samples: {meta['samples']} | Unique: {meta['unique_count']}"
            schema_lines.append(line)

        formatted_schema = "\n".join(schema_lines)

        system_prompt = INTENT_PROMPT.format(
            schema=formatted_schema,
            row_count=self.df_schema['row_count'],
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
                    return result
            except Exception as e:
                _LOGGER.error(f"Provider {provider} failed: {e}")
                continue

        # If we get here, all providers failed
        error_msg = "Internal Agent Error: Failed to generate a plan."
        if not self.is_ollama_available() and not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            error_msg = "No LLM providers configured. Please check your API keys or Ollama status."
            
        return [{"function": "out_of_scope", "params": {"reason": error_msg}}]