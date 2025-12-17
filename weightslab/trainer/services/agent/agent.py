import os
import json
import logging
import requests
import difflib
import re
import pandas as pd

from dataclasses import dataclass
from typing import Optional, List, Union
from .intent_prompt import INTENT_PROMPT


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOGGER = logging.getLogger(__name__)

# ALLOWED_METHODS is now purely for reference; the logic is governed by Intent.kind
ALLOWED_METHODS = {"drop", "sort_values", "query", "head", "tail", "sample"}

# FUNCTION_SYNONYMS is no longer strictly needed but kept for potential future use
# or internal column resolution guidance.
FUNCTION_SYNONYMS = {
    # show / list rows
    "list": "head",
    "show": "head",
    "display": "head",
    "preview": "head",
    "view": "head",
    "print": "head",
    "head": "head",
    "first": "head",
    "top": "head",
    "top_rows": "head",

    # last rows
    "tail": "tail",
    "last": "tail",
    "bottom": "tail",
    "bottom_rows": "tail",

    # sorting
    "sort": "sort_values",
    "sortby": "sort_values",
    "order": "sort_values",
    "orderby": "sort_values",
    "order_by": "sort_values",
    "rank": "sort_values",

    # filtering / keeping (mapped to query)
    "filter": "query",
    "where": "query",
    "select": "query",
    "keep": "query",
    "only": "query",

    # dropping / deleting rows
    "drop_rows": "drop",
    "remove": "drop",
    "delete": "drop",
    "exclude": "drop",
    "drop": "drop",

    # randomness
    "random": "sample",
    "sample_rows": "sample",
    "shuffle": "sample",
    "sample": "sample",
}

@dataclass
class Condition:
    column: str           # user-specified column name (we'll resolve it)
    op: str               # "==", "!=", ">", "<", ">=", "<=", "between"
    value: Optional[Union[float, int, str]] = None
    value2: Optional[Union[float, int]] = None  # for between


@dataclass
class Intent:
    # "keep" (filter rows), "drop" (remove rows), "sort", "head", "tail", "reset", "noop"
    kind: str

    # For keep/drop
    conditions: Optional[List[Condition]] = None

    # For sort
    sort_by: Optional[List[str]] = None
    ascending: Optional[bool] = None

    # For head/tail
    n: Optional[int] = None

    # Optional sampling for drop (e.g. "drop 50% of rows …")
    drop_frac: Optional[float] = None

    # Optional sampling for keep (e.g. "keep 80% of rows …")
    keep_frac: Optional[float] = None


class DataAgentError(Exception):
    """Custom exception for Data Manipulation Agent errors."""
    pass


class DataManipulationAgent:
    def __init__(self, ctx):
        _LOGGER.info("Initializing DataManipulationAgent")
        self.ctx = ctx
        df = ctx._all_datasets_df

        # Include both regular columns AND index columns (e.g., sample_id, origin)
        # so they can be used in queries
        all_columns = df.columns.tolist()
        if isinstance(df.index, pd.MultiIndex):
            # Add index level names if using MultiIndex
            all_columns.extend([name for name in df.index.names if name is not None])
        elif df.index.name is not None:
            # Add single index name if it exists
            all_columns.append(df.index.name)
        
        # Add expected extended stats columns that will be populated during training
        # This ensures the agent recognizes them even before they appear in the DataFrame
        expected_extended_stats = [
            "mean_loss", "max_loss", "min_loss", "std_loss", "median_loss",
            "num_classes_present", "dominant_class", "dominant_class_ratio",
            "background_ratio", "predicted_class"
        ]
        # Also add per-class loss columns (up to 10 classes)
        for i in range(10):
            expected_extended_stats.append(f"loss_class_{i}")
        
        # Add expected columns that aren't already in the DataFrame
        for col in expected_extended_stats:
            if col not in all_columns:
                all_columns.append(col)

        self.df_schema = {
            'columns': all_columns,
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        }
        _LOGGER.info("Agent initialized with schema: columns=%s", self.df_schema['columns'])

        self._build_column_index()

        self._check_ollama_health()

    def _build_column_index(self):
        """
        Precompute tokenized view of column names for fuzzy matching.
        """
        self._cols = list(self.df_schema['columns'])
        self._col_tokens = {}
        for c in self._cols:
            tokens = re.split(r"[ _/\.]+", c.lower())
            self._col_tokens[c] = set(t for t in tokens if t)

        # generic synonym hints (for robustness)
        self._column_synonyms = {
            "loss": {"loss", "error", "score"},
            "score": {"score", "loss", "error"},
            "age": {"age"},
            "label": {"label", "class", "target"},
            "origin": {"origin", "split", "dataset"},
            "sample_id": {"sample_id", "id", "sample", "index"},
        }

    def _resolve_column(self, user_name: str) -> Optional[str]:
        """
        Map a user-provided column name (possibly with typos) to an actual
        DataFrame column. Uses:
          - case-insensitive exact match
          - token overlap / synonyms
          - difflib distance
        Returns None if nothing plausible is found.
        """
        if not user_name:
            return None

        user_name = user_name.strip().lower()
        cols = self._cols

        # 1) exact (case-insensitive)
        for c in cols:
            if c.lower() == user_name:
                return c

        # 2) token overlap / synonyms
        user_tokens = set(re.split(r"[ _/\.]+", user_name))
        expanded_tokens = set(user_tokens)

        for base, syns in self._column_synonyms.items():
            if base in user_tokens or user_name in syns:
                expanded_tokens |= syns

        best_col = None
        best_score = 0.0
        for c in cols:
            ct = self._col_tokens.get(c, set())
            if not ct:
                continue
            jacc = len(expanded_tokens & ct) / len(expanded_tokens | ct)
            if jacc > best_score:
                best_score = jacc
                best_col = c

        # 3) fallback: difflib if overlap is weak
        if best_score < 0.2:
            close = difflib.get_close_matches(user_name, cols, n=1, cutoff=0.6)
            if close:
                best_col = close[0]
                best_score = 0.6

        if best_col is not None:
            _LOGGER.info("Resolved user column %r -> %r (score=%.2f)", user_name, best_col, best_score)
            return best_col

        _LOGGER.warning("Could not resolve user column %r to any schema column", user_name)
        return None

    def _check_ollama_health(self):
        """Check if Ollama is running and accessible."""
        _LOGGER.info("Checking Ollama health...")
        os.environ['OLLAMA_HOST'] = os.environ.get('OLLAMA_HOST', 'localhost').split(':')[0]
        try:
            response = requests.get(
                f"http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}/api/tags", 
                timeout=1  # Reduced from 5 to 1 second for faster initialization
            )
            if response.status_code == 200:
                models = response.json().get('models', [])
                _LOGGER.info("Ollama is running with models: %s", [m.get('name') for m in models])
                if not any('llama3.2:1b' == m.get('name', '') for m in models):
                    _LOGGER.warning(
                        "llama3.2:1b model not found in Ollama. Available models: %s",
                        [m.get('name') for m in models]
                    )
            else:
                _LOGGER.error("Ollama health check failed with status: %s", response.status_code)
        except requests.RequestException as e:
            _LOGGER.error(f"Ollama is not accessible at http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}: %s", e)
            raise DataAgentError("Ollama service is not running. Please start Ollama first.") from e

    def _parse_intent_json(self, json_str: str) -> Intent:
        """Parses the raw LLM JSON output into a structured Intent object."""
        # Minimal cleanup for common LLM markdown output
        json_str = json_str.replace('```json', '').replace('```', '').strip()
        json_str = json_str.replace('False', 'false').replace('True', 'true').replace('None', 'null')

        try:
            data = json.loads(json_str)

            if isinstance(data, list) and data:
                data = data[-1]

            kind = data.get("kind")
            if kind not in ["keep", "drop", "sort", "head", "tail", "reset", "noop"]:
                _LOGGER.warning("Invalid intent kind received: %s. Defaulting to noop.", kind)
                kind = "noop"

            # --- Parse conditions first ---
            conditions_list: list[Condition] = []
            for c_data in data.get("conditions", []) or []:
                conditions_list.append(Condition(
                    column=c_data.get("column", ""),
                    op=c_data.get("op", "=="),
                    value=c_data.get("value"),
                    value2=c_data.get("value2")
                ))

            # --- Normalize / infer sort_by ---
            raw_sort_by = data.get("sort_by")
            sort_by: Optional[list[str]] = None

            if isinstance(raw_sort_by, str):
                sort_by = [raw_sort_by]
            elif isinstance(raw_sort_by, list):
                sort_by = raw_sort_by
            else:
                # NO sort_by provided: try to infer from conditions
                if kind == "sort" and conditions_list:
                    inferred_cols: list[str] = []
                    for cond in conditions_list:
                        # LLM pattern: put sort columns in conditions with value == null
                        # We just use their column names in order.
                        col_name = cond.column
                        if col_name and col_name not in inferred_cols:
                            inferred_cols.append(col_name)

                    if inferred_cols:
                        sort_by = inferred_cols
                        _LOGGER.info(
                            "Inferred sort_by from conditions for sort intent: %s",
                            sort_by,
                        )

            return Intent(
                kind=kind,
                conditions=conditions_list if conditions_list else None,
                sort_by=sort_by,
                ascending=data.get("ascending"),
                n=data.get("n"),
                drop_frac=data.get("drop_frac"),
                keep_frac=data.get("keep_frac"),
            )

        except json.JSONDecodeError as e:
            _LOGGER.error("Failed to parse LLM JSON into Intent: %s. JSON: %s", e, json_str[:500])
            return Intent(kind="noop")
        except Exception as e:
            _LOGGER.error("Error creating Intent object: %s", e)
            return Intent(kind="noop")

    def _build_condition_string(self, conditions: List[Condition]) -> Optional[str]:
        """Safely generates the pandas query expression string from a list of Conditions."""
        if not conditions:
            return None

        # Get the actual DataFrame to check if columns are in index
        df = self.ctx._all_datasets_df
        index_names = []
        if isinstance(df.index, pd.MultiIndex):
            index_names = [name for name in df.index.names if name is not None]
        elif df.index.name is not None:
            index_names = [df.index.name]

        parts = []
        for cond in conditions:
            # 1. Resolve column name safely
            resolved_col = self._resolve_column(cond.column)
            if not resolved_col:
                _LOGGER.warning("Skipping condition due to unresolvable column: %s", cond.column)
                continue

            # 2. Determine column reference
            # For index columns and simple column names, use direct reference
            # For columns with special chars, use backticks
            is_index_col = resolved_col in index_names
            
            if re.match(r'^[\w]+$', resolved_col):
                # Simple alphanumeric name - use as-is (works for both index and regular columns)
                col_ref = resolved_col
            else:
                # Complex name with spaces/special chars - use backticks
                col_ref = f"`{resolved_col}`"

            # 3. Build expression based on operator
            op = cond.op.lower()
            val = cond.value
            
            # Simple operators
            if op in ("==", "!=", ">", "<", ">=", "<="):
                # Quote string values, keep numbers as is
                value_repr = f"'{val}'" if isinstance(val, str) else str(val)
                parts.append(f"({col_ref} {op} {value_repr})")
            
            # Between
            elif op == "between" and cond.value is not None and cond.value2 is not None:
                min_val = float(cond.value)
                max_val = float(cond.value2)
                parts.append(f"({col_ref}.between({min_val}, {max_val}))")
                
            else:
                _LOGGER.warning("Skipping condition due to invalid operator or values: %s", cond)

        return " and ".join(parts) if parts else None

    def _intent_to_pandas_op(self, intent: Intent) -> dict:
        """
        Converts a structured Intent object into the final Pandas operation dictionary.
        This is the new, safe code generation core.
        """
        op = {"function": None, "params": {}}

        if intent.kind == "head" and intent.n is not None:
            op["function"] = "df.head"
            op["params"] = {"n": max(0, int(intent.n))}
        
        elif intent.kind == "tail" and intent.n is not None:
            op["function"] = "df.tail"
            op["params"] = {"n": max(0, int(intent.n))}

        elif intent.kind == "sort" and intent.sort_by:
            # Safely resolve all sort columns
            resolved_cols = []
            for col in intent.sort_by:
                resolved = self._resolve_column(col)
                if resolved:
                    resolved_cols.append(resolved)
            
            if resolved_cols:
                op["function"] = "df.sort_values"
                op["params"] = {
                    "by": resolved_cols,
                    "ascending": bool(intent.ascending) if intent.ascending is not None else True
                }
            else:
                _LOGGER.warning("Sort instruction contained no valid columns.")

        elif intent.kind in ("keep", "drop") and intent.conditions:
            condition_expr = self._build_condition_string(intent.conditions)

            if not condition_expr:
                _LOGGER.warning("Filter/Drop instruction resulted in no valid condition expression.")
                return op

            # safe string literal for use inside df.query(...)
            cond_repr = repr(condition_expr)

            if intent.kind == "keep":
                # --- plain keep: filter by condition, let ApplyDataQuery do in-place filter ---
                if not intent.keep_frac or not (0 < intent.keep_frac <= 1):
                    op["function"] = "df.query"
                    op["params"] = {"expr": condition_expr}
                else:
                    frac = float(intent.keep_frac)
                    index_expr = (
                        "df.index.difference("
                        f"df.query({cond_repr}).sample(frac={frac}).index"
                        ")"
                    )
                    op["function"] = "df.drop"
                    op["params"] = {"index": index_expr}

            elif intent.kind == "drop":
                op["function"] = "df.drop"

                # Base index string: df.query(condition_expr).index
                index_expr = f"df.query({cond_repr}).index"

                # Apply sampling if specified
                if intent.drop_frac and 0 < intent.drop_frac <= 1:
                    frac = float(intent.drop_frac)
                    # df.query(condition_expr).sample(frac=drop_frac).index
                    index_expr = f"df.query({cond_repr}).sample(frac={frac}).index"

                op["params"] = {"index": index_expr}

        elif intent.kind == "reset":
            # When the LLM understands the user wants to reset the view
            op["function"] = "df.reset_view"  # Arbitrary string for logging/debug
            op["params"] = {"__agent_reset__": True}  # The signal the DataService checks for
        
        # If noop, this stays {"function": None, "params": {}}
        return op

    def query(self, instruction: str) -> dict:
        """Main entry point to query the agent: call Ollama and return a Pandas operation."""
        # Load Intent Prompt
        prompt = INTENT_PROMPT.format(instruction=instruction, columns=self.df_schema['columns'])
        
        try:
            response = requests.post(
                f"http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}/api/generate?source=data-agent",
                json={
                    'model': 'llama3.2:1b',
                    'prompt': prompt,
                    'format': 'json',
                    'stream': False,
                    'options': {
                        'num_predict': 512,
                    },
                },
                timeout=600
            )
            
            _LOGGER.debug("Ollama response status: %s", response.status_code)
        except requests.RequestException as e:
            raise DataAgentError(f"Ollama request failed: {e}") from e

        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            _LOGGER.debug("Ollama raw response: %s", result)

            if not result:
                _LOGGER.error("Ollama returned empty response")
                return {"function": None, "params": {}}

            # --- MINIMAL JSON CLEANUP & PARSING ---
            intent = self._parse_intent_json(result)

            # --- OPTIONAL: infer frac from natural language if LLM omitted it ---
            try:
                m = re.search(r'(\d+)\s*%', instruction)
                if m:
                    frac = int(m.group(1)) / 100.0
                    if 0 < frac <= 1:
                        if intent.kind == "keep" and not intent.keep_frac:
                            intent.keep_frac = frac
                        elif intent.kind == "drop" and not intent.drop_frac:
                            intent.drop_frac = frac
            except Exception as e:
                _LOGGER.debug("Failed to infer percentage from instruction %r: %s", instruction, e)

            # --- TRANSLATE INTENT TO PANDAS OPERATION ---
            operation = self._intent_to_pandas_op(intent)
            
            _LOGGER.info("Generated Pandas operation: %s", operation)
            return operation
        else:
            try:
                err_body = response.json()
            except ValueError:
                err_body = response.text

            _LOGGER.error(
                "Ollama request failed: status=%s, body=%r",
                response.status_code, err_body
            )
            raise DataAgentError(
                f"Ollama request failed: status={response.status_code}, body={err_body!r}"
            )
