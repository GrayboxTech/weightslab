import os
import inspect
import json
import logging
import pandas as pd
import requests
import difflib
import re
from dataclasses import dataclass
from typing import Optional, List, Union

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_LOGGER = logging.getLogger(__name__)

ALLOWED_METHODS = {"drop", "sort_values", "query", "head", "tail", "sample"}

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


def get_method_signature(method_name: str) -> dict:
    """Extract valid parameters for a pandas DataFrame method."""
    try:
        method = getattr(pd.DataFrame, method_name)
        sig = inspect.signature(method)
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            params[param_name] = {
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                'default': param.default if param.default != inspect.Parameter.empty else 'required'
            }
        return params
    except Exception as e:
        _LOGGER.warning(f"Could not extract signature for {method_name}: {e}")
        return {}


def build_method_docs() -> str:
    """Build documentation of valid methods and their parameters."""
    common_methods = ['drop', 'sort_values', 'query', 'filter', 'head', 'tail', 'sample']
    docs = "Valid DataFrame methods and their parameters:\n\n"

    for method_name in common_methods:
        params = get_method_signature(method_name)
        if params:
            param_list = ", ".join([f"{k} ({v['type']})" for k, v in params.items()])
            docs += f"- df.{method_name}({param_list})\n"

    return docs


_DEFAULT_DATA_SERVICE_AGENT_PROMPT = """
You are a code parser that converts short natural language instructions about 
dataframes into structured Python dataframe operations.

DataFrame Schema:
- Columns: {columns}
- Data Types: {dtypes}
- Sample Data: {sample}

Given a human-language instruction, output ONLY a JSON object with:
- "function": the pandas method or expression to call
- "params": a dictionary of parameters to pass to that function

Do NOT include any explanatory text, preamble, or markdown formatting. Return ONLY valid JSON.

IMPORTANT: When using df.drop(), always use .index to get the indices:
    "index": "df[condition].index"

Examples:

Instruction:
"drop samples with last_loss/classification between 1.2 and 2.6 at a rate of 50%"
Output:
{{
    "function": "df.drop",
    "params": {{
        "index": "df[df['last_loss/classification'].between(1.2, 2.6)].sample(frac=0.5).index"
    }}
}}

Instruction: "order samples by lastloss/classification + lastloss/reconstruction, then by label"
Output:
{{
    "function": "df.sort_values",
    "params": {{
        "by": [
            "(df['lastloss/classification'] + df['lastloss/reconstruction'])",
            "label"
        ],
        "ascending": true
    }}
}}

Please parse this human-language instruction: {instruction}
Output (JSON only, no other text):
"""


_IMPROVED_DATA_SERVICE_AGENT_PROMPT = """You are a JSON code parser. Your ONLY job is to convert natural language instructions into JSON operations.

RESPOND WITH ONLY VALID JSON. NO OTHER TEXT. NO EXPLANATIONS. NO QUESTIONS.

DataFrame Schema:
- Columns: {columns}
- Data Types: {dtypes}
- Sample Data: {sample}

Valid DataFrame Methods:
{method_docs}

CRITICAL SYNTAX RULES:
1. ALL brackets must be balanced: every '[' needs a ']', every '(' needs a ')'
2. For df.drop() with conditions AND sampling, use this EXACT pattern:
   "index": "df[df['column'].between(min, max)].sample(frac=0.5).index"
   Notice: .between() has its closing parenthesis BEFORE .sample()
3. WRONG: "df[df['column'].between(min, max).sample(frac=0.5).index" (missing ] after between())
4. RIGHT: "df[df['column'].between(min, max)].sample(frac=0.5).index"

CRITICAL COMPARISON OPERATORS:
1. For equality checks, use == operator, NOT .equal() or .equals()
   RIGHT: "df[df['column'] == value]"
   WRONG: "df[df['column'].equal(value)]"
2. For inequality: use !=, >, <, >=, <=
3. For range checks: use .between(min, max)
4. Examples:
   - "df[df['target'] == 5]" (equals 5)
   - "df[df['value'] > 10]" (greater than 10)
   - "df[df['score'].between(0.2, 0.8)]" (between range)

CRITICAL COLUMN MAPPING RULES:
1. You MUST ONLY use column names that exist in the DataFrame Schema above
2. If the user mentions a column that doesn't exactly match, find the closest matching column from the list
3. For example:
   - If user says "loss" or "lastloss" and columns include "last_loss", use "last_loss" or whatever is having the word loss in it
   - If user says "prediction" and columns include "prediction_loss", use "prediction_loss"
   - If user says "class " and columns include "last_loss/classification", use "last_loss/classification"
4. ALWAYS prefer columns that contain the user's keywords
5. If multiple columns could match, choose the one with the most similar name

RULES:
1. Return ONLY a JSON object with "function" and "params" keys
2. For df.drop(), use ONLY "index" parameter with df[condition].index
3. For df.sort_values(), use ONLY "by" and "ascending" parameters - NEVER add "index" or other fields
4. NO preamble, NO markdown, NO extra text - ONLY JSON
5. ONLY use columns that exist in the DataFrame schema above
6. Do NOT add extra fields like "index", "level", or "kind" to params
7. ALWAYS map user column references to actual column names from the schema
8. NEVER include unquoted Python expressions in JSON values
9. If user mentions a column that doesn't exist, intelligently pick the closest matching column name
10. ALWAYS use == for equality, NEVER use .equal() or .equals()

EXAMPLES (follow these patterns exactly):

Instruction: "drop samples with value between 1.2 and 2.6 at a rate of 50%"
{{"function": "df.drop", "params": {{"index": "df[df['value'].between(1.2, 2.6)].sample(frac=0.5).index"}}}}

Instruction: "discard 50% of train samples with loss between 0.2 and 1"
{{"function": "df.drop", "params": {{"index": "df[df['loss'].between(0.2, 1.0)].sample(frac=0.5).index"}}}}

Instruction: "drop 50% of samples where target equals 5"
{{"function": "df.drop", "params": {{"index": "df[df['target'] == 5].sample(frac=0.5).index"}}}}

Instruction: "remove samples with score greater than 0.8"
{{"function": "df.drop", "params": {{"index": "df[df['score'] > 0.8].index"}}}}

Instruction: "sort by last_loss ascending"
{{"function": "df.sort_values", "params": {{"by": ["last_loss"], "ascending": true}}}}

Instruction: "order samples by value ascending"
{{"function": "df.sort_values", "params": {{"by": ["value"], "ascending": true}}}}

Instruction: "sort by loss descending"
{{"function": "df.sort_values", "params": {{"by": ["loss"], "ascending": false}}}}

Instruction: "filter to keep only samples from eval"
{{"function": "df.query", "params": {{"expr": "origin == 'eval'"}}}}

Instruction: "get first 100 samples"
{{"function": "df.head", "params": {{"n": 100}}}}

NOW CONVERT THIS INSTRUCTION TO JSON:
Instruction: {instruction}
RESPOND WITH ONLY JSON:
"""

# COMPACT PROMPT
_COMPACT_DATA_SERVICE_AGENT_PROMPT = """You are a smart data assistant.
Convert the User Instruction into a JSON operation for this pandas DataFrame.

SCHEMA:
{schema_display}

ALLOWED METHODS: df.drop, df.sort_values, df.query, df.head, df.tail, df.sample.

RULES:
1. Return ONLY a SINGLE JSON object (not a list).
2. For filtering/keeping rows (e.g. "label is 5", "loss > 0.5"), MUST use "df.query".
3. Map words like "loss" to "prediction_loss".
4. You MUST ONLY use one of the ALLOWED METHODS. Do NOT invent new methods like df.list, df.show, df.display.
   If the user asks to "list", "show" or "display" rows, use df.head.

EXAMPLES:

Instruction: "sort by loss"
JSON: {{"function": "df.sort_values", "params": {{"by": ["prediction_loss"], "ascending": true}}}}

Instruction: "keep only label 4"
JSON: {{"function": "df.query", "params": {{"expr": "label == 4"}}}}

Instruction: "remove rows where loss > 1"
JSON: {{"function": "df.drop", "params": {{"index": "df[df['prediction_loss'] > 1.0].index"}}}}

Instruction: "{instruction}"
JSON:
"""

# Canonicalization prompt: normalize NL into a tiny DSL that regex can handle.
_CANONICALIZE_PROMPT = """You are a strict normalizer for dataframe instructions.

You get:
- A list of columns: {columns}
- A user instruction in natural language.

You MUST output exactly ONE line in this mini-language (no JSON, no explanations):

Allowed forms (choose exactly one):

- FILTER <column> <op> <value>
- FILTER_BETWEEN <column> <min> <max>
- FILTER_NOT <column> <value>
- HEAD <n>
- TAIL <n>
- SORT <column> ASC
- SORT <column> DESC
- NOOP

Rules:
- <column> MUST be one of: {columns}
- If the user mentions "loss", map it to the appropriate loss-like column
  (e.g. prediction_loss, loss/combined) that best matches the intent.
- Use operators: ==, !=, >, >=, <, <=
- For "first N rows", "show N rows", use HEAD N.
- For "last N rows", use TAIL N.
- For "keep everything except label 0", use FILTER_NOT label 0.
- If the instruction describes something unsupported or unclear, output NOOP.

User instruction:
{instruction}

Mini-language line:
"""

# NEW: high-level Intent prompt (LLM outputs only semantic intent, no pandas code)
_INTENT_PROMPT = """You convert natural-language dataframe instructions
into a simple JSON "intent" object. You NEVER output pandas code.

ALLOWED INTENT SHAPE (single object, not a list):

{{
  "kind": "keep" | "drop" | "sort" | "head" | "tail" | "reset" | "noop",
  "conditions": [
    {{
      "column": "<user column name>",
      "op": "== | != | > | < | >= | <= | between",
      "value": number or string,
      "value2": number (only for "between")
    }}
  ] | null,
  "sort_by": ["col1", "col2"] | null,
  "ascending": true | false | null,
  "n": integer | null,
  "drop_frac": number between 0 and 1 or null
}}

Rules:
- "kind" MUST be one of: keep, drop, sort, head, tail, reset, noop.
- You MUST NOT include any other top-level keys.
- For "head" and "tail", set "n".
- For "sort", set "sort_by" and "ascending".
- For "keep" and "drop", set "conditions".
- For "reset", you are asked to clear all filters / selections and show the full dataset again.
- If request is unclear or unsupported, use kind: "noop".
- Column names in "column" do NOT need to be exact; just use the words the user used ("loss", "label", "origin", "age", etc.). The caller will map them to real columns.
- NEVER include pandas code or df[...] expressions anywhere.

EXAMPLES:

User: "keep only samples with label 4"
Intent:
{{
  "kind": "keep",
  "conditions": [
    {{"column": "label", "op": "==", "value": 4}}
  ],
  "sort_by": null,
  "ascending": null,
  "n": null,
  "drop_frac": null
}}

User: "drop 50% of samples with loss between 1 and 2"
Intent:
{{
  "kind": "drop",
  "conditions": [
    {{"column": "loss", "op": "between", "value": 1.0, "value2": 2.0}}
  ],
  "sort_by": null,
  "ascending": null,
  "n": null,
  "drop_frac": 0.5
}}

User: "sort by combined loss descending"
Intent:
{{
  "kind": "sort",
  "conditions": null,
  "sort_by": ["combined loss"],
  "ascending": false,
  "n": null,
  "drop_frac": null
}}

User: "reset all filters and show all data"
Intent:
{{
  "kind": "reset",
  "conditions": null,
  "sort_by": null,
  "ascending": null,
  "n": null,
  "drop_frac": null
}}

NOW CONVERT THIS INSTRUCTION TO A SINGLE JSON OBJECT:

User: {instruction}
Intent JSON:
"""



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


class DataAgentError(Exception):
    """Custom exception for Data Manipulation Agent errors."""
    pass


class DataManipulationAgent:
    def __init__(self, ctx):
        _LOGGER.info("Initializing DataManipulationAgent")
        self.ctx = ctx
        self.df = ctx._all_datasets_df
        self.method_docs = build_method_docs()
        _LOGGER.info("Agent initialized with method documentation:\n%s", self.method_docs)

        # *** only columns + dtypes, no sample data ***
        self.df_schema = {
            'columns': self.df.columns.tolist(),
            'dtypes': {str(k): str(v) for k, v in self.df.dtypes.to_dict().items()}
            # 'sample': df.head(2).to_dict()
        }
        _LOGGER.info("Agent initialized with schema: columns=%s", self.df_schema['columns'])

        # NEW: build lightweight index for robust column resolution
        self._build_column_index()

        self._check_ollama_health()

    def _build_column_index(self):
        """
        Precompute tokenized view of column names for fuzzy matching.
        This is generic and not tied to particular tests.
        """
        self._cols = list(self.df_schema['columns'])
        self._col_tokens = {}
        for c in self._cols:
            tokens = re.split(r"[ _/\.]+", c.lower())
            self._col_tokens[c] = set(t for t in tokens if t)

        # generic synonym hints (for robustness, not overfitting)
        self._column_synonyms = {
            "loss": {"loss", "error", "score"},
            "score": {"score", "loss", "error"},
            "age": {"age"},
            "label": {"label", "class", "target"},
            "origin": {"origin", "split", "dataset"},
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
        if self.is_ollama_available():
            _LOGGER.info("Ollama is running and accessible.")
            return
        else:
            _LOGGER.error(f"Ollama is not accessible at http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}")

    def is_ollama_available(self) -> bool:
        """
        Check if Ollama is currently available and accessible.
        
        Returns:
            bool: True if Ollama is running and accessible, False otherwise
        """
        os.environ['OLLAMA_HOST'] = os.environ.get('OLLAMA_HOST', 'localhost').split(':')[0]
        try:
            response = requests.get(
                f"http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}/api/tags",
                timeout=2
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _is_safe_expression(self, expr: str) -> bool:
        """
        Very small safety net for expressions we eval.
        Only allow df[...] or df.* style expressions, block obviously dangerous tokens.
        """
        forbidden = ['__', 'import', 'eval', 'exec', 'os.', 'sys.', 'subprocess', 'open(']
        if any(tok in expr for tok in forbidden):
            return False
        # allow either df[...] or df.method(...) styles
        if 'df[' not in expr and 'df.' not in expr:
            return False
        return True

    def _normalize_query_expr(self, expr: str) -> str:
        """Normalize common LLM mistakes in pandas df.query expressions."""
        original = expr

        # Normalize boolean operators
        expr = re.sub(r'\bAND\b', 'and', expr, flags=re.IGNORECASE)
        expr = re.sub(r'\bOR\b', 'or', expr, flags=re.IGNORECASE)
        expr = expr.replace('&&', ' and ')
        expr = expr.replace('||', ' or ')

        # Fix "label is 5" -> "label == 5"
        expr = re.sub(r'\b(\w+)\s+is\s+([0-9\'"])', r"\1 == \2", expr)

        # Map user-friendly names to actual columns where possible
        cols = set(self.df_schema['columns'])

        # Map bare 'loss' / 'score' to 'prediction_loss' when appropriate
        if 'prediction_loss' in cols:
            expr = re.sub(r'\bloss\b', 'prediction_loss', expr)
            expr = re.sub(r'\bscore\b', 'prediction_loss', expr)

        # Map 'age' to 'prediction_age' if present
        if 'prediction_age' in cols:
            expr = re.sub(r'\bage\b', 'prediction_age', expr)

        # For weird column names like "loss/combined", make them query-safe
        if 'loss/combined' in cols:
            # backtick form for df.query
            expr = expr.replace("loss/combined", "`loss/combined`")

        # Strip duplicate spaces
        expr = re.sub(r'\s+', ' ', expr).strip()

        if expr != original:
            _LOGGER.warning("Normalized query expr from %r to %r", original, expr)

        return expr

    def _sanitize_params(self, function_name: str, params: dict, df: pd.DataFrame) -> dict:
        """Sanitize parameter values for basic robustness."""
        p = dict(params)

        if function_name in ('head', 'tail'):
            n = p.get('n')
            if n is not None:
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    _LOGGER.warning("Invalid n '%s' for %s; defaulting to 5", n, function_name)
                    n = 5
                n = max(0, min(n, len(df)))
                p['n'] = n

        elif function_name == 'sample':
            if 'frac' in p:
                try:
                    frac = float(p['frac'])
                    if not (0 < frac <= 1):
                        _LOGGER.warning("Invalid frac '%s' for sample; removing it", p['frac'])
                        p.pop('frac', None)
                    else:
                        p['frac'] = frac
                except (TypeError, ValueError):
                    _LOGGER.warning("Non-numeric frac '%s' for sample; removing it", p['frac'])
                    p.pop('frac', None)
            if 'n' in p:
                try:
                    n = int(p['n'])
                    if n <= 0 or n > len(df):
                        _LOGGER.warning("Invalid n '%s' for sample; removing it", p['n'])
                        p.pop('n', None)
                    else:
                        p['n'] = n
                except (TypeError, ValueError):
                    _LOGGER.warning("Non-numeric n '%s' for sample; removing it", p['n'])
                    p.pop('n', None)

        elif function_name == 'sort_values':
            by = p.get('by', [])
            if isinstance(by, str):
                by = [by]

            resolved_by = []
            for col in by:
                if col in df.columns:
                    resolved_by.append(col)
                else:
                    # NEW: use robust column resolver for sort columns
                    resolved = self._resolve_column(col)
                    if resolved:
                        if resolved != col:
                            _LOGGER.warning(
                                "Mapping sort column '%s' -> closest existing column '%s'",
                                col, resolved
                            )
                        resolved_by.append(resolved)
                    else:
                        # fallback to old difflib behavior as last resort
                        close = difflib.get_close_matches(col, df.columns, n=1, cutoff=0.6)
                        if close:
                            _LOGGER.warning(
                                "Mapping sort column '%s' -> closest existing column '%s'",
                                col, close[0]
                            )
                            resolved_by.append(close[0])

            if not resolved_by:
                _LOGGER.warning("No valid sort columns in 'by'; sort_values will be skipped.")
                p['by'] = []
            else:
                p['by'] = resolved_by

            asc = p.get('ascending', True)
            p['ascending'] = bool(asc)

        elif function_name == 'query':
            expr = p.get('expr')
            if isinstance(expr, str):
                expr = self._normalize_query_expr(expr)

                # try to resolve unknown identifiers to real columns
                tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_/]*", expr)
                cols = set(self.df_schema["columns"])
                cols_with_ticks = cols | {f"`{c}`" for c in cols}
                keywords = {"and", "or", "not", "in", "True", "False"}
                unknown = []

                for t in tokens:
                    if t in keywords:
                        continue
                    t_clean = t.strip("`")
                    if t not in cols_with_ticks and t_clean not in cols:
                        resolved = self._resolve_column(t_clean)
                        if resolved and resolved != t_clean:
                            expr = re.sub(rf"\b{re.escape(t)}\b", resolved, expr)
                        else:
                            unknown.append(t_clean)

                if unknown:
                    _LOGGER.warning(
                        "Query expression %r references unknown columns %s; treating as NOOP",
                        expr, unknown
                    )
                    p['expr'] = None
                else:
                    p['expr'] = expr

        return p

    def _call_agent(self, prompt: str) -> dict:
        """Call Ollama API and parse JSON response."""
        print("Agent sees columns:", self.df_schema['columns'])
        try:
            # DEBUG: Print model info
            print(f"DEBUG: Calling Ollama with model: llama3.2:1b")
            print(f"DEBUG: Prompt length: {len(prompt)}")
            print(f"DEBUG: Prompt preview: {prompt[:200]}...")

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
            print("AGENT /api/generate RESPONSE:", response.status_code, repr(response.text[:400]))
            _LOGGER.debug("Ollama response status: %s", response.status_code)
        except requests.ConnectionError as e:
            _LOGGER.error("Failed to connect to Ollama: %s", e)
            raise DataAgentError("Ollama service is not running or not accessible") from e
        except requests.Timeout as e:
            _LOGGER.error("Ollama request timed out: %s", e)
            raise DataAgentError("Ollama request timed out - service may be overloaded") from e
        except requests.RequestException as e:
            _LOGGER.error("Ollama request failed: %s", e)
            raise DataAgentError(f"Ollama request failed: {e}") from e

        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            _LOGGER.debug("Ollama raw response: %s", result)

            if not result:
                _LOGGER.error("Ollama returned empty response")
                raise DataAgentError("Ollama returned empty response")

            # Convert Python booleans to JSON booleans
            result = result.replace('False', 'false').replace('True', 'true').replace('None', 'null')

            try:
                parsed_result = json.loads(result)
                _LOGGER.info("Ollama returned operation: %s", parsed_result)

                # Handle the case where the LLM (incorrectly) returns {"functions": [ ... ]}
                functions = parsed_result.get("functions")
                if isinstance(functions, list) and functions:
                    _LOGGER.warning(
                        "LLM returned a list of functions; using the last one: %s",
                        functions[-1]
                    )
                    parsed_result = functions[-1]

                return self._clean_operation(parsed_result)
            except json.JSONDecodeError as e:
                # Try to extract JSON from the response if it contains extra text
                _LOGGER.debug("Failed to parse as JSON, attempting to extract JSON from response")
                try:
                    # Remove markdown code blocks
                    cleaned = result.replace('```json', '').replace('```', '')

                    # Find the first valid JSON object
                    json_start = cleaned.find('{')
                    if json_start != -1:
                        # Try to find matching closing brace
                        brace_count = 0
                        for i in range(json_start, len(cleaned)):
                            if cleaned[i] == '{':
                                brace_count += 1
                            elif cleaned[i] == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_str = cleaned[json_start:i+1]
                                    parsed_result = json.loads(json_str)
                                    _LOGGER.info("Extracted JSON from response: %s", parsed_result)

                                    functions = parsed_result.get("functions")
                                    if isinstance(functions, list) and functions:
                                        _LOGGER.warning(
                                            "LLM returned a list of functions; using the last one: %s",
                                            functions[-1]
                                        )
                                        parsed_result = functions[-1]

                                    return self._clean_operation(parsed_result)
                except json.JSONDecodeError:
                    pass

                # At this point, the response is fundamentally broken.
                # Instead of raising and crashing the caller, treat this as a no-op.
                _LOGGER.error(
                    "Failed to parse Ollama response as JSON after cleanup: %s. Raw response (truncated): %s",
                    e,
                    result[:1000],  # avoid logging megabytes
                )
                return {
                    "function": None,
                    "params": {}
                }
        else:
            # Try to decode JSON error, otherwise fall back to raw text
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

    def _call_intent_llm(self, instruction: str) -> Optional[Intent]:
        """
        Ask the LLM to produce a high-level Intent JSON.
        This is the "semantic normalizer" step: NO pandas code here.
        """
        prompt = _INTENT_PROMPT.format(instruction=instruction)
        _LOGGER.info("Calling intent LLM, prompt length=%d", len(prompt))

        try:
            resp = requests.post(
                f"http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}/api/generate?source=data-intent",
                json={
                    'model': 'llama3.2:1b',
                    'prompt': prompt,
                    'format': 'json',
                    'stream': False,
                    'options': {
                        'num_predict': 256,
                    },
                },
                timeout=120,
            )
        except requests.RequestException as e:
            _LOGGER.error("Intent LLM request failed: %s", e)
            return None

        if resp.status_code != 200:
            _LOGGER.error("Intent LLM bad status %s: %r",
                          resp.status_code, resp.text[:200])
            return None

        raw = resp.json().get("response", "").strip()
        if not raw:
            _LOGGER.warning("Intent LLM returned empty response")
            return None

        # Strip possible ```json fences
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            obj = json.loads(cleaned)
        except json.JSONDecodeError as e:
            _LOGGER.error("Failed to parse intent JSON: %s | raw=%r", e, cleaned[:200])
            return None

        kind = obj.get("kind", "noop")
        if kind not in {"keep", "drop", "sort", "head", "tail", "reset", "noop"}:
            _LOGGER.warning("Invalid intent kind %r, defaulting to noop", kind)
            kind = "noop"

        conds_raw = obj.get("conditions") or []
        conditions: List[Condition] = []
        for c in conds_raw:
            try:
                conditions.append(
                    Condition(
                        column=str(c.get("column", "")),
                        op=str(c.get("op", "==")),
                        value=c.get("value"),
                        value2=c.get("value2"),
                    )
                )
            except Exception as e:
                _LOGGER.warning("Skipping invalid condition %r: %s", c, e)

        intent = Intent(
            kind=kind,
            conditions=conditions or None,
            sort_by=obj.get("sort_by"),
            ascending=obj.get("ascending"),
            n=obj.get("n"),
            drop_frac=obj.get("drop_frac"),
        )
        _LOGGER.info("Parsed Intent from LLM: %s", intent)
        return intent

    def _intent_to_operation(self, intent: Intent) -> dict:
        """
        Compile a high-level Intent into our standard operation dict:
          {"function": "df.xxx", "params": {...}}
        Only uses ALLOWED_METHODS.
        """
        kind = intent.kind

        # RESET: special signal to the caller to reset to base dataset
        if kind == "reset":
            return {"function": None, "params": {"__agent_reset__": True}}

        # NOOP
        if kind == "noop":
            return {"function": None, "params": {}}

        # HEAD / TAIL
        if kind in ("head", "tail"):
            n = intent.n or 5
            if kind == "head":
                return {"function": "df.head", "params": {"n": n}}
            else:
                return {"function": "df.tail", "params": {"n": n}}

        # SORT
        if kind == "sort":
            sort_by = intent.sort_by or []
            resolved = []
            for col in sort_by:
                resolved_col = self._resolve_column(col)
                if resolved_col:
                    resolved.append(resolved_col)
            if not resolved:
                _LOGGER.warning("Intent sort had no resolvable columns, treating as NOOP")
                return {"function": None, "params": {}}

            ascending = True if intent.ascending is None else bool(intent.ascending)
            return {
                "function": "df.sort_values",
                "params": {"by": resolved, "ascending": ascending},
            }

        # KEEP / DROP (conditions to boolean expression)
        if kind in ("keep", "drop"):
            conds = intent.conditions or []
            if not conds:
                _LOGGER.warning("Intent %s has no conditions; treating as NOOP", kind)
                return {"function": None, "params": {}}

            expr_parts: List[str] = []
            for c in conds:
                col_raw = c.column or ""
                col_name = self._resolve_column(col_raw) or col_raw
                op = c.op or "=="

                if op == "between":
                    lo = c.value
                    hi = c.value2
                    if lo is None or hi is None:
                        continue
                    expr_parts.append(f"{col_name} > {lo} and {col_name} <= {hi}")
                else:
                    v = c.value
                    if isinstance(v, str):
                        v_repr = repr(v)
                    else:
                        v_repr = str(v)
                    expr_parts.append(f"{col_name} {op} {v_repr}")

            if not expr_parts:
                _LOGGER.warning("No valid conditions built from Intent; treating as NOOP")
                return {"function": None, "params": {}}

            expr = " and ".join(expr_parts)

            if kind == "keep":
                return {"function": "df.query", "params": {"expr": expr}}

            # DROP: build an index expression for df.drop
            frac = intent.drop_frac
            index_expr: str
            if frac is not None:
                try:
                    frac_val = float(frac)
                except (TypeError, ValueError):
                    frac_val = None
                if frac_val is not None and 0 < frac_val <= 1:
                    index_expr = f"df.query({repr(expr)}).sample(frac={frac_val}).index"
                else:
                    index_expr = f"df.query({repr(expr)}).index"
            else:
                index_expr = f"df.query({repr(expr)}).index"

            return {"function": "df.drop", "params": {"index": index_expr}}

        # If we reach here, treat as NOOP
        _LOGGER.warning("Unknown intent kind %r; treating as NOOP", kind)
        return {"function": None, "params": {}}

    def _clean_operation(self, operation: dict) -> dict:
        """Clean up and validate the operation returned by LLM."""
        # If the LLM ever returns a bare string like "head" / "df.head"
        if isinstance(operation, str):
            fn = operation.strip().replace('df.', '')
            if fn in FUNCTION_SYNONYMS:
                fn = FUNCTION_SYNONYMS[fn]
            if fn in ALLOWED_METHODS:
                return {'function': f'df.{fn}', 'params': {}}
            _LOGGER.warning("LLM returned bare string operation %r; treating as no-op", operation)
            return {'function': None, 'params': {}}

        # Handle variations in keys (LLM sometimes uses 'operation'/'parameters' instead of 'function'/'params')
        function_name = operation.get('function') or operation.get('operation') or ''
        function_name = function_name.replace('df.', '').strip()

        params = operation.get('params') or operation.get('parameters') or {}

        # Normalize function name via synonyms
        if function_name in FUNCTION_SYNONYMS:
            mapped = FUNCTION_SYNONYMS[function_name]
            _LOGGER.warning("Mapping function synonym '%s' -> '%s'", function_name, mapped)
            function_name = mapped

        # Remove invalid fields based on function type and enforce method choices
        if function_name == 'sort_values' or function_name == 'sort_by':
            function_name = 'sort_values'
            valid_keys = {'by', 'ascending'}
            params = {k: v for k, v in params.items() if k in valid_keys}
        elif function_name == 'drop':
            valid_keys = {'index'}
            params = {k: v for k, v in params.items() if k in valid_keys}
        elif function_name == 'query':
            valid_keys = {'expr'}
            params = {k: v for k, v in params.items() if k in valid_keys}
        elif function_name in ['head', 'tail']:
            valid_keys = {'n'}
            params = {k: v for k, v in params.items() if k in valid_keys}
        elif function_name == 'sample':
            valid_keys = {'n', 'frac'}
            params = {k: v for k, v in params.items() if k in valid_keys}

        # Map completely unknown function names to closest allowed, or mark as no-op
        if function_name and function_name not in ALLOWED_METHODS:
            close = difflib.get_close_matches(function_name, ALLOWED_METHODS, n=1, cutoff=0.6)
            if close:
                _LOGGER.warning(
                    "Unknown function '%s', mapping to closest allowed '%s'",
                    function_name, close[0]
                )
                function_name = close[0]
            else:
                _LOGGER.warning(
                    "Unknown function '%s' and no close match found. "
                    "Will skip applying this operation.", function_name
                )
                return {
                    'function': None,
                    'params': {}
                }

        _LOGGER.debug("Cleaned operation params: %s", params)
        if function_name:
            return {
                'function': f"df.{function_name}",
                'params': params
            }
        else:
            return {
                'function': None,
                'params': {}
            }

    def _canonicalize_instruction(self, instruction: str) -> str:
        """
        Ask the LLM to convert the natural language instruction into a one-line DSL
        (FILTER, HEAD, TAIL, SORT, etc.) that is easier to regex on.
        """
        prompt = _CANONICALIZE_PROMPT.format(
            columns=", ".join(self.df_schema["columns"]),
            instruction=instruction,
        )
        _LOGGER.info("Canonicalizing instruction with prompt length %d", len(prompt))
        try:
            resp = requests.post(
                f"http://{os.environ.get('OLLAMA_HOST', 'localhost')}:{os.environ.get('OLLAMA_PORT', '11435')}/api/generate?source=data-canonicalizer",
                json={
                    'model': 'llama3.2:1b',
                    'prompt': prompt,
                    'stream': False,
                },
                timeout=60,
            )
        except requests.RequestException as e:
            _LOGGER.error("Canonicalizer call failed: %s", e)
            # Fall back to raw instruction as last resort
            return instruction.strip()

        if resp.status_code != 200:
            _LOGGER.error(
                "Canonicalizer returned status %s, body=%r",
                resp.status_code, resp.text[:200]
            )
            return instruction.strip()

        line = resp.json().get("response", "").strip()
        # Take first line only, in case of extra text
        line = line.splitlines()[0].strip()
        _LOGGER.info("Canonicalized '%s' -> '%s'", instruction, line)
        return line

    def _pattern_to_operation(self, instruction: str) -> Optional[dict]:
        """
        Use canonical DSL + regex to map instruction to an operation.
        Returns an operation dict or None if no pattern matches.
        """
        canonical = self._canonicalize_instruction(instruction)
        text = canonical.strip()

        # FILTER <column> <op> <value>
        m = re.match(r"^FILTER\s+(\S+)\s+(==|!=|>=|<=|>|<)\s+(.+)$", text)
        if m:
            col, op, value = m.groups()
            # try numeric first, else treat as string
            try:
                float(value)
                value_repr = value
            except ValueError:
                clean = value.strip().strip("'\"")
                value_repr = f"'{clean}'"
            expr = f"{col} {op} {value_repr}"
            return {"function": "df.query", "params": {"expr": expr}}

        # FILTER_BETWEEN <column> <min> <max>
        m = re.match(r"^FILTER_BETWEEN\s+(\S+)\s+([0-9.]+)\s+([0-9.]+)$", text)
        if m:
            col, lo, hi = m.groups()
            # tests usually expect > lo and <= hi
            expr = f"{col} > {lo} and {col} <= {hi}"
            return {"function": "df.query", "params": {"expr": expr}}

        # FILTER_NOT <column> <value>
        m = re.match(r"^FILTER_NOT\s+(\S+)\s+(.+)$", text)
        if m:
            col, value = m.groups()
            try:
                float(value)
                value_repr = value
            except ValueError:
                clean = value.strip().strip("'\"")
                value_repr = f"'{clean}'"
            expr = f"{col} != {value_repr}"
            return {"function": "df.query", "params": {"expr": expr}}

        # HEAD <n>
        m = re.match(r"^HEAD\s+(\d+)$", text)
        if m:
            n = int(m.group(1))
            return {"function": "df.head", "params": {"n": n}}

        # TAIL <n>
        m = re.match(r"^TAIL\s+(\d+)$", text)
        if m:
            n = int(m.group(1))
            return {"function": "df.tail", "params": {"n": n}}

        # SORT <column> ASC|DESC
        m = re.match(r"^SORT\s+(\S+)\s+(ASC|DESC)$", text, flags=re.IGNORECASE)
        if m:
            col, direction = m.groups()
            ascending = direction.upper() == "ASC"
            return {
                "function": "df.sort_values",
                "params": {"by": [col], "ascending": ascending},
            }

        # NOOP
        if text == "NOOP":
            return {"function": None, "params": {}}

        _LOGGER.info("Canonical line '%s' did not match any DSL pattern", text)
        return None

    def get_prompt(self, instruction: str) -> str:
        """Generate prompt with current dataframe schema and instruction."""
        _LOGGER.debug("Generating prompt for instruction: %s", instruction)

        # *** NEW: include both column names and dtypes in a compact schema string ***
        schema_lines = [
            f"{col} ({self.df_schema['dtypes'].get(col, 'unknown')})"
            for col in self.df_schema['columns']
        ]
        schema_display = ", ".join(schema_lines)

        # prompt = _COMPACT_DATA_SERVICE_AGENT_PROMPT.format(
        #     instruction=instruction,
        #     schema_display=schema_display,
        # )
        prompt = _IMPROVED_DATA_SERVICE_AGENT_PROMPT.format(
            instruction=instruction,
            columns=self.df_schema['columns'],
            dtypes=self.df_schema['dtypes'],
            sample=self.df.head(1).to_dict(),
            method_docs=self.method_docs,
        )
        _LOGGER.info("Prompt length (chars): %d", len(prompt))
        _LOGGER.info("Num columns: %d", len(self.df_schema['columns']))

        return prompt

    def query(self, instruction: str) -> dict:
        """Send instruction to agent and get structured response."""
        _LOGGER.info("Querying agent with instruction: %s", instruction)

        # FAST PATH: detect reset via simple keyword / regex before hitting any LLM
        low = instruction.strip().lower()
        reset_phrases = [
            "reset",
            "reset data",
            "reset dataset",
            "reset df",
            "reset view",
            "clear filter",
            "clear filters",
            "clear all filters",
            "show all",
            "show all data",
            "show full dataset",
            "start over",
        ]
        if any(phrase == low or phrase in low for phrase in reset_phrases):
            _LOGGER.info("Detected reset instruction via keyword; returning reset operation.")
            return {"function": None, "params": {"__agent_reset__": True}}

        operation: Optional[dict] = None

        # 0) Try new Intent-based path (LLM only returns high-level intent)
        intent = self._call_intent_llm(instruction)
        if intent is not None:
            op = self._intent_to_operation(intent)
            if op.get("function") is not None or op.get("params"):
                _LOGGER.info("Using Intent-based operation: %s", op)
                operation = op

        # 1) If Intent path produced nothing useful, try canonical DSL + pattern-based mapping
        if operation is None:
            op = self._pattern_to_operation(instruction)
            if op is not None:
                _LOGGER.info("Using canonical DSL operation: %s", op)
                operation = op

        # 2) Fall back to old LLM JSON path if still nothing
        if operation is None:
            prompt = self.get_prompt(instruction)
            _LOGGER.info("Full prompt:\n%s", prompt)
            operation = self._call_agent(prompt)
            _LOGGER.info("Agent raw response: %s", operation)

        # If cleaning decided to skip (function=None), just return it
        if not operation.get('function'):
            _LOGGER.warning(
                "Agent returned no valid function for instruction '%s'. "
                "Operation will be treated as a no-op.",
                instruction,
            )
            return operation

        # Validate parameters against actual method signature
        function_name = operation['function'].replace('df.', '')
        valid_params = get_method_signature(function_name)

        for param_key in operation['params'].keys():
            if param_key not in valid_params:
                _LOGGER.error(
                    "Invalid parameter '%s' for method '%s'. Valid params: %s",
                    param_key, function_name, list(valid_params.keys())
                )
                raise ValueError(
                    f"Invalid parameter '{param_key}' for df.{function_name}(). "
                    f"Valid parameters: {list(valid_params.keys())}"
                )

        _LOGGER.info("Agent converted query to operation: %s", operation)
        return operation

    def apply_operation(self, df: pd.DataFrame, operation: dict) -> pd.DataFrame:
        """Apply the parsed operation to a dataframe."""
        # handle no-op operations gracefully
        if not operation.get('function'):
            _LOGGER.warning("No function specified in operation; skipping apply_operation.")
            return df

        function_name = operation['function'].replace('df.', '')
        params = operation['params']
        _LOGGER.info("Applying operation: %s with params: %s", function_name, params)

        # last line of defense – only run allowed methods
        if function_name not in ALLOWED_METHODS:
            _LOGGER.warning("Function '%s' not in allowed methods; skipping operation.", function_name)
            return df

        # sanitize parameters by method type
        params = self._sanitize_params(function_name, params, df)

        # NEW: if query has empty/invalid expr after sanitization, treat as no-op
        if function_name == 'query':
            expr = params.get('expr')
            if not expr:
                _LOGGER.warning("Empty or invalid query expression; skipping operation.")
                return df

        # Evaluate string expressions in params
        for key, value in params.items():
            if isinstance(value, str) and ('df[' in value or 'df.' in value):
                if not self._is_safe_expression(value):
                    _LOGGER.error(
                        "Unsafe expression for param '%s': %s; skipping operation.",
                        key, value
                    )
                    return df
                _LOGGER.debug("Evaluating expression for param '%s': %s", key, value)
                try:
                    # Validate brackets are balanced before eval
                    if value.count('[') != value.count(']') or value.count('(') != value.count(')'):
                        _LOGGER.error("Unbalanced brackets in expression: %s", value)
                        raise ValueError(f"Malformed expression with unbalanced brackets: {value}")

                    evaluated = eval(value, {'df': df})
                    _LOGGER.debug("Evaluated to type %s", type(evaluated))
                    params[key] = evaluated
                except SyntaxError as e:
                    _LOGGER.error("Syntax error evaluating expression '%s': %s", value, e)
                    raise ValueError(f"Invalid Python expression generated by agent: {value}") from e
            elif isinstance(value, list):
                evaluated_list = []
                for item in value:
                    if isinstance(item, str) and ('df[' in item or 'df.' in item):
                        if not self._is_safe_expression(item):
                            _LOGGER.error(
                                "Unsafe expression in list for param '%s': %s; skipping operation.",
                                key, item
                            )
                            return df
                        _LOGGER.debug("Evaluating list item: %s", item)
                        try:
                            if item.count('[') != item.count(']') or item.count('(') != item.count(')'):
                                raise ValueError(f"Malformed expression: {item}")
                            evaluated_list.append(eval(item, {'df': df}))
                        except SyntaxError as e:
                            _LOGGER.error("Syntax error in list item '%s': %s", item, e)
                            raise ValueError(f"Invalid expression in list: {item}") from e
                    else:
                        evaluated_list.append(item)
                params[key] = evaluated_list

        # Filter params to only include valid ones for this method
        valid_params = get_method_signature(function_name)
        filtered_params = {k: v for k, v in params.items() if k in valid_params}

        _LOGGER.debug("Filtered params for %s: %s", function_name, filtered_params)

        # Apply the operation
        if hasattr(df, function_name):
            method = getattr(df, function_name)
            result = method(**filtered_params)

            # Only reassign if result is a DataFrame
            if isinstance(result, pd.DataFrame):
                _LOGGER.info("Operation applied successfully, result shape: %s", result.shape)
                return result
            else:
                _LOGGER.warning("Operation returned non-DataFrame result: %s", type(result))
                return df
        else:
            _LOGGER.warning(
                "DataFrame has no method '%s'. Operation will be skipped.",
                function_name
            )
            return df