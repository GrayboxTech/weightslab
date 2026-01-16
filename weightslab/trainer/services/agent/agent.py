import os
import logging
import difflib
import re
import pandas as pd
from typing import Optional, List, Union, Literal
from dotenv import load_dotenv
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_LOGGER = logging.getLogger(__name__)

from .intent_prompt import INTENT_PROMPT

# Try to find .env in weightslab/ or parent root
env_path = Path(__file__).resolve().parents[3] / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# --- Pydantic Models ---
class Condition(BaseModel):
    column: str = Field(description="The column name to filter/check")
    op: Literal["==", "!=", ">", "<", ">=", "<=", "between", "contains", "in"] = Field(description="The operator")
    value: Optional[Union[float, int, str]] = Field(default=None, description="The primary value")
    value2: Optional[Union[float, int]] = Field(default=None, description="The secondary value for 'between'")

class AtomicIntent(BaseModel):
    kind: Literal["keep", "drop", "sort", "head", "tail", "reset", "analysis", "noop"] = Field(description="The kind of operation")
    conditions: Optional[List[Condition]] = Field(default=None, description="Conditions for keep/drop")
    sort_by: Optional[List[str]] = Field(default=None, description="Columns to sort by")
    ascending: Optional[bool] = Field(default=None, description="Sort ascending (true) or descending (false)")
    n: Optional[int] = Field(default=None, description="Number of rows for head/tail")
    drop_frac: Optional[float] = Field(default=None, description="Fraction of rows to drop (0.0 to 1.0)")
    keep_frac: Optional[float] = Field(default=None, description="Fraction of rows to keep (0.0 to 1.0)")
    analysis_expression: Optional[str] = Field(default=None, description="Pandas expression string for analysis queries")

class Intent(BaseModel):
    reasoning: str = Field(description="Step-by-step logic explaining the plan.")
    primary_goal: Literal["ui_manipulation", "data_analysis"] = Field(
        description="Whether the user wants to change the grid view (ui_manipulation) or get an answer/calculation (data_analysis)."
    )
    steps: List[AtomicIntent] = Field(description="A sequence of atomic operations to execute in order.")



class DataManipulationAgent:
    def __init__(self, context):
        """Initializes the agent with context and builds the column schema/index."""
        _LOGGER.info("Initializing DataManipulationAgent (Gemini 2.0 Ready)")
        self.ctx = context

        self._setup_schema()
        self._build_column_index()
        self._load_config()
        self._setup_providers()

    def _setup_schema(self):
        """Builds the column schema from the context dataframe."""
        df = self.ctx._all_datasets_df
        all_columns = df.columns.tolist()

        if isinstance(df.index, pd.MultiIndex):
            all_columns.extend([name for name in df.index.names if name is not None])
        elif df.index.name is not None:
            all_columns.append(df.index.name)

        origin_values = []
        if 'origin' in df.columns:
            origin_values = df['origin'].unique().tolist()
        elif 'origin' in all_columns:
            try:
                origin_values = df.index.get_level_values('origin').unique().tolist()
            except: pass

        expected_stats = [
            "mean_loss", "max_loss", "min_loss", "std_loss", "median_loss",
            "num_classes_present", "dominant_class", "dominant_class_ratio",
            "background_ratio", "predicted_class"
        ]
        expected_stats.extend([f"loss_class_{i}" for i in range(10)])

        for col in expected_stats:
            if col not in all_columns:
                all_columns.append(col)

        self.df_schema = {
            'columns': all_columns,
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'origin_values': origin_values
        }

    def _load_config(self):
        """Loads agent configuration from defaults, environment variables, and config files."""
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

        config_paths = [
            repo_root / "agent_config.yaml",
            inner_pkg / "agent_config.yaml",
            Path.cwd() / "agent_config.yaml"
        ]
        
        for path in config_paths:
            if not path.exists():
                continue
                
            try:
                import yaml
                with open(path, 'r') as f:
                    cfg = yaml.safe_load(f)
                    
                if not cfg or "agent" not in cfg:
                    continue
                    
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

    def _get_hyperparams(self) -> Optional[dict]:
        """Robustly extracts hyperparameters from context."""
        if not hasattr(self.ctx, "_ctx"):
            return None

        hp_comp = self.ctx._ctx.components.get("hyperparams")
        if not hp_comp:
            return None

        if isinstance(hp_comp, dict):
            return hp_comp

        if hasattr(hp_comp, "_data"):
            return hp_comp._data

        if hasattr(hp_comp, "get"):
            try: return hp_comp.get()
            except: return None

        return None

    def _setup_providers(self):
        """Initializes direct clients and LangChain chains for LLM providers."""
        self.chain_openai = None
        self.client_google = None
        self.chain_ollama = None
        self.chain_openrouter = None

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                llm = ChatOpenAI(model=self.openai_model, temperature=0, api_key=api_key)
                self.chain_openai = llm.with_structured_output(Intent)
                _LOGGER.info(f"OpenAI active ({self.openai_model})")
            except Exception as e: _LOGGER.error(f"OpenAI error: {e}")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                aliases = {
                    "gemini-1.5-flash-latest": "gemini-1.5-flash",
                    "gemini-1.5-pro-latest": "gemini-1.5-pro"
                }
                self.google_model = aliases.get(self.google_model, self.google_model)
                _LOGGER.info(f"Google active ({self.google_model})")
            except Exception as e: _LOGGER.error(f"Google error: {e}")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            try:
                llm = ChatOpenAI(
                    model=self.openrouter_model,
                    temperature=0,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    streaming=False,
                    max_retries=1,
                    request_timeout=15.0,
                )
                self.chain_openrouter = llm
                _LOGGER.info(f"OpenRouter active with manual JSON parsing ({self.openrouter_model})")
            except Exception as e: 
                _LOGGER.error(f"OpenRouter error: {e}")

        try:
            host = self.ollama_host.split(':')[0]
            port = self.ollama_port
            model_ollama = self.ollama_model

            llm = ChatOllama(base_url=f"http://{host}:{port}", model=model_ollama, temperature=0, timeout=15)
            self.chain_ollama = llm
            _LOGGER.info(f"Ollama active with manual JSON parsing ({model_ollama})")
        except Exception as e: _LOGGER.error(f"Ollama error: {e}")

    def is_ollama_available(self) -> bool:
        return self.chain_ollama is not None

    def query(self, instruction: str) -> dict:
        """Main entry point for natural language instructions."""
        _LOGGER.info(f"[Agent] Query started: '{instruction}'")
        self._setup_schema()

        cols_desc = ", ".join(self.df_schema['columns'])
        if self.df_schema['origin_values']:
            cols_desc += f" (Note: 'origin' has values: {self.df_schema['origin_values']})"

        system_prompt = INTENT_PROMPT.format(
            columns=cols_desc,
            instruction="{instruction}"
        )
        
        _LOGGER.info(f"[Agent] System prompt size: {len(system_prompt)} chars, Columns: {len(self.df_schema['columns'])}")

        order = [self.preferred_provider]
        if self.fallback_to_local and self.preferred_provider != "ollama":
            order.append("ollama")

        _LOGGER.info(f"[Agent] Provider order: {order}")

        for provider in order:
            try:
                _LOGGER.info(f"[Agent] Trying provider: {provider}")
                result = self._try_query_provider(provider, instruction, system_prompt)
                
                if not result:
                    _LOGGER.warning(f"[Agent] ✗ Provider {provider} returned None (likely parsing failure)")
                    continue
                    
                _LOGGER.info(f"[Agent] ✓ Provider {provider} succeeded with {len(result)} operations")
                return result
            except Exception as e:
                _LOGGER.warning(f"[Agent] ✗ Provider {provider} failed critically: {e}")
                continue

        _LOGGER.error(f"[Agent] All providers failed for query: '{instruction}' - returning empty list")
        return []

    def _wrap_analysis_response(self, provider: str, original_question: str, raw_answer: str) -> str:
        """Uses the LLM to wrap the raw analysis result in a conversational sentence."""
        try:
            return f"Analysis Result: {raw_answer}"
        except Exception as e:
            return str(raw_answer)

    def _try_query_provider(self, provider: str, instruction: str, system_prompt: str) -> Optional[List[dict]]:
        """Dispatches query to the specific LLM implementation."""
        if provider == "google" and self.client_google:
            return self._query_google(instruction, system_prompt)

        chain = getattr(self, f"chain_{provider}", None)
        if chain:
            return self._query_langchain(provider, chain, instruction, system_prompt)

        return None

    def _query_google(self, instruction: str, system_prompt: str) -> Optional[dict]:
        """Executes query using the Google Generative AI SDK."""
        try:
            _LOGGER.info(f"Calling Google Direct ({self.google_model})")
            response = self.client_google.models.generate_content(
                model=self.google_model,
                contents=f"{system_prompt}\n\nUSER REQUEST: {instruction}",
                config={'response_mime_type': 'application/json', 'response_schema': Intent, 'temperature': 0}
            )
            _LOGGER.info(f"Google returned: {response.parsed}")
            return self._intent_to_pandas_op(response.parsed)
        except Exception as e:
            _LOGGER.warning(f"Google failed: {e}")
            return None

    def _parse_intent_from_response(self, name: str, intent) -> Optional[Intent]:
        """Parse Intent from various response types (dict, str, AIMessage)."""
        import json
        
        if isinstance(intent, dict):
            try:
                return Intent(**intent)
            except Exception as e:
                _LOGGER.warning(f"[{name}] Failed to create Intent from dict: {e}")
                return None
        
        if hasattr(intent, 'content'):
            response_text = intent.content
        elif isinstance(intent, str):
            response_text = intent
        else:
            response_text = str(intent)
        
        _LOGGER.debug(f"[{name}] Response text preview: {response_text[:500]}...")
        
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            _LOGGER.error(f"[{name}] No JSON found in response")
            return None
        
        try:
            json_str = json_match.group(0)
            parsed_dict = json.loads(json_str)
            _LOGGER.info(f"[{name}] ✓ Successfully parsed JSON from text response")
            return Intent(**parsed_dict)
        except json.JSONDecodeError as e:
            _LOGGER.error(f"[{name}] JSON decode error: {e}")
            return None
        except Exception as e:
            _LOGGER.error(f"[{name}] Failed to create Intent from parsed JSON: {e}")
            return None

    def _query_langchain(self, name: str, chain, instruction: str, system_prompt: str) -> Optional[dict]:
        """Executes query using a LangChain chain (OpenAI, OpenRouter, Ollama)."""
        try:
            _LOGGER.info(f"[{name}] Starting LangChain query")
            escaped_sys = system_prompt.replace("{", "{{").replace("}", "}}").replace("{{instruction}}", "{instruction}")
            prompt = ChatPromptTemplate.from_messages([("system", escaped_sys), ("human", "{instruction}")])

            _LOGGER.info(f"[{name}] Invoking chain with instruction: '{instruction[:100]}...'")
            try:
                intent = (prompt | chain).invoke({"instruction": instruction})
                _LOGGER.info(f"[{name}] ✓ Chain.invoke() returned successfully")
            except Exception as invoke_err:
                _LOGGER.error(f"[{name}] Invoke failed/timed out: {invoke_err}")
                raise invoke_err

            if isinstance(intent, Intent):
                _LOGGER.info(f"[{name}] Returned intent: {intent}")
                operations = self._intent_to_pandas_op(intent)
                _LOGGER.info(f"[{name}] Converted to {len(operations)} pandas operations")
                return operations

            _LOGGER.info(f"[{name}] Response is not an Intent object (type: {type(intent).__name__}), attempting to parse...")
            parsed_intent = self._parse_intent_from_response(name, intent)
            
            if not parsed_intent:
                return None
            
            operations = self._intent_to_pandas_op(parsed_intent)
            _LOGGER.info(f"[{name}] Converted to {len(operations)} pandas operations")
            return operations
        except Exception as e:
            _LOGGER.warning(f"[{name}] Failed with exception: {e}", exc_info=True)
            return None

    def _build_column_index(self):
        """Builds a fast lookup index for column names and synonyms."""
        self._cols = list(self.df_schema['columns'])
        self._col_tokens = {c: set(t for t in re.split(r"[ _/\.]+", c.lower()) if t) for c in self._cols}
        self._column_synonyms = {
            "loss": {"loss", "error", "score"}, "score": {"score", "loss", "error"},
            "age": {"age"}, "label": {"label", "class", "target"},
            "origin": {"origin", "split", "dataset"}, "sample_id": {"sample_id", "id", "sample", "index"},
        }

    def _resolve_column(self, user_name: str) -> Optional[str]:
        """Maps a user-provided string to the best matching column name."""
        if not user_name:
            return None
        user_name = user_name.strip().lower()

        for c in self._cols:
            if c.lower() == user_name:
                return c

        user_tokens = set(re.split(r"[ _/\.]+", user_name))
        expanded = set(user_tokens)
        for base, syns in self._column_synonyms.items():
            if base in user_tokens or user_name in syns:
                expanded |= syns

        best_col, best_score = None, 0.0
        for c, ct in self._col_tokens.items():
            if not ct:
                continue
            score = len(expanded & ct) / len(expanded | ct)
            if score > best_score:
                best_score, best_col = score, c

        if best_score < 0.2:
            close = difflib.get_close_matches(user_name, self._cols, n=1, cutoff=0.6)
            if close:
                best_col = close[0]

        return best_col

    def _build_condition_string(self, conditions: List[Condition]) -> Optional[str]:
        """Converts structured conditions into a pandas mask expression."""
        if not conditions:
            return None
            
        parts = []
        for cond in conditions:
            resolved_col = self._resolve_column(cond.column)
            if not resolved_col:
                _LOGGER.warning("Skipping condition due to unresolvable column: %s", cond.column)
                continue

            if re.match(r'^[\w]+$', resolved_col):
                col_ref = resolved_col
            else:
                col_ref = f"`{resolved_col}`"

            op = cond.op.lower()
            val = cond.value

            if op in ("==", "!=", ">", "<", ">=", "<="):
                if isinstance(cond.value, str) and (cond.value.startswith("df[") or cond.value.startswith("df.")):
                    val = cond.value
                else:
                    val = f"'{cond.value}'" if isinstance(cond.value, str) else str(cond.value)
                parts.append(f"({col_ref} {op} {val})")
            elif op == "between" and cond.value is not None and cond.value2 is not None:
                parts.append(f"({col_ref}.between({cond.value}, {cond.value2}))")
            elif op == "contains" and cond.value is not None:
                parts.append(f"({col_ref}.str.contains('{cond.value}', na=False, regex=False))")
            elif op == "in" and cond.value is not None:
                # For checking if value is in a list/array column (e.g., tags)
                # Handle both string representations and actual lists
                parts.append(f"({col_ref}.apply(lambda x: '{cond.value}' in (x if isinstance(x, list) else str(x).split(',')) if pd.notna(x) else False))")

        return " & ".join(parts) if parts else None

    def _intent_to_pandas_op(self, intent: Intent) -> List[dict]:
        """Converts an Intent object (with steps) into a list of dictionaries describing pandas operations."""
        ops = []

        for step in intent.steps:
            if step.kind == "noop":
                continue
                
            op = {"function": None, "params": {}}
            kind = step.kind

            if kind in ("head", "tail"):
                op["function"] = f"df.{kind}"
                op["params"] = {"n": int(step.n) if step.n else 5}

            elif kind == "sort":
                sort_cols = []
                ascending = bool(step.ascending) if step.ascending is not None else True

                if not step.sort_by and step.conditions:
                    for cond in step.conditions:
                        col = self._resolve_column(cond.column)
                        if col:
                            sort_cols.append(col)
                            if isinstance(cond.value, str) and any(x in cond.value.lower() for x in ['desc', 'false']):
                                ascending = False

                if step.sort_by:
                    resolved = [self._resolve_column(c) for c in step.sort_by]
                    sort_cols.extend([c for c in resolved if c])

                if sort_cols:
                    op["function"] = "df.sort_values"
                    op["params"] = {"by": sort_cols, "ascending": ascending}

            elif kind in ("keep", "drop") and step.conditions:
                expr = self._build_condition_string(step.conditions)
                if not expr:
                    continue

                if kind == "keep":
                    if step.keep_frac:
                        op["function"] = "df.drop"
                        op["params"] = {"index": f"df.index.difference(df.query({repr(expr)}).sample(frac={step.keep_frac}).index)"}
                    else:
                        op["function"] = "df.query"
                        op["params"] = {"expr": expr}
                else:
                    base = f"df.query({repr(expr)})"
                    op["function"] = "df.drop"
                    op["params"] = {"index": f"{base}.sample(frac={step.drop_frac}).index" if step.drop_frac else f"{base}.index"}

            elif (kind == "analysis" or (kind in ("keep", "drop") and not step.conditions)) and step.analysis_expression:
                op["function"] = "df.analyze"
                op["params"] = {"code": step.analysis_expression}

            elif kind == "reset":
                op["function"] = "df.reset_view"
                op["params"] = {"__agent_reset__": True}

            if op["function"]:
                ops.append(op)

        return ops
