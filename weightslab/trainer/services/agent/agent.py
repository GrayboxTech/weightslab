import os
import json
import logging
import requests
import difflib
import re
import pandas as pd
from typing import Optional, List, Union, Literal
from .intent_prompt import INTENT_PROMPT
from dotenv import load_dotenv
from pathlib import Path

# New Google GenAI SDK
from google import genai

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

_LOGGER = logging.getLogger(__name__)

# Try to find .env in weightslab/ or parent root
env_path = Path(__file__).resolve().parents[3] / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# --- Pydantic Models ---
class Condition(BaseModel):
    column: str = Field(description="The column name to filter/check")
    op: Literal["==", "!=", ">", "<", ">=", "<=", "between"] = Field(description="The operator")
    value: Optional[Union[float, int, str]] = Field(default=None, description="The primary value")
    value2: Optional[Union[float, int]] = Field(default=None, description="The secondary value for 'between'")

class Intent(BaseModel):
    kind: Literal["keep", "drop", "sort", "head", "tail", "reset", "noop"] = Field(description="The kind of operation")
    conditions: Optional[List[Condition]] = Field(default=None, description="Conditions for keep/drop")
    sort_by: Optional[List[str]] = Field(default=None, description="Columns to sort by")
    ascending: Optional[bool] = Field(default=None, description="Sort ascending (true) or descending (false)")
    n: Optional[int] = Field(default=None, description="Number of rows for head/tail")
    drop_frac: Optional[float] = Field(default=None, description="Fraction of rows to drop (0.0 to 1.0)")
    keep_frac: Optional[float] = Field(default=None, description="Fraction of rows to keep (0.0 to 1.0)")

class DataManipulationAgent:
    def __init__(self, context):
        """
        Initializes the agent with context and builds the column schema/index.
        """
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
        
        # Add index names to columns list
        if isinstance(df.index, pd.MultiIndex):
            all_columns.extend([name for name in df.index.names if name is not None])
        elif df.index.name is not None:
            all_columns.append(df.index.name)

        # Add common prediction stats if missing
        expected_stats = [
            "mean_loss", "max_loss", "min_loss", "std_loss", "median_loss",
            "num_classes_present", "dominant_class", "dominant_class_ratio",
            "background_ratio", "predicted_class"
        ]
        # Add per-class losses (0-9)
        expected_stats.extend([f"loss_class_{i}" for i in range(10)])

        for col in expected_stats:
            if col not in all_columns:
                all_columns.append(col)

        self.df_schema = {
            'columns': all_columns,
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        }

    def _load_config(self):
        """Loads agent configuration from environment and YAML hyperparameters."""
        # 1. Defaults from ENV
        self.preferred_provider = os.environ.get("AGENT_PROVIDER", "google").lower()
        self.google_model = os.environ.get("AGENT_MODEL_GOOGLE", "gemini-1.5-flash")
        self.openai_model = os.environ.get("AGENT_MODEL_OPENAI", "gpt-4o-mini")
        self.openrouter_model = os.environ.get("AGENT_MODEL_OPENROUTER", "mistralai/mistral-7b-instruct:free")
        self.fallback_to_local = True 

        # 2. Override with Experiment YAML Config
        try:
            hp = self._get_hyperparams()
            if hp and "agent" in hp:
                cfg = hp["agent"]
                self.preferred_provider = cfg.get("provider", self.preferred_provider).lower()
                
                # Update model ID for the active provider
                model_id = cfg.get("model")
                if model_id:
                    attr_map = {"google": "google_model", "openai": "openai_model", "openrouter": "openrouter_model"}
                    if self.preferred_provider in attr_map:
                        setattr(self, attr_map[self.preferred_provider], model_id)
                
                self.fallback_to_local = bool(cfg.get("fallback_to_local", self.fallback_to_local))
                _LOGGER.info(f"Agent config loaded from YAML: provider={self.preferred_provider}, model={model_id}")
        except Exception as e:
            _LOGGER.warning(f"Could not load agent config from hyperparams: {e}")

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
        
        # 1. OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                llm = ChatOpenAI(model=self.openai_model, temperature=0, api_key=api_key)
                self.chain_openai = llm.with_structured_output(Intent)
                _LOGGER.info(f"OpenAI active ({self.openai_model})")
            except Exception as e: _LOGGER.error(f"OpenAI error: {e}")

        # 2. Google (Direct SDK)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            try:
                self.client_google = genai.Client(api_key=api_key)
                # Cleanup model ID
                if self.google_model.startswith("models/"):
                    self.google_model = self.google_model.replace("models/", "")
                aliases = {"gemini-flash": "gemini-1.5-flash", "gemini-pro": "gemini-1.5-pro"}
                self.google_model = aliases.get(self.google_model, self.google_model)
                _LOGGER.info(f"Google active ({self.google_model})")
            except Exception as e: _LOGGER.error(f"Google error: {e}")

        # 3. OpenRouter
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            try:
                llm = ChatOpenAI(
                    model=self.openrouter_model, temperature=0, api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                self.chain_openrouter = llm.with_structured_output(Intent)
                _LOGGER.info(f"OpenRouter active ({self.openrouter_model})")
            except Exception as e: _LOGGER.error(f"OpenRouter error: {e}")

        # 4. Ollama (Local)
        try:
            host = os.environ.get('OLLAMA_HOST', 'localhost').split(':')[0]
            port = os.environ.get('OLLAMA_PORT', '11435')
            llm = ChatOllama(base_url=f"http://{host}:{port}", model="llama3.2:1b", temperature=0)
            self.chain_ollama = llm.with_structured_output(Intent)
            _LOGGER.info("Ollama active (llama3.2:1b)")
        except Exception as e: _LOGGER.error(f"Ollama error: {e}")

    def is_ollama_available(self) -> bool:
        return self.chain_ollama is not None

    def query(self, instruction: str) -> dict:
        """Main entry point for natural language instructions."""
        system_prompt = INTENT_PROMPT.format(
            columns=self.df_schema['columns'],
            instruction="{instruction}"
        )
        
        # Build attempt order
        order = [self.preferred_provider]
        if self.fallback_to_local and self.preferred_provider != "ollama":
            order.append("ollama")
        
        # Deduplicate while preserving order
        order = list(dict.fromkeys(order))

        for provider in order:
            result = self._try_query_provider(provider, instruction, system_prompt)
            if result:
                return result
        
        return {"function": None, "params": {}}

    def _try_query_provider(self, provider: str, instruction: str, system_prompt: str) -> Optional[dict]:
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

    def _query_langchain(self, name: str, chain, instruction: str, system_prompt: str) -> Optional[dict]:
        """Executes query using a LangChain chain (OpenAI, OpenRouter, Ollama)."""
        try:
            _LOGGER.info(f"Calling {name.title()}")
            # Escape braces for LangChain f-string parser
            escaped_sys = system_prompt.replace("{", "{{").replace("}", "}}").replace("{{instruction}}", "{instruction}")
            prompt = ChatPromptTemplate.from_messages([("system", escaped_sys), ("human", "{instruction}")])
            intent = (prompt | chain).invoke({"instruction": instruction})
            _LOGGER.info(f"{name.title()} returned: {intent}")
            return self._intent_to_pandas_op(intent)
        except Exception as e:
            _LOGGER.warning(f"{name.title()} failed: {e}")
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
        if not user_name: return None
        user_name = user_name.strip().lower()
        
        # 1. Exact match
        for c in self._cols:
            if c.lower() == user_name: return c
            
        # 2. Token-based Jaccard similarity
        user_tokens = set(re.split(r"[ _/\.]+", user_name))
        expanded = set(user_tokens)
        for base, syns in self._column_synonyms.items():
            if base in user_tokens or user_name in syns: expanded |= syns
            
        best_col, best_score = None, 0.0
        for c, ct in self._col_tokens.items():
            if not ct: continue
            score = len(expanded & ct) / len(expanded | ct)
            if score > best_score:
                best_score, best_col = score, c
                
        # 3. Fuzzy match fallback
        if best_score < 0.2:
            close = difflib.get_close_matches(user_name, self._cols, n=1, cutoff=0.6)
            if close: best_col = close[0]
            
        return best_col

    def _build_condition_string(self, conditions: List[Condition]) -> Optional[str]:
        """Converts structured conditions into a pandas query string."""
        if not conditions: return None
        parts = []
        for cond in conditions:
            col = self._resolve_column(cond.column)
            if not col: continue
            
            ref = col if re.match(r'^[\w]+$', col) else f"`{col}`"
            op = cond.op.lower().strip()
            # Normalize operators
            if op in ("===", "==", "is", "equals"): op = "=="
            elif op in ("!==", "!="): op = "!="
            
            if op in ("==", "!=", ">", "<", ">=", "<="):
                val = f"'{cond.value}'" if isinstance(cond.value, str) else str(cond.value)
                parts.append(f"({ref} {op} {val})")
            elif op == "between" and cond.value is not None and cond.value2 is not None:
                parts.append(f"({ref}.between({cond.value}, {cond.value2}))")
                
        return " and ".join(parts) if parts else None

    def _intent_to_pandas_op(self, intent: Intent) -> dict:
        """Converts an Intent object into a dictionary describing the pandas operation."""
        op = {"function": None, "params": {}}
        kind = intent.kind
        
        if kind == "noop": return op
        
        if kind in ("head", "tail"):
            op["function"] = f"df.{kind}"
            op["params"] = {"n": int(intent.n) if intent.n else 5}
            
        elif kind == "sort" and intent.sort_by:
            cols = [self._resolve_column(c) for c in intent.sort_by]
            cols = [c for c in cols if c]
            if cols:
                op["function"] = "df.sort_values"
                op["params"] = {"by": cols, "ascending": bool(intent.ascending) if intent.ascending is not None else True}
                
        elif kind in ("keep", "drop") and intent.conditions:
            expr = self._build_condition_string(intent.conditions)
            if not expr: return op
            
            if kind == "keep":
                if intent.keep_frac:
                    op["function"] = "df.drop"
                    op["params"] = {"index": f"df.index.difference(df.query({repr(expr)}).sample(frac={intent.keep_frac}).index)"}
                else:
                    op["function"] = "df.query"
                    op["params"] = {"expr": expr}
            else:
                base = f"df.query({repr(expr)})"
                op["function"] = "df.drop"
                op["params"] = {"index": f"{base}.sample(frac={intent.drop_frac}).index" if intent.drop_frac else f"{base}.index"}
                
        elif kind == "reset":
            op["function"] = "df.reset_view"
            op["params"] = {"__agent_reset__": True}
            
        return op
