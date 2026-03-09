import ast
import multiprocessing
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)

# --- 1. AST WHITELIST CONFIGURATION ---
# These are the only Python instructions allowed in our sandbox.
SAFE_NODES = {
    ast.Expression, ast.Module, ast.Interactive,  # Root nodes
    ast.Expr, ast.BinOp, ast.UnaryOp, ast.Compare, # Math & Logic
    ast.BoolOp, ast.And, ast.Or, ast.Not,          # Boolean logic
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert, # Bitwise (often used by Pandas for masks)
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.Call, ast.Attribute, ast.Subscript,        # Methods, properties, and [index]
    ast.Index, ast.Slice, ast.ExtSlice,            # Specialized indexing
    ast.Constant, ast.Name, ast.Load,              # Literals and variables
    ast.List, ast.Tuple, ast.Set, ast.Dict,        # Collection literals
}

# Values that are allowed to be referenced by Name
SAFE_NAMES = {"df", "np", "pd", "True", "False", "None", "nan", "NaN"}

def is_safe_payload(code: str) -> bool:
    """Performs static analysis on the code string to ensure it follows safe patterns."""
    try:
        # Use 'exec' mode to support multi-line statements in direct ops
        tree = ast.parse(code)
    except Exception as e:
        _LOGGER.error(f"Security: Failed to parse code: {e}")
        return False

    for node in ast.walk(tree):
        # A) Node Type Check
        if type(node) not in SAFE_NODES:
            _LOGGER.warning(f"Security Violation: Forbidden instruction '{type(node).__name__}'")
            return False

        # B) Attribute Privacy Check (Block df.__class__ etc)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                _LOGGER.warning(f"Security Violation: Access to private/dunder attribute '{node.attr}'")
                return False

        # C) Variable Access Check
        if isinstance(node, ast.Name):
            if node.id not in SAFE_NAMES:
                _LOGGER.warning(f"Security Violation: Reference to unauthorized variable '{node.id}'")
                return False

    return True

# --- 2. ISOLATED EXECUTION (TIMEOUT) ---

def _worker_wrapper(func, code, globals_dict, locals_dict, conn):
    """Internal wrapper using Pipe for faster transfer."""
    try:
        # Re-inject missing context if needed
        import pandas as pd
        import numpy as np
        globals_dict.update({"pd": pd, "np": np})
        
        if func == "eval":
            result = eval(code, globals_dict, locals_dict)
        else:
            exec(code, globals_dict, locals_dict)
            result = locals_dict.get('df', None)
        
        conn.send(("success", result))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()

def extract_referenced_columns(code: str) -> Optional[list]:
    """Analyze the AST to find which df['column'] or df.column are referenced."""
    try:
        tree = ast.parse(code)
        columns = set()
        for node in ast.walk(tree):
            # Handle df['col']
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == 'df':
                    # Support both df['col'] (Constant) and older Index nodes
                    slc = node.slice
                    if hasattr(ast, 'Constant') and isinstance(slc, ast.Constant) and isinstance(slc.value, str):
                        columns.add(slc.value)
                    elif hasattr(ast, 'Str') and isinstance(slc, ast.Str): # Legacy compat
                        columns.add(slc.s)
            
            # Handle df.col
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'df':
                    columns.add(node.attr)
        
        return list(columns) if columns else None
    except:
        return None

def safe_execute(code: str, df: pd.DataFrame, mode: str = "eval", timeout: int = 20) -> Any:
    # 1. Static AST Check
    if not is_safe_payload(code):
        raise SecurityError("Unauthorized code detected in query.")

    # 2. Performance Optimization: Prune the DataFrame
    referenced_cols = extract_referenced_columns(code)
    if referenced_cols and mode == "eval":
        valid_cols = [c for c in referenced_cols if c in df.columns]
        if valid_cols:
            df_to_send = df[valid_cols]
            _LOGGER.debug(f"Security: Pruned DF to {len(valid_cols)} context columns for sandbox.")
        else:
            df_to_send = df
    else:
        df_to_send = df

    # 3. Setup Sandbox
    safe_globals = {"__builtins__": {}}
    safe_locals = {"df": df_to_send}

    # 4. Use Pipe instead of Queue (faster for large data)
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # We use 'spawn' to be safe with gRPC, but we keep the worker context minimal
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=_worker_wrapper, args=(mode, code, safe_globals, safe_locals, child_conn))
    
    p.start()
    child_conn.close() # Close child end in parent

    try:
        if parent_conn.poll(timeout):
            status, result = parent_conn.recv()
        else:
            p.terminate()
            p.join()
            raise TimeoutError(f"Operation timed out after {timeout} seconds.")
    except EOFError:
        raise RuntimeError("Worker process died unexpectedly.")
    finally:
        parent_conn.close()
        p.join(0.1)
        if p.is_alive():
            p.terminate()

    if status == "error":
        raise RuntimeError(f"Execution Error: {result}")
    
    return result




class SecurityError(Exception):
    """Raised when code fails the AST safety check."""
    pass
