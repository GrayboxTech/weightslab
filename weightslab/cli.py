"""Simple optional CLI UI for weightslab experiments.

Usage (from user experiment):
    import weightslab.cli as cli
    cli.initialize(launch_client=True)

This will start a small TCP command server bound to localhost and (optionally)
open a new terminal window running a REPL client connected to that server.

The server exposes a few convenience commands: `pause`, `resume`, `status`,
`list_models`, `list_optimizers`, `dump`, `operate <op_type> <layer_id> <nb|[list]>`.

Security: the server binds to localhost only and accepts plain-text commands.
This is meant for local interactive debugging during development.
"""

from __future__ import annotations

import threading
import socket
import json
import sys
import os
import time
from typing import Optional, Any

from .art import _BANNER

from weightslab.ledgers import GLOBAL_LEDGER
from weightslab.ledgers import list_hyperparams, get_hyperparams, Proxy


def _sanitize_for_json(obj):
    """Recursively convert objects that are not JSON-serializable into
    serializable representations. In particular, handle `Proxy` objects by
    extracting their underlying target (via `get()`) when possible.
    """
    # primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    # Proxy handling
    if isinstance(obj, Proxy):
        try:
            inner = obj.get()
            return _sanitize_for_json(inner)
        except Exception:
            return repr(obj)
    # dict -> sanitize values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = k if isinstance(k, str) else str(k)
            except Exception:
                key = str(k)
            out[key] = _sanitize_for_json(v)
        return out
    # list/tuple/set -> list
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]
    # fallback: try to stringify
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _format_model_tree(model, max_depth: int = 6) -> str:
    """Return a simple ASCII tree of `model` using named_children().

    This is intentionally light-weight (no shapes) and is safe for typical
    torch.nn.Module subclasses. If the object is not a module, fallback to
    repr().
    """
    try:
        import torch.nn as nn
    except Exception:
        nn = None

    def _rec(m, prefix='', depth=0, visited=None):
        if visited is None:
            visited = set()
        lines = []
        if depth > max_depth:
            lines.append(prefix + '...')
            return lines
        try:
            cls_name = m.__class__.__name__
        except Exception:
            cls_name = repr(m)
        # avoid recursion cycles
        if id(m) in visited:
            lines.append(prefix + f'{cls_name} (recursed)')
            return lines
        visited.add(id(m))

        lines.append(prefix + cls_name)
        # iterate named children if available
        try:
            children = list(getattr(m, 'named_children')() ) if hasattr(m, 'named_children') else []
        except Exception:
            children = []

        for i, (n, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            branch = '└─ ' if is_last else '├─ '
            sub_prefix = '   ' if is_last else '│  '
            lines.extend(_rec(child, prefix + branch, depth + 1, visited))
            # adjust subsequent sibling prefixes
            if i < len(children) - 1:
                # insert connector for next siblings
                pass
        return lines

    try:
        # If it's a torch module, traverse; otherwise return repr
        if nn is not None and isinstance(model, nn.Module):
            tree_lines = _rec(model)
            return '\n'.join(tree_lines)
        else:
            return repr(model)
    except Exception:
        return repr(model)

# Globals for the running server
_server_thread: Optional[threading.Thread] = None
_server_sock: Optional[socket.socket] = None
_server_host: str = '127.0.0.1'
_server_port: int = 0


def _handle_command(cmd: str) -> Any:
    """Handle a single textual command and return a JSON-serializable result."""
    cmd = cmd.strip()
    if not cmd:
        return {'ok': True}

    parts = cmd.split()
    verb = parts[0].lower()

    def _auto_pause_for_cli_interaction():
        # Pause all registered models to avoid concurrent training changes
        try:
            names = GLOBAL_LEDGER.list_models()
            for n in names:
                try:
                    m = GLOBAL_LEDGER.get_model(n)
                    try:
                        m.pause_ctrl.pause()
                    except Exception:
                        try:
                            m.pause()
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # best-effort; ignore failures
            pass

    try:
        # Any CLI interaction other than help/? should pause training to
        # avoid race conditions while a user inspects or edits state.
        if verb not in ('help', '?'):
            _auto_pause_for_cli_interaction()
        if verb in ('help', '?'):
            # Provide a structured, machine-friendly help payload so the client
            # can pretty-print it and users can discover hyperparam commands.
            return {
                'ok': True,
                'description': 'weightslab CLI - available commands and usage',
                'commands': {
                    'pause / p': 'Pause training. Syntax: pause [model_name]',
                    'resume / r': 'Resume training. Syntax: resume [model_name] ',
                    'status': 'Show basic status: registered models and optimizers',
                    'list_models': 'List registered model names in the ledger',
                    'list_optimizers': 'List registered optimizer names in the ledger',
                                    'dump': 'Return a sanitized dump of the ledger contents',
                    'operate': 'Edit model architecture. Syntax: operate [<model_name>] <op_type:int> <layer_id:int> <nb|[list]>',
                            'plot_model': 'Show ASCII tree of model architecture. Syntax: plot_model [<model_name>]',
                    'hp / hyperparams': 'List or show hyperparameters. Syntax: hp -> list, hp <name> -> show',
                    # editing hyperparameters is disabled in the CLI (read-only)
                    'quit / exit': 'Close the client connection'
                },
                'hyperparams_examples': {
                    'list': 'hp',
                    'show': 'hp fashion_mnist',
                    'set_number': "set_hp fashion_mnist data.train_dataset.batch_size 32",
                    'set_list': "set_hp fashion_mnist optimizer.Adam.lr 0.001",
                }
            }

        if verb == 'pause' or verb == 'p':
            # syntax: pause [model_name]
            target_model = None
            if len(parts) > 1:
                name = parts[1]
                try:
                    target_model = GLOBAL_LEDGER.get_model(name)
                except Exception:
                    target_model = None
            if target_model is not None:
                try:
                    target_model.pause_ctrl.pause()
                except Exception:
                    # try attribute on object itself
                    try:
                        target_model.pause()
                    except Exception:
                        pass
                return {'ok': True, 'action': 'paused', 'model': name}

            # fallback: try to get single model from ledger
            try:
                tgt = GLOBAL_LEDGER.get_model()
                try:
                    tgt.pause_ctrl.pause()
                except Exception:
                    try:
                        tgt.pause()
                    except Exception:
                        pass
                return {'ok': True, 'action': 'paused', 'model': None}
            except Exception:
                return {'ok': False, 'error': 'no_model_registered'}

        if verb == 'resume' or verb == 'r':
            # syntax: resume [model_name]
            target_model = None
            if len(parts) > 1:
                name = parts[1]
                try:
                    target_model = GLOBAL_LEDGER.get_model(name)
                except Exception:
                    target_model = None
            if target_model is not None:
                try:
                    target_model.pause_ctrl.resume()
                except Exception:
                    try:
                        target_model.resume()
                    except Exception:
                        pass
                return {'ok': True, 'action': 'resumed', 'model': name}

            # fallback: try to get single model from ledger
            try:
                tgt = GLOBAL_LEDGER.get_model()
                try:
                    tgt.pause_ctrl.resume()
                except Exception:
                    try:
                        tgt.resume()
                    except Exception:
                        pass
                return {'ok': True, 'action': 'resumed', 'model': None}
            except Exception:
                return {'ok': False, 'error': 'no_model_registered'}

        if verb == 'status':
            # Provide a compact ledger snapshot in status: models, dataloaders,
            # optimizers and hyperparams. This gives a quick overview of the
            # current registrations without returning full object dumps.
            snap = GLOBAL_LEDGER.snapshot()
            try:
                snap['hyperparams'] = GLOBAL_LEDGER.list_hyperparams()
            except Exception:
                snap['hyperparams'] = []

            # try to include simple model info (age) from a single registered model
            model_age = None
            try:
                single = GLOBAL_LEDGER.get_model()
                try:
                    model_age = single.get_age()
                except Exception:
                    model_age = None
            except Exception:
                model_age = None

            return {'ok': True, 'snapshot': snap, 'model_age': model_age}

        if verb == 'list_models':
            return {'ok': True, 'models': GLOBAL_LEDGER.list_models()}

        if verb in ('plot_model', 'plot_arch', 'plot'):
            # syntax: plot_model [model_name]
            model_name = None
            if len(parts) > 1:
                model_name = parts[1]
            try:
                if model_name:
                    m = GLOBAL_LEDGER.get_model(model_name)
                else:
                    m = GLOBAL_LEDGER.get_model()
            except Exception:
                return {'ok': False, 'error': 'no_model_registered'}

            # unwrap Proxy if present
            try:
                if isinstance(m, Proxy):
                    m = m.get()
            except Exception:
                pass

            try:
                ascii_tree = _format_model_tree(m)
                return {'ok': True, 'model': model_name, 'plot': ascii_tree}
            except Exception as e:
                return {'ok': False, 'error': str(e)}

        if verb == 'list_dataloaders':
            return {'ok': True, 'dataloaders': GLOBAL_LEDGER.list_dataloaders()}

        if verb == 'list_optimizers':
            return {'ok': True, 'optimizers': GLOBAL_LEDGER.list_optimizers()}

        # Return a lightweight snapshot of all ledger registries
        if verb in ('ledgers', 'ledger', 'snapshot'):
            snap = GLOBAL_LEDGER.snapshot()
            # include hyperparams names as well
            try:
                snap['hyperparams'] = GLOBAL_LEDGER.list_hyperparams()
            except Exception:
                snap['hyperparams'] = []
            return {'ok': True, 'snapshot': snap}

        # Dump full ledger contents (may be large). Sanitizes values.
        if verb in ('ledger_dump', 'dump_ledger', 'dump_ledger_all'):
            out = {}
            snap = GLOBAL_LEDGER.snapshot()
            # models / dataloaders / optimizers
            for k in ('models', 'dataloaders', 'optimizers'):
                out[k] = {}
                for name in snap.get(k, []):
                    try:
                        getter = {
                            'models': GLOBAL_LEDGER.get_model,
                            'dataloaders': GLOBAL_LEDGER.get_dataloader,
                            'optimizers': GLOBAL_LEDGER.get_optimizer,
                        }[k]
                        val = getter(name)
                        out[k][name] = _sanitize_for_json(val)
                    except Exception as e:
                        out[k][name] = str(e)
            # hyperparams
            try:
                out['hyperparams'] = {}
                for name in GLOBAL_LEDGER.list_hyperparams():
                    try:
                        hp = GLOBAL_LEDGER.get_hyperparams(name)
                        out['hyperparams'][name] = _sanitize_for_json(hp)
                    except Exception as e:
                        out['hyperparams'][name] = str(e)
            except Exception:
                out['hyperparams'] = {}
            return {'ok': True, 'ledger': out}

        if verb == 'dump':
            # legacy `dump` now returns the ledger dump (sanitized)
            out = {}
            snap = GLOBAL_LEDGER.snapshot()
            for k in ('models', 'dataloaders', 'optimizers'):
                out[k] = {}
                for name in snap.get(k, []):
                    try:
                        getter = {
                            'models': GLOBAL_LEDGER.get_model,
                            'dataloaders': GLOBAL_LEDGER.get_dataloader,
                            'optimizers': GLOBAL_LEDGER.get_optimizer,
                        }[k]
                        val = getter(name)
                        out[k][name] = _sanitize_for_json(val)
                    except Exception as e:
                        out[k][name] = str(e)
            try:
                out['hyperparams'] = {}
                for name in GLOBAL_LEDGER.list_hyperparams():
                    try:
                        hp = GLOBAL_LEDGER.get_hyperparams(name)
                        out['hyperparams'][name] = _sanitize_for_json(hp)
                    except Exception as e:
                        out['hyperparams'][name] = str(e)
            except Exception:
                out['hyperparams'] = {}
            return {'ok': True, 'ledger': out}

        if verb == 'operate':
            # syntax:
            #  - operate <op_type:int> <layer_id:int> <nb>
            #  - operate <model_name> <op_type:int> <layer_id:int> <nb>
            if len(parts) < 4:
                return {'ok': False, 'error': 'usage: operate [<model_name>] <op_type> <layer_id> <nb|[list]>'}

            # detect whether second token is model name or op_type
            try:
                possible_op = int(parts[1])
                # form: operate <op_type> <layer_id> <nb>

                # no wl_exp: operate on single ledger model
                op_type = int(parts[1])
                layer_id = int(parts[2])
                raw = ' '.join(parts[3:])
                try:
                    nb = eval(raw, {}, {})
                except Exception:
                    try:
                        nb = int(parts[3])
                    except Exception:
                        nb = raw
                try:
                    m = GLOBAL_LEDGER.get_model()
                except Exception:
                    return {'ok': False, 'error': 'no_model_registered'}
                try:
                    with m as mm:
                        mm.operate(layer_id, nb, op_type)
                    print(f'[cli] operated on model via context manager')
                    print(f'[cli] new model info: {m}')
                except Exception:
                    # try direct call
                    m.operate(layer_id, nb, op_type)
                return {'ok': True, 'operated': True, 'op': (op_type, layer_id, nb), 'model': None}
            except ValueError:
                # parts[1] is not an int => treat as model name
                model_name = parts[1]
                if len(parts) < 5:
                    return {'ok': False, 'error': 'usage: operate <model_name> <op_type> <layer_id> <nb|[list]>'}
                try:
                    op_type = int(parts[2])
                    layer_id = int(parts[3])
                except Exception:
                    return {'ok': False, 'error': 'op_type and layer_id must be ints'}
                raw = ' '.join(parts[4:])
                try:
                    nb = eval(raw, {}, {})
                except Exception:
                    try:
                        nb = int(parts[4])
                    except Exception:
                        nb = raw
                try:
                    m = GLOBAL_LEDGER.get_model(model_name)
                except Exception:
                    return {'ok': False, 'error': f'model_not_found: {model_name}'}
                try:
                    with m as mm:
                        mm.operate(layer_id, nb, op_type)
                except Exception:
                    m.operate(layer_id, nb, op_type)
                return {'ok': True, 'operated': True, 'op': (op_type, layer_id, nb), 'model': model_name}

        # Hyperparameters: list / show / set
        if verb in ('hp', 'hyperparams'):
            # hp -> list
            names = GLOBAL_LEDGER.list_hyperparams() if hasattr(GLOBAL_LEDGER, 'list_hyperparams') else []
            if len(parts) == 1:
                return {'ok': True, 'hyperparams': names}
            # support: hp list  -> same as hp
            name = parts[1]
            if name.lower() in ('list', 'ls', 'all'):
                return {'ok': True, 'hyperparams': names}
            # support: hp show <name>
            if name.lower() in ('show',) and len(parts) > 2:
                name = parts[2]
            try:
                hp = GLOBAL_LEDGER.get_hyperparams(name)
                return {'ok': True, 'name': name, 'hyperparams': hp}
            except Exception as e:
                return {'ok': False, 'error': str(e)}

        # Editing hyperparameters via CLI is intentionally disabled.

        return {'ok': False, 'error': f'unknown_command: {verb}'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def _client_handler(conn: socket.socket, addr):
    with conn:
        conn_file = conn.makefile('rwb')
        # greet
        greeting = {'ok': True, 'welcome': 'weightslab-cli', 'time': time.time()}
        conn_file.write((json.dumps(greeting) + "\n").encode('utf8'))
        conn_file.flush()
        while True:
            line = conn_file.readline()
            if not line:
                break
            cmd = line.decode('utf8').rstrip('\n')
            if cmd.lower().strip() in ('quit', 'exit'):
                resp = {'ok': True, 'bye': True}
                conn_file.write((json.dumps(resp) + "\n").encode('utf8'))
                conn_file.flush()
                break
            resp = _handle_command(cmd)
            # sanitize response to ensure JSON serialization succeeds
            try:
                safe = _sanitize_for_json(resp)
                conn_file.write((json.dumps(safe) + "\n").encode('utf8'))
            except Exception:
                # fallback: send a stringified representation
                try:
                    conn_file.write((json.dumps({'ok': False, 'error': str(resp)}) + "\n").encode('utf8'))
                except Exception:
                    conn_file.write((json.dumps({'ok': False, 'error': 'unserializable_response'}) + "\n").encode('utf8'))
            conn_file.flush()


def _server_loop(host: str, port: int):
    global _server_sock, _server_port, _server_host
    _server_host = host
    _server_port = port
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(5)
    # update actual port if 0 was chosen
    _server_port = srv.getsockname()[1]
    _server_sock = srv
    try:
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=_client_handler, args=(conn, addr), daemon=True)
            t.start()
    finally:
        try:
            srv.close()
        except Exception:
            pass


def initialize(host: str = '127.0.0.1', port: int = 0, launch_client: bool = True):
    """Start the CLI server and optionally open a client in a new console.

    This CLI now operates on objects registered in the global ledger only.
    """
    global _server_thread, _server_port, _server_host

    if _server_thread is not None and _server_thread.is_alive():
        return {'ok': False, 'error': 'server_already_running'}

    # start server thread
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    actual_port = sock.getsockname()[1]
    sock.close()

    _server_thread = threading.Thread(target=_server_loop, args=(host, actual_port), daemon=True)
    _server_thread.start()
    # wait briefly for server to come up
    time.sleep(0.05)

    try:
        print(_BANNER)
    except Exception:
        pass

    if launch_client:
        # spawn a new console running the client REPL
        import subprocess
        cmd = [sys.executable, '-u', '-c',
               "import weightslab.cli as _cli; _cli.cli_client_main('%s', %d)" % (host, actual_port)]
        kwargs = {}
        if os.name == 'nt':
            # CREATE_NEW_CONSOLE causes a new terminal window on Windows
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(cmd, cwd=os.getcwd(), **kwargs)

    return {'ok': True, 'host': host, 'port': actual_port}


def cli_client_main(host: str, port: int):
    """Simple client REPL that connects to the CLI server.

    This is intended to be launched in a separate console window.
    """
    addr = (host, port)
    print(f"weightslab CLI connecting to {host}:{port}...")
    try:
        s = socket.create_connection(addr)
    except Exception as e:
        print('Failed to connect to server:', e)
        return

    f = s.makefile('rwb')
    # read greeting
    line = f.readline()
    try:
        gre = json.loads(line.decode('utf8'))
        print('Server:', gre.get('welcome'), 'time:', gre.get('time'))
    except Exception:
        pass

    try:
        while True:
            try:
                cmd = input('wl> ').strip()
            except EOFError:
                cmd = 'exit'
            if not cmd:
                continue

            # Local client-side convenience: clear the terminal without
            # sending a command to the server. Support both 'clear' and 'cls'.
            if cmd.lower() in ('clear', 'cls'):
                try:
                    if os.name == 'nt':
                        os.system('cls')
                    else:
                        os.system('clear')
                except Exception:
                    pass
                continue
            f.write((cmd + "\n").encode('utf8'))
            f.flush()
            resp_line = f.readline()
            if not resp_line:
                print('Connection closed by server')
                break
            try:
                resp = json.loads(resp_line.decode('utf8'))
                # Pretty-print JSON responses for readability (help, status, etc.)
                try:
                    pretty = json.dumps(resp, indent=2, ensure_ascii=False)
                except Exception:
                    pretty = str(resp)
                print(pretty)
            except Exception:
                # fallback: print raw line decoded
                try:
                    print(resp_line.decode('utf8').rstrip('\n'))
                except Exception:
                    print(resp_line)
            if cmd.lower() in ('exit', 'quit'):
                break
    finally:
        try:
            f.close()
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass


if __name__ == '__main__':
    # allow running a client directly for debugging
    if len(sys.argv) >= 3:
        cli_client_main(sys.argv[1], int(sys.argv[2]))
    else:
        print('weightslab.cli: run initialize() from your experiment to launch server')
