"""Simple optional CLI UI for weightslab experiments.

Usage (from user experiment):
    import weightslab.backend.cli as cli
    cli.initialize(launch_cli_client=True)

This will start a small TCP command server bound to localhost and (optionally)
open a new terminal window running a REPL client connected to that server.

The server exposes a few convenience commands: `pause`, `resume`, `status`,
`list_models`, `list_optimizers`, `dump`, `operate <op_type> <layer_id> <nb|[list]>`.

Security: the server binds to localhost only and accepts plain-text commands.
This is meant for local interactive debugging during development.
"""

from __future__ import annotations

import threading
import logging
import socket
import json
import sys
import os
import time
from typing import Optional, Any

from weightslab.backend.ledgers import GLOBAL_LEDGER, resolve_hp_name, Proxy, set_hyperparam, list_hyperparams
from weightslab.components.global_monitoring import weightslab_rlock, pause_controller


# Get global logger
logger = logging.getLogger(__name__)


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

    try:
        # Any CLI interaction other than help/? should pause training to
        # avoid race conditions while a user inspects or edits state.
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
                    'set': "set_hp <hp_name?> <key.path> <value>  # e.g. set_hp fashion_mnist data.train_loader.batch_size 32",
                }
            }

        name = resolve_hp_name()
        if verb == 'pause' or verb == 'p':
            pause_controller.pause()
            set_hyperparam(name, 'is_training', False)
            return {'ok': True, 'action': 'paused'}

        if verb == 'resume' or verb == 'r':
            pause_controller.resume()
            set_hyperparam(name, 'is_training', True)
            return {'ok': True, 'action': 'resumed'}

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
                # Simplified plot: return only the model's printed representatio
                return {'ok': True, 'model': model_name, 'plot': repr(m)}
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
            with weightslab_rlock:
                try:
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

                    # Operate
                    with m as mm:
                        mm.operate(layer_id, nb, op_type)
                    print(f'[cli] operated on model via context manager')
                    print(f'[cli] new model info: {m}')

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

                    # Operate
                    with m as mm:
                        mm.operate(layer_id, nb, op_type)

                    return {'ok': True, 'operated': True, 'op': (op_type, layer_id, nb), 'model': model_name}

        # Hyperparameters: list / show details and set
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

        if verb in ('set_hp', 'sethp', 'set-hp'):
            # syntax: set_hp [hp_name] <key.path> <value>
            try:
                names = list_hyperparams()
                # decide whether hp_name was supplied
                if len(parts) < 3:
                    return {'ok': False, 'error': 'usage: set_hp [hp_name] <key.path> <value>'}

                # If multiple hyperparam sets exist, require explicit name
                candidate = parts[1]
                if candidate in names and len(parts) >= 4:
                    hp_name = candidate
                    key = parts[2]
                    raw_value = ' '.join(parts[3:])
                else:
                    # no explicit hp_name provided
                    if len(names) == 1:
                        hp_name = names[0]
                        key = parts[1]
                        raw_value = ' '.join(parts[2:])
                    else:
                        return {'ok': False, 'error': 'Multiple hyperparam sets present; provide hp_name explicitly'}

                # Try to coerce value: JSON first, then common literals
                value = None
                try:
                    import json as _json
                    value = _json.loads(raw_value)
                except Exception:
                    # fallback: try booleans / numbers
                    lv = raw_value.strip()
                    if lv.lower() in ('true', 'false'):
                        value = True if lv.lower() == 'true' else False
                    else:
                        try:
                            if '.' in lv:
                                value = float(lv)
                            else:
                                value = int(lv)
                        except Exception:
                            value = lv

                # apply change
                try:
                    set_hyperparam(hp_name, key, value)
                    return {'ok': True, 'hp_name': hp_name, 'key': key, 'value': value}
                except Exception as e:
                    return {'ok': False, 'error': str(e)}
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
            try:
                line = conn_file.readline()
            except ConnectionResetError:
                break
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
    """Legacy entry: bind and serve in this thread."""
    global _server_sock, _server_port, _server_host
    _server_host = host
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(5)
        _server_port = srv.getsockname()[1]
        _server_sock = srv
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=_client_handler, args=(conn, addr), name='cli_serving_loop', daemon=True)
            t.start()
    finally:
        try:
            srv.close()
        except Exception:
            pass


def _server_loop_sock(srv: socket.socket):
    """Serve on an already bound/listening socket (avoids rebind race)."""
    global _server_sock, _server_port, _server_host
    try:
        _server_sock = srv
        _server_host = srv.getsockname()[0]
        _server_port = srv.getsockname()[1]
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=_client_handler, args=(conn, addr), name='cli_serving_loop', daemon=True)
            t.start()
    finally:
        try:
            srv.close()
        except Exception:
            pass


def cli_serve(cli_host: str = 'localhost', cli_port: int = 60000, *, spawn_client: bool = True, **_):
    """
        Start the CLI server and optionally open a client in a new console.
        This CLI now operates on objects registered in the global ledger only.

        Args:
            cli_host: Host to bind the server to (default: localhost).
    """

    # Lazy import of banner to avoid importing the top-level
    # package (and thus torch) when CUDA DLLs are unavailable.
    # On some Windows setups, importing torch can fail with
    # WinError 1455; the CLI should still be able to start.
    _BANNER = None
    try:
        from weightslab import _BANNER as _WB
        _BANNER = _WB
    except Exception:
        # Skip banner if importing the package fails
        _BANNER = None

    global _server_thread, _server_port, _server_host
    cli_host = os.environ.get('CLI_HOST', cli_host)
    cli_port = int(os.environ.get('CLI_PORT', cli_port))

    if _server_thread is not None and _server_thread.is_alive():
        return {'ok': False, 'error': 'server_already_running'}

    # start server thread on a pre-bound socket to avoid races
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((cli_host, cli_port))
        srv.listen(5)
        actual_port = srv.getsockname()[1]
    except Exception as e:
        logger.exception("cli_bind_failed")
        return {'ok': False, 'error': f'bind_failed: {e}'}

    _server_thread = threading.Thread(
        target=_server_loop_sock,
        args=(srv,),
        daemon=True,
        name="WeightsLab CLI Server",
    )
    _server_thread.start()
    logger.info("cli_thread_started", extra={
        "thread_name": _server_thread.name,
        "thread_id": _server_thread.ident,
        "cli_host": cli_host,
        "cli_port": cli_port,
        "actual_port": actual_port
    })
    # wait briefly for server to come up
    time.sleep(0.05)

    if _BANNER:
        try:
            print(_BANNER)
        except Exception:
            pass

    # optionally spawn a new console running the client REPL
    if spawn_client:
        import subprocess
        cmd = [sys.executable, '-u', '-c',
                "import weightslab.backend.cli as _cli; _cli.cli_client_main('%s', %d)" % (cli_host, actual_port)]
        kwargs = {}
        if os.name == 'nt':
            # CREATE_NEW_CONSOLE causes a new terminal window on Windows
            kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(cmd, cwd=os.getcwd(), **kwargs)

    return {'ok': True, 'host': cli_host, 'port': actual_port}


def cli_client_main(cli_host: str = 'localhost', cli_port: int = 60000):
    """Simple client REPL that connects to the CLI server.

    This is intended to be launched in a separate console window.
    """
    addr = (cli_host, cli_port)
    print(f"weightslab CLI connecting to {cli_host}:{cli_port}...")
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
    # Simple argv parsing for serve/client modes
    import argparse
    parser = argparse.ArgumentParser(description='WeightsLab CLI server/client')
    sub = parser.add_subparsers(dest='mode')

    pserve = sub.add_parser('serve', help='Start CLI server')
    pserve.add_argument('--host', default='localhost')
    pserve.add_argument('--port', type=int, default=0)
    pserve.add_argument('--no-spawn-client', action='store_true', help='Do not spawn client console')

    pclient = sub.add_parser('client', help='Start CLI client')
    pclient.add_argument('--host', default='localhost')
    pclient.add_argument('--port', type=int, required=True)

    args = parser.parse_args()

    # Default to serving if no subcommand provided
    if args.mode in (None, 'serve'):
        spawn = False if (getattr(args, 'no_spawn_client', False)) else True
        info = cli_serve(cli_host=getattr(args, 'host', 'localhost'), cli_port=getattr(args, 'port', 0), spawn_client=spawn)
        if not info.get('ok'):
            print('Failed to start server:', info)
            sys.exit(1)
        host, port = info['host'], info['port']
        print(f'CLI server running on {host}:{port}. Press Ctrl+C to stop.')
        try:
            # Keep process alive so daemon thread does not exit
            while _server_thread is not None and _server_thread.is_alive():
                time.sleep(1.0)
        except KeyboardInterrupt:
            print('Stopping CLI server...')
            # Close server socket to unblock accept()
            try:
                if _server_sock:
                    _server_sock.close()
            except Exception:
                pass
            sys.exit(0)
    else:
        # client mode
        cli_client_main(cli_host=getattr(args, 'host', 'localhost'), cli_port=getattr(args, 'port'))
