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


def _safe_repr(obj) -> str:
    try:
        return str(obj)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            try:
                return object.__repr__(obj)
            except Exception:
                return f"<{type(obj).__name__}>"


def _sanitize_for_json(obj):
    """Recursively convert objects that are not JSON-serializable into
    serializable representations. In particular, handle `Proxy` objects by
    extracting their underlying target (via `get()`) when possible.
    """
    # primitives
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='replace')
    # Proxy handling
    if isinstance(obj, Proxy):
        try:
            inner = obj.get()
            return _sanitize_for_json(inner)
        except Exception:
            return _safe_repr(obj)
    # dict -> sanitize values
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = k if isinstance(k, str) else _safe_repr(k)
            out[key] = _sanitize_for_json(v)
        return out
    # list/tuple/set -> list
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]
    # fallback: attempt JSON, else safe string
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return _safe_repr(obj)


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
        if verb in ('help', 'h', '?'):
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
                    'list_loaders': 'List registered dataloader names in the ledger',
                    'list_uids': 'List data sample UIDs. Syntax: list_uids [loader_name] [--discarded] [--limit N]',
                    'dump': 'Return a sanitized dump of the ledger contents',
                    'operate': 'Edit model architecture. Syntax: operate [<model_name>] <op_type:int> <layer_id:int> <nb|[list]>',
                    'plot_model': 'Show ASCII tree of model architecture. Syntax: plot_model [<model_name>]',
                    'discard': 'Discard data samples. Syntax: discard <uid> [uid2 ...] [--loader loader_name]',
                    'undiscard': 'Un-discard data samples. Syntax: undiscard <uid> [uid2 ...] [--loader loader_name]',
                    'add_tag': 'Add tag to data sample. Syntax: add_tag <uid> <tag> [--loader loader_name]',
                    'hp / hyperparams': 'List or show hyperparameters. Syntax: hp -> list, hp <name> -> show',
                    'quit / exit': 'Close the client connection'
                },
                'hyperparams_examples': {
                    'list': 'hp',
                    'show': 'hp fashion_mnist',
                    'set': "set_hp <hp_name?> <key.path> <value>  # e.g. set_hp fashion_mnist data.train_loader.batch_size 32",
                },
                'data_examples': {
                    'list all UIDs': 'list_uids',
                    'list specific loader': 'list_uids train_loader',
                    'list discarded only': 'list_uids --discarded',
                    'discard samples': 'discard sample_001 sample_002',
                    'undiscard samples': 'undiscard sample_001',
                    'add tag': 'add_tag sample_001 difficult',
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
                # Return pretty-printed model with proper line breaks
                # Try str() first (which typically has nice formatting), fallback to repr()
                try:
                    model_str = str(m)
                except Exception:
                    model_str = repr(m)

                # Ensure the string preserves line breaks for console output
                # Split into lines and rejoin to normalize line endings
                lines = model_str.split('\n')
                formatted_plot = '\n'.join(lines)

                return {
                    'ok': True,
                    'model_name': model_name or 'default',
                    'plot': formatted_plot,
                    'line_count': len(lines)
                }
            except Exception as e:
                return {'ok': False, 'error': str(e)}

        if verb == 'list_dataloaders':
            return {'ok': True, 'dataloaders': GLOBAL_LEDGER.list_dataloaders()}

        if verb in ('list_loaders', 'loaders'):
            return {'ok': True, 'loaders': GLOBAL_LEDGER.list_dataloaders()}

        if verb == 'list_optimizers':
            return {'ok': True, 'optimizers': GLOBAL_LEDGER.list_optimizers()}

        if verb in ('list_uids', 'uids', 'samples'):
            # Syntax: list_uids [loader_name] [--discarded] [--limit N]
            loader_name = None
            show_discarded_only = False
            limit = None

            # Parse arguments
            i = 1
            while i < len(parts):
                if parts[i] == '--discarded':
                    show_discarded_only = True
                elif parts[i] == '--limit' and i + 1 < len(parts):
                    try:
                        limit = int(parts[i + 1])
                        i += 1
                    except ValueError:
                        return {'ok': False, 'error': 'Invalid limit value'}
                elif not parts[i].startswith('--'):
                    loader_name = parts[i]
                i += 1

            try:
                loaders_to_check = [loader_name] if loader_name else GLOBAL_LEDGER.list_dataloaders()

                result = {'ok': True, 'uids': {}}

                for lname in loaders_to_check:
                    try:
                        loader = GLOBAL_LEDGER.get_dataloader(lname)
                        # Unwrap proxy
                        if hasattr(loader, 'get') and callable(loader.get):
                            loader = loader.get()

                        if loader is None:
                            continue

                        # Try to get dataset from loader
                        dataset = None
                        if hasattr(loader, 'dataset'):
                            dataset = loader.dataset

                        if dataset is None:
                            continue

                        # Get UIDs and discard status
                        uids_list = []

                        # Try different methods to get UIDs
                        if hasattr(dataset, 'get_sample_uids'):
                            # Method to get all UIDs
                            all_uids = dataset.get_sample_uids()
                        elif hasattr(dataset, 'sample_ids'):
                            all_uids = dataset.sample_ids
                        elif hasattr(dataset, 'uids'):
                            all_uids = dataset.uids
                        elif hasattr(dataset, '__len__'):
                            # Fallback: generate UIDs from indices
                            all_uids = [f"sample_{i:06d}" for i in range(len(dataset))]
                        else:
                            all_uids = []

                        # Check discard status for each UID
                        for uid in all_uids:
                            is_discarded = False

                            # Try to get discard status
                            if hasattr(dataset, 'is_discarded'):
                                try:
                                    is_discarded = dataset.is_discarded(uid)
                                except Exception:
                                    pass
                            elif hasattr(dataset, 'discarded_samples'):
                                is_discarded = uid in dataset.discarded_samples

                            # Filter based on --discarded flag
                            if show_discarded_only and not is_discarded:
                                continue

                            # Get tags if available
                            tags = []
                            if hasattr(dataset, 'get_tags'):
                                try:
                                    tags = dataset.get_tags(uid)
                                except Exception:
                                    pass
                            elif hasattr(dataset, 'sample_tags') and hasattr(dataset.sample_tags, 'get'):
                                tags = dataset.sample_tags.get(uid, [])

                            uids_list.append({
                                'uid': uid,
                                'discarded': is_discarded,
                                'tags': tags
                            })

                            # Apply limit
                            if limit and len(uids_list) >= limit:
                                break

                        result['uids'][lname] = uids_list

                    except Exception as e:
                        result['uids'][lname] = {'error': str(e)}

                return result

            except Exception as e:
                return {'ok': False, 'error': str(e)}

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

        if verb == 'dump' or verb == 'd':
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

        if verb in ('discard', 'undiscard'):
            # Syntax: discard <uid> [uid2 ...] [--loader loader_name]
            # Syntax: undiscard <uid> [uid2 ...] [--loader loader_name]

            if len(parts) < 2:
                return {'ok': False, 'error': f'usage: {verb} <uid> [uid2 ...] [--loader loader_name]'}

            loader_name = None
            uids = []

            # Parse arguments
            i = 1
            while i < len(parts):
                if parts[i] == '--loader' and i + 1 < len(parts):
                    loader_name = parts[i + 1]
                    i += 2
                else:
                    uids.append(parts[i])
                    i += 1

            if not uids:
                return {'ok': False, 'error': 'No UIDs specified'}

            discard_status = 1 if verb == 'discard' else 0

            try:
                # Get loader(s)
                if loader_name:
                    loaders_to_update = {loader_name: GLOBAL_LEDGER.get_dataloader(loader_name)}
                else:
                    # Try all loaders
                    loader_names = GLOBAL_LEDGER.list_dataloaders()
                    loaders_to_update = {name: GLOBAL_LEDGER.get_dataloader(name) for name in loader_names}

                results = {'ok': True, 'updated': {}, 'errors': {}}

                for lname, loader in loaders_to_update.items():
                    # Unwrap proxy
                    if hasattr(loader, 'get') and callable(loader.get):
                        loader = loader.get()

                    if loader is None:
                        continue

                    # Get dataset
                    dataset = getattr(loader, 'dataset', None) if loader else None

                    if dataset is None:
                        continue

                    updated_uids = []

                    for uid in uids:
                        try:
                            # Try different methods to set discard status
                            if hasattr(dataset, 'set_discard'):
                                dataset.set_discard(uid, discard_status)
                                updated_uids.append(uid)
                            elif hasattr(dataset, 'discard_sample'):
                                if discard_status == 1:
                                    dataset.discard_sample(uid)
                                else:
                                    # undiscard
                                    if hasattr(dataset, 'undiscard_sample'):
                                        dataset.undiscard_sample(uid)
                                updated_uids.append(uid)
                            elif hasattr(dataset, 'discarded_samples'):
                                # Direct set manipulation
                                if discard_status == 1:
                                    dataset.discarded_samples.add(uid)
                                else:
                                    dataset.discarded_samples.discard(uid)
                                updated_uids.append(uid)
                            else:
                                results['errors'][uid] = f'No discard method available in {lname}'
                        except Exception as e:
                            results['errors'][uid] = str(e)

                    if updated_uids:
                        results['updated'][lname] = updated_uids

                return results

            except Exception as e:
                return {'ok': False, 'error': str(e)}

        if verb in ('add_tag', 'tag'):
            # Syntax: add_tag <uid> <tag> [--loader loader_name]

            if len(parts) < 3:
                return {'ok': False, 'error': 'usage: add_tag <uid> <tag> [--loader loader_name]'}

            uid = parts[1]
            tag = parts[2]
            loader_name = None

            # Parse optional --loader argument
            if len(parts) >= 5 and parts[3] == '--loader':
                loader_name = parts[4]

            try:
                # Get loader(s)
                if loader_name:
                    loaders_to_update = {loader_name: GLOBAL_LEDGER.get_dataloader(loader_name)}
                else:
                    # Try all loaders
                    loader_names = GLOBAL_LEDGER.list_dataloaders()
                    loaders_to_update = {name: GLOBAL_LEDGER.get_dataloader(name) for name in loader_names}

                results = {'ok': True, 'updated': {}, 'errors': {}}

                for lname, loader in loaders_to_update.items():
                    # Unwrap proxy
                    if hasattr(loader, 'get') and callable(loader.get):
                        loader = loader.get()

                    if loader is None:
                        continue

                    # Get dataset
                    dataset = getattr(loader, 'dataset', None) if loader else None

                    if dataset is None:
                        continue

                    try:
                        # Try different methods to add tags
                        if hasattr(dataset, 'add_tag'):
                            dataset.add_tag(uid, tag)
                            results['updated'][lname] = f'Added tag "{tag}" to {uid}'
                        elif hasattr(dataset, 'sample_tags'):
                            # Direct dict manipulation
                            if uid not in dataset.sample_tags:
                                dataset.sample_tags[uid] = []
                            if tag not in dataset.sample_tags[uid]:
                                dataset.sample_tags[uid].append(tag)
                            results['updated'][lname] = f'Added tag "{tag}" to {uid}'
                        else:
                            results['errors'][lname] = 'No tag method available'
                    except Exception as e:
                        results['errors'][lname] = str(e)

                if not results['updated']:
                    return {'ok': False, 'error': 'Could not add tag to any loader'}

                return results

            except Exception as e:
                return {'ok': False, 'error': str(e)}

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


def _server_loop_sock(srv: socket.socket):
    """Serve on an already bound/listening socket (avoids rebind race)."""
    global _server_sock, _server_port, _server_host
    try:
        _server_sock = srv
        _server_host = srv.getsockname()[0]
        _server_port = srv.getsockname()[1]
        while True:
            conn, addr = srv.accept()
            t = threading.Thread(target=_client_handler, args=(conn, addr), name='WL-CLI_serving_loop', daemon=True)
            t.start()
    finally:
        try:
            srv.close()
        except Exception:
            pass


def cli_serve(cli_host: str = 'localhost', cli_port: int = 0, *, spawn_client: bool = True, **_):
    """
        Start the CLI server and optionally open a client in a new console.
        This CLI now operates on objects registered in the global ledger only.

        Args:
            cli_host: Host to bind the server to (default: localhost).
    """

    global _server_thread, _server_port, _server_host
    cli_host = os.environ.get('CLI_HOST', cli_host)
    cli_port = int(os.environ.get('CLI_PORT', cli_port))

    if _server_thread is not None and _server_thread.is_alive():
        return {'ok': False, 'error': 'server_already_running'}

    # start server thread on a pre-bound socket to avoid races
    # Try binding to the requested port, and if it fails (port in use),
    # automatically try up to 10 alternative ports
    srv = None
    last_error = None
    max_attempts = 10
    
    for attempt in range(max_attempts):
        try_port = cli_port + attempt
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # On Windows, try SO_EXCLUSIVEADDRUSE but don't fail if it's not supported
            if os.name == 'nt':
                try:
                    srv.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                except (OSError, AttributeError):
                    # SO_EXCLUSIVEADDRUSE not supported or not available on this system
                    pass
            srv.bind((cli_host, try_port))
            srv.listen(5)
            actual_port = srv.getsockname()[1]
            if attempt > 0:
                logger.warning(f"cli_port_changed: Original port {cli_port} unavailable, using port {actual_port}")
            break
        except (OSError, PermissionError) as e:
            last_error = e
            if srv is not None:
                try:
                    srv.close()
                except Exception:
                    pass
                srv = None
            if attempt < max_attempts - 1:
                continue  # Try next port
            else:
                # All attempts failed
                logger.exception("cli_bind_failed_all_attempts")
                return {'ok': False, 'error': f'bind_failed after {max_attempts} attempts. Last error: {e}. Port {cli_port} may be in use or require admin privileges.'}
    
    if srv is None:
        return {'ok': False, 'error': f'bind_failed: {last_error}'}

    _server_thread = threading.Thread(
        target=_server_loop_sock,
        args=(srv,),
        daemon=True,
        name='WL-CLI_serving_loop',
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
