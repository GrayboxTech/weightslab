from typing import Any

from threading import Event, RLock, Lock
import threading
import time

from weightslab.ledgers import list_hyperparams, get_hyperparams, set_hyperparam

from weightslab.components.tracking import TrackingMode


class OpContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self):
        self.op_guard = Lock()
        self.model = None

    def __enter__(self):
        """
        Executed upon entering the 'with' block. Sets the model to training mode.
        """

        self.op_guard.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """
        Executed upon exiting the 'with' block (after user code runs). 
        Reverts the model state.
        """

        self.op_guard.__exit__(exc_type, exc_value, traceback)

        # If exc_type is not None, an exception occurred in the block.
        # Returning False (default) allows the exception to propagate.
        return False 


class GuardContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self, for_training: bool, op_context: OpContext = None):
        self.for_training = for_training
        self.architecture_guard = RLock()
        self.model = None
        self.op_context = op_context

    def __enter__(self):
        """
        Executed upon entering the 'with' block. Sets the model to training mode.
        """

        self.architecture_guard.__enter__()

        # The exact logic requested by the user:
        if self.for_training:
            self.model.set_tracking_mode(TrackingMode.TRAIN)
            self.model.train()
        else:
            self.model.set_tracking_mode(TrackingMode.EVAL)
            self.model.eval()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """
        Executed upon exiting the 'with' block (after user code runs). 
        Reverts the model state.
        """

        self.architecture_guard.__exit__(exc_type, exc_value, traceback)

        # Use provided op_context if present, otherwise fall back to module-level
        ctx = self.op_context if getattr(self, 'op_context', None) is not None else op_context
        with ctx:
            # decrement training steps and store result in ledgered hyperparams
            try:
                # resolve a sensible hyperparam set name (reuse helper in this module)
                name = _resolve_hp_name()
                if name is not None:
                    try:
                        hp_handle = get_hyperparams(name)
                    except Exception:
                        hp_handle = None

                    try:
                        if hp_handle is None:
                            raise RuntimeError('no hyperparams')
                        # unwrap proxy if present
                        if hasattr(hp_handle, 'get') and not isinstance(hp_handle, dict):
                            hp = hp_handle.get()
                        else:
                            hp = hp_handle

                        if not isinstance(hp, dict):
                            raise RuntimeError('hyperparams not a dict')

                        cur = hp.get('training_steps_to_do', 0)
                        try:
                            cur_int = int(cur)
                        except Exception:
                            cur_int = 0
                        new = max(0, cur_int - 1)

                        # try ledger API first
                        try:
                            set_hyperparam(name, 'training_steps_to_do', new)
                            set_hyperparam(name, 'is_training', new > 0)
                        except Exception:
                            # best-effort fallback: update dict directly
                            try:
                                hp['training_steps_to_do'] = new
                                hp['is_training'] = new > 0
                            except Exception:
                                pass
                    except Exception:
                        # swallow errors - don't let monitoring break training
                        pass
            except Exception:
                pass

        # If exc_type is not None, an exception occurred in the block.
        # Returning False (default) allows the exception to propagate.
        return False 


class PauseController:
    """
        Shared between model (reader: wait) and control thread (writer: pause/resume).
    """
    def __init__(self):
        self._event = Event()

    def wait_if_paused(self):
        # Called from main thread / model forward. Blocks if paused.
        self._event.wait()   # releases GIL while waiting

    def pause(self):
        self._event.clear()
        set_hyperparam(None, 'is_training', False)

    def resume(self):
        self._event.set()
        set_hyperparam(None, 'is_training', True)

    def is_paused(self):
        return not self._event.is_set()


# Define Global Object here
op_context = OpContext()
guard_training_context = GuardContext(for_training=True)
pause_controller = PauseController()


# Background sync: keep ledger hyperparam `is_training` and pause_controller in sync.
# Behavior:
# - If ledger `is_training` == True and controller is paused -> resume controller.
# - If ledger `is_training` == False and controller is running -> pause controller.
# - If controller is paused/resumed externally, update ledger `is_training` to match.

_pause_sync_thread_started = False

def _resolve_hp_name() -> str | None:
    names = list_hyperparams()
    if 'main' in names:
        return 'main'
    if 'experiment' in names:
        return 'experiment'
    if len(names) == 1:
        return names[0]
    return None

def _pause_hp_sync_loop(poll_interval: float = 0.5):
    last_name = None
    while True:
        try:
            name = _resolve_hp_name()
            if name is None:
                time.sleep(poll_interval)
                continue

            # lazy-resolve hyperparams handle
            try:
                hp_handle = get_hyperparams(name)
            except Exception:
                time.sleep(poll_interval)
                continue

            # unwrap Proxy-like handle if present
            try:
                if hasattr(hp_handle, 'get') and not isinstance(hp_handle, dict):
                    hp = hp_handle.get()
                else:
                    hp = hp_handle
            except Exception:
                hp = None

            if not isinstance(hp, dict):
                time.sleep(poll_interval)
                continue

            hp_is_training = hp.get('is_training')
            controller_paused = pause_controller.is_paused()
            controller_running = not controller_paused

            # Drive controller from ledger when ledger explicitly sets the flag
            if isinstance(hp_is_training, bool):
                if hp_is_training and controller_paused:
                    pause_controller.resume()
                    controller_running = True
                elif not hp_is_training and controller_running:
                    pause_controller.pause()
                    controller_running = False

            # Propagate controller state back to ledger if it differs
            if not isinstance(hp_is_training, bool) or hp_is_training != controller_running:
                try:
                    set_hyperparam(name, 'is_training', controller_running)
                except Exception:
                    # best-effort: update underlying dict directly
                    try:
                        hp['is_training'] = controller_running
                    except Exception:
                        pass

        except Exception:
            # swallow to keep thread alive
            pass

        time.sleep(poll_interval)

# Start sync thread once at module import
if not _pause_sync_thread_started:
    _pause_sync_thread_started = True
    t = threading.Thread(target=_pause_hp_sync_loop, name='pause-hp-sync', daemon=True)
    t.start()
