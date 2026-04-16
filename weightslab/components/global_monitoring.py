import os
from typing import Any, Optional
from enum import Enum
import contextvars

from threading import Event, Lock
import threading
import time
import logging

from weightslab.backend.ledgers import get_hyperparams, set_hyperparam, resolve_hp_name, get_checkpoint_manager
from weightslab.components.tracking import TrackingMode
from weightslab.watchdog.lock_monitor import MonitoredRLock


# Module-level logger
logger = logging.getLogger(__name__)

# Global locks
# weightslab_rlock is a MonitoredRLock so the watchdog can detect when it is
# held too long and raise _WatchdogInterrupt in the holder thread, causing
# any finally/with block to release it cleanly.
weightslab_rlock = MonitoredRLock()

# Timeout for acquiring weightslab_rlock in gRPC handlers.
# Mirrors GRPC_WATCHDOG_STUCK_SECONDS so an RPC that cannot grab the lock fails
# cleanly before the watchdog would flag it as an infinite hang.
# TODO (GP): Now set to -1 (no timeout) to avoid interrupting long-running RPCs that are doing heavy work while holding the lock. We should eventually remove this timeout and rely solely on the watchdog to detect and handle stuck locks, to avoid unintended interruptions.
_GRPC_LOCK_TIMEOUT_S: float = float(os.getenv("GRPC_WATCHDOG_STUCK_SECONDS", "-1"))


def try_acquire_rlock(timeout_s: float = _GRPC_LOCK_TIMEOUT_S) -> bool:
    """Try to acquire weightslab_rlock with a watchdog-aligned timeout.

    Returns True if the lock was acquired (caller must release it).
    Returns False if timed out — caller should abort the RPC instead of hanging.
    """
    return weightslab_rlock.acquire(timeout=timeout_s)


# Context management for training vs testing
class Context(Enum):
    """Enum for current execution context (training or testing)."""
    TRAINING = "training"
    TESTING = "testing"
    UNKNOWN = "unknown"


# Thread-local context variable to track current context
_current_context: contextvars.ContextVar[Context] = contextvars.ContextVar(
    'weightslab_context', default=Context.UNKNOWN
)

# NEW: Track active dataset origin (e.g. 'train_loader')
_active_origin: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'weightslab_origin', default=None
)


def get_current_context() -> Context:
    """Get the current WeightsLab execution context (training or testing)."""
    return _current_context.get()


def set_current_context(context: Context) -> contextvars.Token[Context]:
    """Set the current WeightsLab execution context and return a token for restoration."""
    return _current_context.set(context)


def get_active_origin() -> Optional[str]:
    """Get the current active dataset origin name."""
    return _active_origin.get()


def set_active_origin(origin: Optional[str]) -> contextvars.Token[Optional[str]]:
    """Set the current active dataset origin name."""
    return _active_origin.set(origin)


class PauseController:
    """
        Shared between model (reader: wait) and control thread (writer: pause/resume).
    """
    def __init__(self):
        # Event is used for efficient waiting in the main thread during pause, and for signaling resume.
        self._event = Event()
        self._event.clear()

        # Get checkpoint manager instance
        self.checkpoint_manager = None

        # Get the proxy that wraps our dict
        self.hyperparams = get_hyperparams()

    def wait_if_paused(self, skip_pause: bool = False):
        # Called from main thread / model forward. Blocks if paused.
        # Use timeout to allow signal handlers (Ctrl+C, SIGTERM) to be processed.
        # Also wakes up early when an evaluation is pending/running so the
        # training loop and dataloaders can service evaluation mode.
        while not self._event.wait(timeout=0.5):
            # Timeout occurred – check for evaluation request before looping
            try:
                if skip_pause:
                    # An eval was requested while paused: unblock so the
                    # training loop can reach run_pending_evaluation().
                    return
            except Exception:
                pass

    def pause(self):
        self._event.clear()
        set_hyperparam(key_path='is_training', value=False)
        set_hyperparam(key_path='pause_at_step', value=0)
        logger.info('\nTraining paused.')

    def _resume(self):
        self._event.set()
        set_hyperparam(key_path='is_training', value=True)

    def resume(self, force: bool = False) -> bool:
        hash_by_module = None
        logger.info('\nAttempting to resume training...')

        # On resume, first dump any pending changes to checkpoint manager
        if self.checkpoint_manager == None:
            self.checkpoint_manager = get_checkpoint_manager()
        if self.checkpoint_manager != None:
            self.checkpoint_manager.update_experiment_hash(first_time=True)
            self.checkpoint_manager.save_pending_changes()  # Write pending change to disk
            hash_by_module = self.checkpoint_manager.hash_by_module
        else:
            logger.warning('Cannot access checkpoint manager on resume.')
        logger.info(f'Hashes by module: {hash_by_module}')

        # Then resume execution
        if self.checkpoint_manager == None or self._is_hash_computed() or force:
            logger.info('Resuming training now...')
            self._resume()
            logger.info(f'Hashes by module on resume: {hash_by_module}')
            logger.info(f'\nTraining resumed as modules hashes have been computed: {hash_by_module}.')
            return True
        else:
            logger.warning(f'Cannot resume training: experiment hash not computed yet for every modules {hash_by_module}.')
            return False

    def is_paused(self):
        return not self._event.is_set()

    def _get_checkpoint_manager(self):
        if self.checkpoint_manager is None:
            self.checkpoint_manager = get_checkpoint_manager()

    def _is_hash_computed(self):
        self._get_checkpoint_manager()
        if self.checkpoint_manager == None:
            return False
        fl = self.checkpoint_manager.get_hp_hash() != "00000000" and self.checkpoint_manager.get_model_hash() != "00000000" and self.checkpoint_manager.get_data_hash() != "00000000"

        return fl


# Global pause controller instance
pause_controller = PauseController()


class GuardContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self, for_training: bool):
        self.for_training = for_training
        self.architecture_guard = weightslab_rlock
        self.model = None
        self._context_token = None

    def __enter__(self):
        """
        Executed upon entering the 'with' block. Sets the model to training mode.
        """
        pause_controller.wait_if_paused()
        self.architecture_guard.__enter__()

        # Set the current context for this execution
        context = Context.TRAINING if self.for_training else Context.TESTING
        self._context_token = set_current_context(context)

        # The exact logic requested by the user:
        if self.model is not None:
            # Save current mode to restore on exit
            self._prev_training_mode = getattr(self.model, 'training', True)

            # Check for Audit Mode override
            is_audit = False
            try:
                hp_name = resolve_hp_name()
                hp = get_hyperparams(hp_name)
                if hp and (bool(hp.get('auditorMode')) or bool(hp.get('auditor_mode'))):
                    is_audit = True
            except Exception:
                pass

            if self.for_training and not is_audit:
                self.model.set_tracking_mode(TrackingMode.TRAIN)
                self.model.train()
            elif self.for_training and is_audit:
                # In audit mode: keep TRAIN tracking so current_step increments
                # and the signal logger can flush its buffer on each step change.
                # Weight updates are already blocked by OptimizerInterface.step().
                # We also set the model to eval() mode to freeze BN stats and Dropout.
                self.model.set_tracking_mode(TrackingMode.TRAIN)
                self.model.eval()

                # Throttle logging
                if not hasattr(self, '_last_audit_msg'):
                    self._last_audit_msg = 0
                if time.time() - self._last_audit_msg > 10.0:
                    logger.info("[WeightsLab] Audit Mode active: Model set to eval() (BN stats frozen).")
                    self._last_audit_msg = time.time()
            else:
                self.model.set_tracking_mode(TrackingMode.EVAL)
                self.model.eval()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """
        Executed upon exiting the 'with' block (after user code runs).
        Reverts the model state.
        """
        # Revert the model state
        if self.model is not None and hasattr(self, '_prev_training_mode'):
            self.model.train(self._prev_training_mode)

        # Reset context to unknown
        if self._context_token is not None:
            _current_context.reset(self._context_token)
            self._context_token = None

        if exc_type is RuntimeError:
            logger.debug(f"Suppressing exception: {exc_value} in GuardContext.__exit__")
            self.architecture_guard.__exit__(exc_type, exc_value, traceback)
            return True  # suppress the exception

        self.architecture_guard.__exit__(exc_type, exc_value, traceback)

        return False


# Define Global Object here
guard_training_context = GuardContext(for_training=True)
guard_testing_context = GuardContext(for_training=False)


# Background sync: keep ledger hyperparam `is_training` and pause_controller in sync.
# Behavior:
# - If ledger `is_training` == True and controller is paused -> resume controller.
# - If ledger `is_training` == False and controller is running -> pause controller.
# - If controller is paused/resumed externally, update ledger `is_training` to match.

_enable_pause_sync_thread = os.environ.get('WL_ENABLE_HP_SYNC', True)
_pause_sync_thread_started = bool(_enable_pause_sync_thread) and _enable_pause_sync_thread != '0'
checkpoint_manager = get_checkpoint_manager()

def _pause_hp_sync_loop(poll_interval: float = 3):
    firstresume = True
    while True:
        try:
            name = resolve_hp_name()
            if name is None:
                time.sleep(poll_interval)
                continue

            # lazy-resolve hyperparams proxy
            try:
                hp = get_hyperparams(name)
            except Exception:
                time.sleep(poll_interval)
                continue

            # hp logger issue
            if hp == None:
                logger.warning(f"Hyperparams proxy is None for name {name}. Check if the ledger is properly initialized and the hyperparams are set up. Retrying in {poll_interval} seconds...")

            # Check if hp is dict-like (has required methods) rather than isinstance
            if not hasattr(hp, '__getitem__') or not hasattr(hp, '__setitem__'):
                time.sleep(poll_interval)
                continue

            # Training status from ledger
            try:
                # Use .get() if available (dict or Proxy with dict), otherwise use []
                hp_is_training = hp.get('is_training') if hasattr(hp, 'get') else hp['is_training']
            except (KeyError, TypeError, AttributeError):
                time.sleep(poll_interval)
                continue
            if hp_is_training is not None:
                controller_paused = pause_controller.is_paused()

                # TODO (GP): The logic here is a bit tricky because we want to avoid race conditions where both the controller and the ledger are trying to update each other at the same time. The current approach is:
                # - The controller is the source of truth for the paused state, since it's what actually blocks the training loop. The ledger's `is_training` is a reflection of that state for visibility in the UI and for external control.
                # - On each loop, we check the ledger's `is_training` against the controller's state. If they are out of sync, we update the controller to match the ledger. This allows external changes to the ledger to take effect.
                # - After potentially updating the controller, we check if the controller is paused and if this is not the first resume (to avoid overwriting the ledger state on startup). If the controller is paused but the ledger does
                # not reflect that, we update the ledger to match the controller. This ensures that if the controller is paused externally (e.g. via pause_controller.pause()), the ledger state is updated accordingly.

                # # Drive controller from ledger when ledger explicitly sets the flag
                # controller_running = not controller_paused
                # if isinstance(hp_is_training, bool):
                #     if controller_paused and hp_is_training:
                #         resumed = pause_controller.resume()
                #         firstresume = False if resumed else True
                #     elif controller_running and not hp_is_training:
                #         pause_controller.pause()

                # Re-evaluate controller state after potential changes
                controller_paused = pause_controller.is_paused()

                # Propagate controller state back to ledger if it differs
                if controller_paused and not firstresume:
                    try:
                        set_hyperparam(key_path='is_training', value=False)
                    except Exception:
                        set_hyperparam(key_path='is_training', value=False)

        except Exception as e:
            # swallow to keep thread alive
            logger.debug(f"Exception in pause-hp sync loop: {e}")
            pass

        time.sleep(poll_interval)

def start_hp_sync_thread_event():
    t = threading.Thread(target=_pause_hp_sync_loop, name='WL-HP_Sync_Loop', daemon=True)
    t.start()

# Start sync thread once at module import
if _pause_sync_thread_started:
    _pause_sync_thread_started = False  # already activated
    start_hp_sync_thread_event()
