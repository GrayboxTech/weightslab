from typing import Any

from threading import Event, RLock, Lock
import threading
import time
import logging

from weightslab.backend.ledgers import get_hyperparams, set_hyperparam, resolve_hp_name, get_checkpoint_manager, get_optimizers, get_optimizer, register_hyperparams
from weightslab.components.tracking import TrackingMode


# Module-level logger
logger = logging.getLogger(__name__)
# Global locks
weightslab_rlock = RLock()
weightslab_lock = Lock()


class PauseController:
    """
        Shared between model (reader: wait) and control thread (writer: pause/resume).
    """
    def __init__(self):
        self._event = Event()
        self._event.clear()

        # Get checkpoint manager instance
        self.checkpoint_manager = None

        # Get the proxy that wraps our dict
        self.hyperparams = get_hyperparams()

    def wait_if_paused(self):
        # Called from main thread / model forward. Blocks if paused.
        self._event.wait()   # releases GIL while waiting

    def pause(self):
        self._event.clear()
        self.hyperparams['is_training'] = False
        logger.info('\nTraining paused.')

    def resume(self):
        # On resume, first dump any pending changes to checkpoint manager
        if self.checkpoint_manager is None:
            self.checkpoint_manager = get_checkpoint_manager()
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.update_experiment_hash(firsttime=True)
            self.checkpoint_manager.dump_pending_changes()

        # Then resume execution
        if self._is_hash_computed():
            self._event.set()
            self.hyperparams['is_training'] = True
            logger.info(f'\nTraining resumed as modules hashes have been computed: {self.checkpoint_manager.hash_by_module}.')
            return True
        else:
            logger.warning(f'Cannot resume training: experiment hash not computed yet for every modules {self.checkpoint_manager.hash_by_module}.')
            return False

    def is_paused(self):
        return not self._event.is_set()

    def _get_checkpoint_manager(self):
        if self.checkpoint_manager is None:
            self.checkpoint_manager = get_checkpoint_manager()

    def _is_hash_computed(self):
        self._get_checkpoint_manager()
        if self.checkpoint_manager is None:
            return False
        fl = self.checkpoint_manager.hash_by_module[0] != "00000000" and self.checkpoint_manager.hash_by_module[1] != "00000000" and self.checkpoint_manager.hash_by_module[2] != "00000000"

        return fl

# Global pause controller instance
pause_controller = PauseController()


class OpContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self):
        self.op_guard = weightslab_lock
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

op_context = OpContext()


class GuardContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self, for_training: bool):
        self.for_training = for_training
        self.architecture_guard = weightslab_rlock
        self.model = None

    def __enter__(self):
        """
        Executed upon entering the 'with' block. Sets the model to training mode.
        """
        pause_controller.wait_if_paused()
        self.architecture_guard.__enter__()

        # The exact logic requested by the user:
        if self.model is not None:
            if self.for_training:
                self.model.set_tracking_mode(TrackingMode.TRAIN)
            else:
                self.model.set_tracking_mode(TrackingMode.EVAL)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """
        Executed upon exiting the 'with' block (after user code runs).
        Reverts the model state.
        """

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

_pause_sync_thread_started = False
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
                controller_running = not controller_paused

                # Drive controller from ledger when ledger explicitly sets the flag
                if isinstance(hp_is_training, bool):
                    if controller_paused and hp_is_training:
                        resumed = pause_controller.resume()
                        firstresume = False if resumed else True
                    elif controller_running and not hp_is_training:
                        pause_controller.pause()

                # Re-evaluate controller state after potential changes
                controller_paused = pause_controller.is_paused()

                # Propagate controller state back to ledger if it differs
                if controller_paused and not firstresume:
                    try:
                        hp['is_training'] = False
                    except Exception:
                        set_hyperparam(name, 'is_training', False)

        except Exception as e:
            # swallow to keep thread alive
            logger.debug(f"Exception in pause-hp sync loop: {e}")
            pass

        time.sleep(poll_interval)

# Start sync thread once at module import
if not _pause_sync_thread_started:
    _pause_sync_thread_started = True
    t = threading.Thread(target=_pause_hp_sync_loop, name='WL-HP_Sync_Loop', daemon=True)
    t.start()
