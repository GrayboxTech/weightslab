from typing import Any

from threading import Event, RLock

from weightslab.components.tracking import TrackingMode


class GuardContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self, for_training: bool):
        self.for_training = for_training
        self.architecture_guard = RLock()
        self.model = None

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

        # If exc_type is not None, an exception occurred in the block.
        # Returning False (default) allows the exception to propagate.
        return False 


class PauseController:
    """
        Shared between model (reader: wait) and control thread (writer: pause/resume).
    """
    def __init__(self):
        self._event = Event()
        self._event.set()  # start in 'running' state

    def wait_if_paused(self):
        # Called from main thread / model forward. Blocks if paused.
        self._event.wait()   # releases GIL while waiting

    def pause(self):
        self._event.clear()

    def resume(self):
        self._event.set()

    def is_paused(self):
        return not self._event.is_set()


# Define Global Object here
guard_training_context = GuardContext(for_training=True)
pause_controller = PauseController()
