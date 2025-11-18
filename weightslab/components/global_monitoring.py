import torch as th
import torch.nn.functional as F

from typing import Any
from copy import deepcopy

from threading import Event

from weightslab.components.tracking import TrackingMode


class GuardContext:
    """
    The actual context manager class that handles __enter__ and __exit__.
    It holds a reference to the outer WeightsLab instance.
    """
    def __init__(self, weights_lab_instance, for_training: bool):
        self.wl = weights_lab_instance
        self.for_training = for_training
        
    def __enter__(self):
        """
        Executed upon entering the 'with' block. Sets the model to training mode.
        """

        self.wl.architecture_guard.__enter__()

        # The exact logic requested by the user:
        if self.for_training:
            self.wl.model.set_tracking_mode(TrackingMode.TRAIN)
            self.wl.model.train()
        else:
            self.wl.model.set_tracking_mode(TrackingMode.EVAL)
            self.wl.model.eval()

        # Optional: You can return the current instance (self.wl) or any resource needed inside the block
        return self.wl

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        """
        Executed upon exiting the 'with' block (after user code runs). 
        Reverts the model state.
        """

        self.wl.architecture_guard.__exit__(exc_type, exc_value, traceback)

        # Revert the model state back to evaluation/non-tracking
        if self.for_training:
            with self.wl.lock:
                self.wl.training_steps_to_do -= 1
                self.wl.for_training = self.wl.training_steps_to_do > 0

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

    def resume(self):
        self._event.set()

    def is_paused(self):
        return not self._event.is_set()


# Define Global Object here
pause_controller = PauseController()
