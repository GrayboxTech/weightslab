"""weightslab package

Expose commonly used helpers at package level so users can do::

    import weightslab as wl
    wl.watch_or_edit(...)

This file re-exports selected symbols from `weightslab.src`.
"""
from .src import watch_or_edit, serve, keep_serving, save_signals
from .art import _BANNER
from .utils.logs import setup_logging

import os
import threading

# Change the name of the current (main) thread
threading.current_thread().name = "WL-MainThread"

print(threading.current_thread().name)  # Output: MyMainThread

# Auto-initialize logging if not already configured
# Check for environment variable to control log level
log_level = os.getenv(
	'WEIGHTSLAB_LOG_LEVEL',
	'INFO'
)
log_to_file = os.getenv(
	'WEIGHTSLAB_LOG_TO_FILE',
	'true'
).lower() == 'true'

# Initialize logging (ensure console + file handlers are configured).
# setup_logging resets handlers, so it's safe to call here and guarantees
# both a console StreamHandler and a FileHandler (when requested).
setup_logging(log_level, log_to_file=log_to_file)

# Get Package Metadata
__version__ = "0.0.0"
__author__ = 'Alexandru-Andrei ROTARY'
__maintainer__ = 'Guillaume PELLUET'
__credits__ = 'GrayBox'
__license__ = 'BSD 2-clause'

__all__ = [
    "watch_or_edit",
    "serve",
    "keep_serving",
    "save_signals",

    "_BANNER",

    "__version__",
    "__license__",
    "__author__",
    "__maintainer__",
    "__credits__"
]
