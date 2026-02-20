"""weightslab package

Expose commonly used helpers at package level so users can do::

    import weightslab as wl
    wl.watch_or_edit(...)

This file re-exports selected symbols from `weightslab.src`.
"""
from .src import watch_or_edit, serve, keep_serving, save_signals, signal, compute_signals
from .art import _BANNER
from .utils.logs import setup_logging

import os
import logging
import threading

from .src import watch_or_edit, serve, keep_serving, save_signals, tag_samples, discard_samples, get_samples_by_tag, get_discarded_samples
from .art import _BANNER
from .utils.logs import setup_logging, set_log_directory


# Change the name of the current (main) thread
threading.current_thread().name = "WL-MainThread"

# Track if initialization has already happened
_initialized = False

# Auto-initialize logging if not already configured
if not _initialized:
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

	# Setup and use logger
	logger = logging.getLogger(__name__)
	logger.info(f"WeightsLab package initialized - Log level: {log_level}, Log to file: {log_to_file}")
	logger.info(_BANNER)
	
	_initialized = True

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
    "signal",
    "compute_signals",
    "set_log_directory",
	"tag_samples", 
	"discard_samples", 
	"get_samples_by_tag", 
	"get_discarded_samples",

    "_BANNER__",
    "__version__",
    "__license__",
    "__author__",
    "__maintainer__",
    "__credits__"
]
