"""weightslab package

Expose commonly used helpers at package level so users can do::

    import weightslab as wl
    wl.watch_or_edit(...)

This file re-exports selected symbols from `weightslab.src`.
"""
import os
import logging
import threading

from .src import watch_or_edit, serve, keep_serving, save_signals, save_group_signals, tag_samples, discard_samples, get_samples_by_tag, get_discarded_samples, signal, eval_fn, compute_signals, SignalContext, clear_all, run_pending_evaluation, trigger_pending_evaluation_async
from .art import _BANNER
from .utils.logs import setup_logging, set_log_directory
from .utils.tools import seed_everything
from .components.global_monitoring import guard_training_context, guard_testing_context

# If you already have other top-level exports, keep them.
# This snippet ensures __version__ is available even when setuptools_scm hasn't written the file yet.


# Change the name of the current (main) thread
threading.current_thread().name = "WL-MainThread"

# Track if initialization has already happened

# Auto-initialize logging if not already configured
if os.environ.get('WEIGHTSLAB_initialized', 'false').lower() == 'false':
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

	# Initialize secure certs and auth tokens if not disabled
	if os.environ.get('WEIGHTSLAB_SKIP_SECURE_INIT', 'false').lower() != 'true':
		try:
			from weightslab.security import CertAuthManager

			enable_auth = os.environ.get('WL_ENABLE_GRPC_AUTH_TOKEN', 'true').lower() == 'true'
			manager = CertAuthManager.from_env_or_default(enable_auth=enable_auth)

			success, msg = manager.initialize()
			if success:
				logger.debug(f"Secure initialization: {msg}")
			else:
				logger.warning(f"Secure initialization warning: {msg}")
		except Exception as e:
			logger.debug(f"Secure initialization skipped: {e}")

	os.environ['WEIGHTSLAB_initialized'] = 'true'  # Ensure WL init once

# Get Package Metadata
try:
    # setuptools_scm will write weightslab/_version.py during build
    from ._version import __version__  # type: ignore
except Exception:
    # Fallback when developing locally or before build; keeps behavior stable.
	from datetime import datetime
	__version__ = datetime.utcnow().strftime("%Y%m%d%H%M%S")
__author__ = 'Alexandru-Andrei ROTARY'
__maintainer__ = 'Guillaume PELLUET'
__credits__ = 'GrayBx'
__license__ = 'BSD 2-clause'
__all__ = [
    "watch_or_edit",
    "serve",
    "keep_serving",
    "save_signals",
    "save_group_signals",
    "signal",
    "compute_signals",
    "set_log_directory",
	  "tag_samples",
  	"discard_samples",
  	"get_samples_by_tag",
    "get_discarded_samples",
    "SignalContext",
    "clear_all",
  	"seed_everything",
    "run_pending_evaluation",
	  "eval_fn",
	  "trigger_pending_evaluation_async",
  	"guard_training_context",
    "guard_testing_context",

    "_BANNER",
    "__version__",
    "__license__",
    "__author__",
    "__maintainer__",
    "__credits__"
]
