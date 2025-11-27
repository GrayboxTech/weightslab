"""weightslab package

Expose commonly used helpers at package level so users can do::

	import weightslab as wl
	wl.watch_or_edit(...)

This file re-exports selected symbols from `weightslab.src`.
"""
from .src import watch_or_edit, serve
from .art import _BANNER
from .utils.logs import setup_logging

logger = None
try:
	import logging
	import os
	
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
	
	# Only setup if no handlers exist yet (avoid duplicate setup)
	if not logging.getLogger().handlers:
		setup_logging(log_level, log_to_file=log_to_file)
	
	logger = logging.getLogger(__name__)
	logger.info(_BANNER)
except Exception:
	pass

__version__ = "0.0.0"
__author__ = 'Alexandru-Andrei ROTARY'
__maintainer__ = 'Guillaume PELLUET'
__credits__ = 'GrayBox'
__license__ = 'BSD 2-clause'

__all__ = [
	"watch_or_edit",
	"serve",
    "_BANNER",
	"logger",
	"__version__",
	"__license__",
    "__author__",
    "__maintainer__",
    "__credits__"
]
