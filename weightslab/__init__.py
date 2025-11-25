"""weightslab package

Expose commonly used helpers at package level so users can do::

	import weightslab as wl
	wl.watch_or_edit(...)

This file re-exports selected symbols from `weightslab.src`.
"""
from .src import watch_or_edit, serve
from .art import _BANNER

try:
	print(_BANNER)
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
	"__version__",
	"__license__",
    "__author__",
    "__maintainer__",
    "__credits__"
]
