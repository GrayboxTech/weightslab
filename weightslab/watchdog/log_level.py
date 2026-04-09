"""Custom WATCHDOG log level for weightslab.

Sits between WARNING (30) and ERROR (40) so watchdog events are visible
without drowning out actual errors, and can be filtered independently.

After this module is imported, any logger supports:

    logger.watchdog("Lock '%s' held for %.1fs", name, duration)
"""

import logging

# Numeric level between WARNING (30) and ERROR (40)
WATCHDOG: int = 35

logging.addLevelName(WATCHDOG, "WATCHDOG")


def _watchdog(self: logging.Logger, message: str, *args, **kwargs) -> None:
    if self.isEnabledFor(WATCHDOG):
        self._log(WATCHDOG, message, args, **kwargs)


# Patch Logger class once so every logger instance gets the method
logging.Logger.watchdog = _watchdog  # type: ignore[attr-defined]
