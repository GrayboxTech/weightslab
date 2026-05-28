import json
import csv
import os
import threading
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """Immutable audit event structure."""
    timestamp: str  # ISO format string
    action_type: str
    status: str  # "success" or "failed"
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AuditLogger:
    """
    Thread-safe audit logger that writes events to JSON or CSV files.
    Events are persisted to existing files (or created if missing).
    Output format is configurable via AUDIT_LOG_FORMAT environment variable.
    Set format to "none" to disable audit logging entirely.

    Features:
    - Persistent: Appends to existing audit logs when restarting experiments
    - Reverse chronological: Recent events appear first (newest at top)
    - Buffered writes: Events are buffered and flushed periodically to reduce I/O
    """

    # Valid output formats (including "none" to disable)
    VALID_FORMATS = ("json", "csv", "none")

    # Buffer size before flushing to disk (similar to logger chunking approach)
    DEFAULT_BUFFER_SIZE = 50

    def __init__(
        self,
        root_log_dir: str,
        experiment_name: str = "default",
        format: Optional[Literal["json", "csv", "none"]] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):
        """
        Initialize audit logger.

        Args:
            root_log_dir: Directory where audit logs will be stored
            experiment_name: Name of the experiment (for context, not used in filename)
            format: Output format ("json", "csv", or "none" to disable).
                   If None, uses AUDIT_LOG_FORMAT environment variable.
                   Defaults to "json" if not specified.
            buffer_size: Number of events to buffer before flushing to disk (default: 50)
        """
        self.root_log_dir = Path(root_log_dir)
        self.experiment_name = experiment_name

        # Ensure directory exists
        self.root_log_dir.mkdir(parents=True, exist_ok=True)

        # Determine output format from parameter, environment variable, or default
        if format is None:
            format = os.getenv("AUDIT_LOG_FORMAT", "json").lower().strip()

        if format not in self.VALID_FORMATS:
            logger.warning(
                f"[AuditLogger] Invalid format '{format}', using 'json'. "
                f"Valid formats: json, csv, none (to disable logging)"
            )
            format = "json"

        self.format = format
        self.json_path = self.root_log_dir / "audit_log.json"
        self.csv_path = self.root_log_dir / "audit_log.csv"
        self.buffer_size = buffer_size

        # Thread lock for file operations
        self._lock = threading.Lock()

        # Event buffer for batching writes (reduces I/O similar to logger chunking)
        self._event_buffer = []

        status = "disabled" if format == "none" else f"enabled ({format} format)"
        logger.debug(
            f"[AuditLogger] Initialized for experiment '{experiment_name}' "
            f"at {self.root_log_dir} - audit logging {status}"
        )

    def log_event(
        self,
        action_type: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log an audit event with before/after details.

        Args:
            action_type: Type of action (e.g., "hp_change", "tag_add", etc.)
            status: "success" or "failed"
            details: Dict containing what changed (before/after values, affected items)
            error: Error message if status == "failed"
        """
        # Skip logging if disabled
        if self.format == "none":
            return

        # Create event with ISO timestamp
        timestamp = datetime.utcnow().isoformat(timespec='microseconds') + 'Z'
        event = AuditEvent(
            timestamp=timestamp,
            action_type=action_type,
            status=status,
            details=details or {},
            error=error,
        )

        # Add to buffer
        with self._lock:
            try:
                self._event_buffer.append(event)

                # Flush buffer if it reaches threshold
                if len(self._event_buffer) >= self.buffer_size:
                    self._flush_buffer()

                logger.debug(
                    f"[AuditLogger] Logged {action_type} ({status}) - buffered "
                    f"({len(self._event_buffer)}/{self.buffer_size})"
                )
            except Exception as e:
                logger.error(
                    f"[AuditLogger] Failed to log event: {action_type} - {e}",
                    exc_info=True
                )

    def _flush_buffer(self) -> None:
        """Flush buffered events to disk. Must be called within lock."""
        if not self._event_buffer:
            return

        try:
            if self.format == "json":
                self._flush_to_json()
            elif self.format == "csv":
                self._flush_to_csv()

            logger.debug(f"[AuditLogger] Flushed {len(self._event_buffer)} events to disk")
            self._event_buffer.clear()
        except Exception as e:
            logger.error(f"[AuditLogger] Failed to flush buffer: {e}", exc_info=True)

    def flush(self) -> None:
        """Manually flush any pending events to disk."""
        with self._lock:
            self._flush_buffer()

    def _flush_to_json(self) -> None:
        """Flush buffered events to JSON log file in reverse chronological order (newest first)."""
        if not self._event_buffer:
            return

        # Read existing events (for persistence when restarting)
        events = []
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        events = json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"[AuditLogger] Could not read existing JSON log: {e}. "
                    f"Starting fresh."
                )
                events = []

        # Prepend new events in reverse order (latest in buffer is most recent)
        new_events = [asdict(event) for event in reversed(self._event_buffer)]
        events = new_events + events

        # Write back to file
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, default=str)

    def _flush_to_csv(self) -> None:
        """Flush buffered events to CSV log file (appends to end, newest rows at bottom)."""
        if not self._event_buffer:
            return

        fieldnames = ['timestamp', 'action_type', 'status', 'details', 'error']

        # Check if file exists and has content
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        # Append buffered events to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header only if file is new
            if not file_exists:
                writer.writeheader()

            for event in self._event_buffer:
                details_json = json.dumps(event.details) if event.details else ''
                row = {
                    'timestamp': event.timestamp,
                    'action_type': event.action_type,
                    'status': event.status,
                    'details': details_json,
                    'error': event.error or '',
                }
                writer.writerow(row)

    def get_log_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the audit log.

        Returns:
            Dict with counts by action_type and status
        """
        summary = {
            'total_events': 0,
            'by_action_type': {},
            'by_status': {},
        }

        if not self.json_path.exists():
            return summary

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                events = json.load(f)
                summary['total_events'] = len(events)

                for event in events:
                    action = event.get('action_type', 'unknown')
                    status = event.get('status', 'unknown')

                    summary['by_action_type'][action] = \
                        summary['by_action_type'].get(action, 0) + 1
                    summary['by_status'][status] = \
                        summary['by_status'].get(status, 0) + 1
        except Exception as e:
            logger.warning(f"[AuditLogger] Could not read log summary: {e}")

        return summary
