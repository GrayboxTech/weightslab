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
    Thread-safe audit logger that writes events immediately to JSON or CSV files.
    Events are persisted to existing files (or created if missing).
    Output format is configurable via AUDIT_LOG_FORMAT environment variable.
    Set format to "none" to disable audit logging entirely.

    Features:
    - Persistent: Appends to existing audit logs when restarting experiments
    - Reverse chronological: Recent events appear first (newest at top)
    - Immediate writes: Events written to disk immediately after logging (no data loss on crash)
    """

    # Valid output formats (including "none" to disable)
    VALID_FORMATS = ("json", "csv", "none")

    def __init__(
        self,
        root_log_dir: str,
        experiment_name: str = "default",
        format: Optional[Literal["json", "csv", "none"]] = None,
    ):
        """
        Initialize audit logger.

        Args:
            root_log_dir: Directory where audit logs will be stored
            experiment_name: Name of the experiment (for context, not used in filename)
            format: Output format ("json", "csv", or "none" to disable).
                   If None, uses AUDIT_LOG_FORMAT environment variable.
                   Defaults to "json" if not specified.
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

        # Thread lock for file operations
        self._lock = threading.Lock()

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
        Events are written immediately to disk (no buffering).

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

        # Write immediately to disk
        with self._lock:
            try:
                if self.format == "json":
                    self._write_json(event)
                elif self.format == "csv":
                    self._write_csv(event)

                logger.debug(
                    f"[AuditLogger] Logged {action_type} ({status})"
                )
            except Exception as e:
                logger.error(
                    f"[AuditLogger] Failed to log event: {action_type} - {e}",
                    exc_info=True
                )

    def _write_json(self, event: AuditEvent) -> None:
        """Write a single event to JSON log file in reverse chronological order (newest first)."""
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

        # Prepend new event (reverse chronological: newest first)
        new_event = asdict(event)
        events = [new_event] + events

        # Write back to file
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=2, default=str)

    def _write_csv(self, event: AuditEvent) -> None:
        """Append a single event to CSV log file."""
        fieldnames = ['timestamp', 'action_type', 'status', 'details', 'error']

        # Check if file exists and has content
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        # Append event to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            # Write header only if file is new
            if not file_exists:
                writer.writeheader()

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
