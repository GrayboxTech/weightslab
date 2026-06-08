import json
import csv
import tempfile
import threading
from pathlib import Path
from datetime import datetime
import time

from weightslab.backend.audit_logger import AuditLogger, AuditEvent


class TestAuditEvent:
    """Test AuditEvent dataclass."""

    def test_audit_event_creation(self):
        """Test creating an AuditEvent with all fields."""
        event = AuditEvent(
            timestamp="2026-05-27T14:30:00.123456Z",
            action_type="hp_change",
            status="success",
            details={"field": "learning_rate", "value": 0.001},
            error=None,
        )
        assert event.timestamp == "2026-05-27T14:30:00.123456Z"
        assert event.action_type == "hp_change"
        assert event.status == "success"
        assert event.details == {"field": "learning_rate", "value": 0.001}
        assert event.error is None

    def test_audit_event_with_error(self):
        """Test creating an AuditEvent with error message."""
        event = AuditEvent(
            timestamp="2026-05-27T14:30:00.123456Z",
            action_type="checkpoint_restore",
            status="failed",
            details={"checkpoint_id": "ckpt_001"},
            error="Checkpoint not found",
        )
        assert event.status == "failed"
        assert event.error == "Checkpoint not found"


class TestAuditLoggerInitialization:
    """Test AuditLogger initialization."""

    def test_logger_initialization(self):
        """Test AuditLogger initialization with valid directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            assert logger.root_log_dir == Path(tmpdir)
            assert logger.experiment_name == "test_experiment"
            assert logger.json_path == Path(tmpdir) / "audit_log.json"
            assert logger.csv_path == Path(tmpdir) / "audit_log.csv"

    def test_logger_creates_directory(self):
        """Test that AuditLogger creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "path"
            assert not nested_dir.exists()
            # Instantiating the logger must create the (nested) directory.
            AuditLogger(str(nested_dir), "exp")
            assert nested_dir.exists()

    def test_logger_with_default_experiment_name(self):
        """Test AuditLogger with default experiment name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir)
            assert logger.experiment_name == "default"

    def test_logger_format_json(self):
        """Test AuditLogger with JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, format="json")
            assert logger.format == "json"

    def test_logger_format_csv(self):
        """Test AuditLogger with CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, format="csv")
            assert logger.format == "csv"

    def test_logger_invalid_format_defaults_to_json(self):
        """Test that invalid format defaults to json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, format="invalid")
            assert logger.format == "json"


class TestAuditLoggerFormat:
    """Test format selection and behavior."""

    def test_json_format_only_writes_json(self):
        """Test that JSON format only creates JSON file, not CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, format="json")
            logger.log_event("test_action", "success", {"data": "value"})

            assert logger.json_path.exists()
            assert not logger.csv_path.exists()

    def test_csv_format_only_writes_csv(self):
        """Test that CSV format only creates CSV file, not JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, format="csv")
            logger.log_event("test_action", "success", {"data": "value"})

            assert logger.csv_path.exists()
            assert not logger.json_path.exists()

    def test_format_from_environment_variable(self, monkeypatch):
        """Test that format can be set via AUDIT_LOG_FORMAT environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test CSV from env var
            monkeypatch.setenv("AUDIT_LOG_FORMAT", "csv")
            logger = AuditLogger(tmpdir)
            assert logger.format == "csv"

    def test_explicit_format_overrides_environment(self, monkeypatch):
        """Test that explicit format parameter overrides environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("AUDIT_LOG_FORMAT", "csv")
            logger = AuditLogger(tmpdir, format="json")
            assert logger.format == "json"

    def test_none_format_disables_logging(self):
        """Test that format='none' disables audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, format="none")
            assert logger.format == "none"

            # Log events but they should be skipped
            logger.log_event("test_action", "success", {"data": "value"})
            logger.log_event("another_action", "success", {"data": "value2"})

            # No files should be created
            assert not logger.json_path.exists()
            assert not logger.csv_path.exists()

    def test_none_format_from_environment_variable(self, monkeypatch):
        """Test that AUDIT_LOG_FORMAT=none disables logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("AUDIT_LOG_FORMAT", "none")
            logger = AuditLogger(tmpdir)
            assert logger.format == "none"

            logger.log_event("test_action", "success", {"data": "value"})

            # No files should be created
            assert not logger.json_path.exists()
            assert not logger.csv_path.exists()

    def test_explicit_format_none_overrides_json_default(self, monkeypatch):
        """Test that explicit format='none' overrides JSON default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("AUDIT_LOG_FORMAT", "json")
            logger = AuditLogger(tmpdir, format="none")
            assert logger.format == "none"

            logger.log_event("test_action", "success", {"data": "value"})

            # No files should be created even though env var is json
            assert not logger.json_path.exists()
            assert not logger.csv_path.exists()


class TestAuditLoggerJSON:
    """Test JSON logging functionality."""

    def test_log_event_creates_json_file(self):
        """Test that logging an event creates the JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            logger.log_event("hp_change", "success", {"param": "learning_rate"})

            assert logger.json_path.exists()
            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert len(events) == 1
            assert events[0]["action_type"] == "hp_change"
            assert events[0]["status"] == "success"

    def test_log_event_json_format(self):
        """Test JSON event format with all fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            details = {"field": "learning_rate", "before": 0.001, "after": 0.0005}
            logger.log_event("hp_change", "success", details)

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            event = events[0]

            assert "timestamp" in event
            assert event["action_type"] == "hp_change"
            assert event["status"] == "success"
            assert event["details"] == details
            assert event["error"] is None

    def test_log_event_appends_to_json(self):
        """Test that logging multiple events appends to JSON in reverse chronological order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")

            logger.log_event("hp_change", "success", {"param": "lr"})
            logger.log_event("pause", "success", {"state": "paused"})
            logger.log_event("resume", "success", {"state": "running"})

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert len(events) == 3
            # Reverse chronological order: newest first
            assert events[0]["action_type"] == "resume"
            assert events[1]["action_type"] == "pause"
            assert events[2]["action_type"] == "hp_change"

    def test_json_timestamp_format(self):
        """Test that JSON timestamps are in ISO 8601 format with microseconds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            logger.log_event("test_action", "success")

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            timestamp = events[0]["timestamp"]

            # Check ISO 8601 format with Z suffix and microseconds
            assert timestamp.endswith('Z')
            assert 'T' in timestamp
            # Check microseconds are present (at least 6 decimal places before Z)
            assert '.' in timestamp


class TestAuditLoggerCSV:
    """Test CSV logging functionality."""

    def test_log_event_creates_csv_file(self):
        """Test that logging an event creates the CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="csv")
            logger.log_event("hp_change", "success", {"param": "learning_rate"})

            assert logger.csv_path.exists()
            with open(logger.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["action_type"] == "hp_change"
            assert rows[0]["status"] == "success"

    def test_csv_headers(self):
        """Test that CSV has correct headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="csv")
            logger.log_event("test_action", "success", {"data": "value"})

            with open(logger.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == ['timestamp', 'action_type', 'status', 'details', 'error']

    def test_csv_details_escaped_as_json(self):
        """Test that details in CSV are properly escaped JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="csv")
            details = {"field": "learning_rate", "before": 0.001, "after": 0.0005}
            logger.log_event("hp_change", "success", details)

            with open(logger.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            row = rows[0]

            # Details should be valid JSON
            details_json = json.loads(row["details"])
            assert details_json == details

    def test_csv_appends_to_file(self):
        """Test that logging multiple events appends to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="csv")

            logger.log_event("hp_change", "success", {"param": "lr"})
            logger.log_event("pause", "success", {"state": "paused"})
            logger.log_event("resume", "success", {"state": "running"})

            with open(logger.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 3
            assert rows[0]["action_type"] == "hp_change"
            assert rows[1]["action_type"] == "pause"
            assert rows[2]["action_type"] == "resume"

    def test_csv_with_error_message(self):
        """Test CSV logging with error messages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="csv")
            logger.log_event(
                "checkpoint_restore",
                "failed",
                {"checkpoint_id": "ckpt_001"},
                error="Checkpoint file not found",
            )

            with open(logger.csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            row = rows[0]

            assert row["status"] == "failed"
            assert row["error"] == "Checkpoint file not found"


class TestAuditLoggerErrorHandling:
    """Test error handling and edge cases."""

    def test_log_event_with_none_details(self):
        """Test logging with None details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            logger.log_event("test_action", "success", details=None)

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert events[0]["details"] == {}

    def test_log_event_with_complex_nested_details(self):
        """Test logging with complex nested data structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="json")
            details = {
                "config": {
                    "metrics": ["accuracy", "f1", "loss"],
                    "dataset_split": "val",
                    "batch_size": 32,
                    "nested": {
                        "level2": {
                            "value": "deep"
                        }
                    }
                },
                "samples_count": 500,
            }
            logger.log_event("evaluation_start", "success", details)

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert events[0]["details"] == details

    def test_log_event_with_special_characters(self):
        """Test logging with special characters in details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="json")
            details = {
                "message": 'Special chars: "quotes", \'apostrophes\', \\backslash',
                "path": "C:\\Users\\test\\file.txt",
                "unicode": "emoji: 🎉 test",
            }
            logger.log_event("test_action", "success", details)

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert events[0]["details"] == details

    def test_log_event_with_empty_details(self):
        """Test logging with empty details dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            logger.log_event("test_action", "success", {})

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert events[0]["details"] == {}


class TestAuditLoggerThreadSafety:
    """Test thread-safe concurrent logging."""

    def test_concurrent_logging(self):
        """Test that concurrent logging works correctly with file locking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment", format="json")
            num_threads = 10
            events_per_thread = 10

            def log_events(thread_id):
                for i in range(events_per_thread):
                    logger.log_event(
                        f"action_{thread_id}",
                        "success",
                        {"thread": thread_id, "iteration": i},
                    )

            threads = [
                threading.Thread(target=log_events, args=(i,))
                for i in range(num_threads)
            ]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Verify all events were logged
            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert len(events) == num_threads * events_per_thread


class TestAuditLoggerSummary:
    """Test log summary statistics."""

    def test_get_log_summary_empty(self):
        """Test summary for empty log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            summary = logger.get_log_summary()

            assert summary["total_events"] == 0
            assert summary["by_action_type"] == {}
            assert summary["by_status"] == {}

    def test_get_log_summary_with_events(self):
        """Test summary with multiple events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")

            logger.log_event("hp_change", "success", {})
            logger.log_event("hp_change", "success", {})
            logger.log_event("pause", "success", {})
            logger.log_event("checkpoint_restore", "failed", {})

            summary = logger.get_log_summary()

            assert summary["total_events"] == 4
            assert summary["by_action_type"]["hp_change"] == 2
            assert summary["by_action_type"]["pause"] == 1
            assert summary["by_action_type"]["checkpoint_restore"] == 1
            assert summary["by_status"]["success"] == 3
            assert summary["by_status"]["failed"] == 1


class TestAuditLoggerPersistence:
    """Test audit log persistence and restart scenarios."""

    def test_persistence_when_restarting(self):
        """Test that audit logs are appended when restarting an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # First run: create some audit logs
            logger1 = AuditLogger(tmpdir, "experiment", format="json")
            logger1.log_event("hp_change", "success", {"lr": 0.001})
            logger1.log_event("tag_add", "success", {"tag": "defect"})

            with open(logger1.json_path) as f:
                events_after_first_run = json.load(f)
            assert len(events_after_first_run) == 2

            # Second run: restart from the same directory
            logger2 = AuditLogger(tmpdir, "experiment", format="json")
            logger2.log_event("pause", "success", {"state": "paused"})

            with open(logger2.json_path) as f:
                events_after_restart = json.load(f)

            # Should have 3 events total (2 from first run + 1 from restart)
            assert len(events_after_restart) == 3
            # Check that original events are still there (newest first)
            assert events_after_restart[0]["action_type"] == "pause"
            assert events_after_restart[1]["action_type"] == "tag_add"
            assert events_after_restart[2]["action_type"] == "hp_change"

    def test_reverse_chronological_order_json(self):
        """Test that JSON events are in reverse chronological order (newest first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test", format="json")

            logger.log_event("hp_change", "success", {"param": "lr"})
            time.sleep(0.01)
            logger.log_event("tag_add", "success", {"tag": "defect"})
            time.sleep(0.01)
            logger.log_event("pause", "success", {"state": "paused"})

            with open(logger.json_path) as f:
                events = json.load(f)

            # Should be in reverse chronological order: newest first
            assert len(events) == 3
            assert events[0]["action_type"] == "pause"  # Most recent
            assert events[1]["action_type"] == "tag_add"
            assert events[2]["action_type"] == "hp_change"  # Oldest

            # Timestamps should be in reverse order
            ts0 = datetime.fromisoformat(events[0]["timestamp"].replace('Z', '+00:00'))
            ts1 = datetime.fromisoformat(events[1]["timestamp"].replace('Z', '+00:00'))
            ts2 = datetime.fromisoformat(events[2]["timestamp"].replace('Z', '+00:00'))
            assert ts0 > ts1 > ts2


    def test_get_log_summary_nonexistent_log(self):
        """Test summary when log file doesn't exist yet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")
            summary = logger.get_log_summary()

            assert summary["total_events"] == 0
            assert summary["by_action_type"] == {}
            assert summary["by_status"] == {}


class TestAuditLoggerRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_hyperparameter_change_scenario(self):
        """Test logging hyperparameter changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "mnist_experiment")

            # User changes learning rate
            logger.log_event(
                "hp_change",
                "success",
                {
                    "changed_params": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                    }
                },
            )

            with open(logger.json_path, 'r') as f:
                events = json.load(f)
            assert events[0]["details"]["changed_params"]["learning_rate"] == 0.001

    def test_data_editing_scenario(self):
        """Test logging data editing operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "image_classification_experiment")

            # User adds tags to samples
            logger.log_event(
                "tag_add",
                "success",
                {
                    "tag_name": "defect",
                    "samples_affected": 5,
                    "sample_ids": ["s1", "s2", "s3", "s4", "s5"],
                    "origins": ["train", "train", "val", "val", "test"],
                },
            )

            # User discards low-quality samples
            logger.log_event(
                "sample_discard",
                "success",
                {
                    "samples_affected": 10,
                    "sample_ids": ["s6", "s7", "s8", "s9", "s10"],
                },
            )

            summary = logger.get_log_summary()
            assert summary["total_events"] == 2
            assert "tag_add" in summary["by_action_type"]
            assert "sample_discard" in summary["by_action_type"]

    def test_training_control_scenario(self):
        """Test logging training state changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "training_experiment")

            logger.log_event("resume", "success", {"trainer_state": "running"})
            time.sleep(0.01)  # Simulate training
            logger.log_event("pause", "success", {"trainer_state": "paused"})
            time.sleep(0.01)
            logger.log_event("resume", "success", {"trainer_state": "running"})

            with open(logger.json_path, 'r') as f:
                events = json.load(f)

            # Verify reverse chronological order with timestamps
            assert len(events) == 3
            assert events[0]["action_type"] == "resume"  # Newest
            assert events[1]["action_type"] == "pause"
            assert events[2]["action_type"] == "resume"  # Oldest

            # Verify timestamps are in reverse chronological order (newest first)
            ts1 = events[0]["timestamp"]
            ts2 = events[1]["timestamp"]
            ts3 = events[2]["timestamp"]
            assert ts1 > ts2 > ts3

    def test_failure_scenario(self):
        """Test logging failed operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir, "test_experiment")

            # Failed checkpoint restore
            logger.log_event(
                "checkpoint_restore",
                "failed",
                {"checkpoint_id": "invalid_ckpt"},
                error="Checkpoint file not found at /path/to/checkpoint",
            )

            # Failed query
            logger.log_event(
                "query_execute",
                "failed",
                {"query_type": "natural_language", "query_text": "invalid syntax"},
                error="Query parsing error: unexpected token",
            )

            summary = logger.get_log_summary()
            assert summary["by_status"]["failed"] == 2

            with open(logger.json_path, 'r') as f:
                events = json.load(f)

            # Reverse chronological order: query_execute (most recent) is first
            assert events[0]["error"] == "Query parsing error: unexpected token"
            assert events[1]["error"] == "Checkpoint file not found at /path/to/checkpoint"
