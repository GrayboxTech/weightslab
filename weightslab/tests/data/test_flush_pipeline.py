"""
Unit tests for the buffer → DataFrame → H5 flush pipeline in LedgeredDataFrameManager.

Verified behaviors:
  1. flush() releases _buffer_lock before DF/H5 work — training can enqueue during a flush.
  2. flush_async() returns after buffer drain, not after H5 write completes.
  3. If buffer refills while H5 write is ongoing, training waits only until the flush
     thread drains the buffer again (not until H5 finishes).
  4. In-memory buffer is bounded to ≤ flush_max_rows records at any point.
"""

import time
import threading
import unittest
import numpy as np

from unittest.mock import MagicMock, patch

from weightslab.data.dataframe_manager import LedgeredDataFrameManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mgr(flush_max_rows=4, enable_flushing_threads=False) -> LedgeredDataFrameManager:
    mgr = LedgeredDataFrameManager(
        flush_interval=60.0,          # disable periodic timer during tests
        flush_max_rows=flush_max_rows,
        enable_flushing_threads=enable_flushing_threads,
        enable_h5_persistence=False,
    )
    return mgr


def _enqueue(mgr, sample_ids, step=0):
    n = len(sample_ids)
    mgr.enqueue_batch(
        sample_ids=sample_ids,
        preds_raw=np.random.rand(n, 1, 4, 4).astype(np.float32),
        preds=np.zeros(n, dtype=np.int64),
        losses={"loss": np.ones(n, dtype=np.float32) * 0.1},
        targets=np.zeros(n, dtype=np.int64),
        step=step,
    )


# ---------------------------------------------------------------------------
# Test 1: flush() releases _buffer_lock before DF/H5 work
# ---------------------------------------------------------------------------

class TestFlushReleasesBufferLockEarly(unittest.TestCase):
    """flush() must release _buffer_lock right after draining so that
    enqueue_batch() is not blocked during the (slow) DF/H5 phase."""

    def test_enqueue_succeeds_while_flush_doing_df_write(self):
        mgr = _make_mgr(enable_flushing_threads=False)

        # Seed the DataFrame so _apply_buffer_records has rows to update.
        import pandas as pd
        for i in range(4):
            mgr._buffer[str(i)] = {"sample_id": str(i), "origin": "train"}
        mgr._drain_buffer()

        enqueue_started = threading.Event()
        enqueue_finished = threading.Event()
        df_write_started = threading.Event()
        df_write_may_proceed = threading.Event()

        original_apply = mgr._apply_buffer_records

        def slow_apply(records):
            df_write_started.set()
            df_write_may_proceed.wait(timeout=5)
            original_apply(records)

        def do_flush():
            with patch.object(mgr, "_apply_buffer_records", side_effect=slow_apply):
                # Pre-fill buffer so flush has something to drain.
                with mgr._buffer_lock:
                    for i in range(4):
                        mgr._buffer[str(i)] = {"sample_id": str(i), "origin": "train"}
                mgr.flush()

        def do_enqueue():
            enqueue_started.set()
            _enqueue(mgr, ["99", "100"])
            enqueue_finished.set()

        flush_thread = threading.Thread(target=do_flush)
        flush_thread.start()

        # Wait until flush is inside the slow DF write (buffer already drained).
        df_write_started.wait(timeout=5)

        # Now enqueue from a second thread — must NOT be blocked on _buffer_lock.
        enqueue_thread = threading.Thread(target=do_enqueue)
        enqueue_thread.start()

        # Give enqueue thread 1 second to complete; if it can't, flush is holding
        # _buffer_lock too long.
        enqueue_finished.wait(timeout=1.0)
        self.assertTrue(
            enqueue_finished.is_set(),
            "enqueue_batch() was blocked while flush() was doing DF work — "
            "_buffer_lock was held too long.",
        )

        df_write_may_proceed.set()
        flush_thread.join(timeout=5)
        enqueue_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Test 2: flush_async() returns after buffer drain, not after H5 write
# ---------------------------------------------------------------------------

class TestFlushAsyncReturnsAfterBufferDrain(unittest.TestCase):
    """flush_async() must return as soon as the buffer has been drained —
    not after the (potentially slow) H5 write.

    Scenario: fill buffer once. flush_async() (called inside enqueue_batch) should
    return after the flush thread drains the buffer (~thread wake-up time), well
    before the slow H5 write completes.
    """

    def test_flush_async_does_not_wait_for_h5(self):
        H5_WRITE_DELAY = 1.0  # seconds — intentionally slow

        mgr = _make_mgr(flush_max_rows=4, enable_flushing_threads=True)

        original_flush_to_h5 = mgr._flush_to_h5_if_needed

        def slow_h5(*args, **kwargs):
            time.sleep(H5_WRITE_DELAY)
            original_flush_to_h5(*args, **kwargs)

        with patch.object(mgr, "_flush_to_h5_if_needed", side_effect=slow_h5):
            # Fill buffer to capacity — this triggers flush_async() inside enqueue_batch.
            # flush_async() should return after buffer drain (~thread wake-up, <<1s),
            # NOT after the H5 write (1s).
            t0 = time.time()
            _enqueue(mgr, ["0", "1", "2", "3"])
            elapsed = time.time() - t0

        mgr.stop()

        # The flush thread needs a few hundred ms to wake up and drain.
        # Anything under 80% of H5_WRITE_DELAY proves we did NOT wait for H5.
        self.assertLess(
            elapsed,
            H5_WRITE_DELAY * 0.8,
            f"flush_async() waited {elapsed:.2f}s — it should return after buffer "
            f"drain (~ms), not after the {H5_WRITE_DELAY}s H5 write.",
        )


# ---------------------------------------------------------------------------
# Test 3: buffer refills while H5 writing — training waits, then resumes
# ---------------------------------------------------------------------------

class TestBufferRefillDuringH5Write(unittest.TestCase):
    """If the buffer fills while an H5 write is in progress, the training
    thread must wait (bounded wait) and resume once the flush thread drains
    the buffer again in the next cycle."""

    def test_training_resumes_after_second_drain(self):
        FLUSH_MAX = 4
        H5_WRITE_DELAY = 0.3  # seconds

        mgr = _make_mgr(flush_max_rows=FLUSH_MAX, enable_flushing_threads=True)

        flush_cycle_count = {"n": 0}
        original_flush_to_h5 = mgr._flush_to_h5_if_needed

        def counting_slow_h5(*args, **kwargs):
            flush_cycle_count["n"] += 1
            time.sleep(H5_WRITE_DELAY)
            original_flush_to_h5(*args, **kwargs)

        second_enqueue_returned = threading.Event()

        def training_sim():
            _enqueue(mgr, [str(i) for i in range(FLUSH_MAX)])  # fills buffer, triggers flush
            time.sleep(0.05)  # let flush thread start H5 write
            _enqueue(mgr, [str(i) for i in range(FLUSH_MAX, FLUSH_MAX * 2)])  # refill
            second_enqueue_returned.set()

        with patch.object(mgr, "_flush_to_h5_if_needed", side_effect=counting_slow_h5):
            t = threading.Thread(target=training_sim)
            t.start()
            completed = second_enqueue_returned.wait(timeout=H5_WRITE_DELAY * 6)

        mgr.stop()
        t.join(timeout=5)

        self.assertTrue(
            completed,
            "Training thread never resumed after the second buffer fill.",
        )
        self.assertGreaterEqual(
            flush_cycle_count["n"], 1,
            "Flush thread never ran an H5 write cycle.",
        )


# ---------------------------------------------------------------------------
# Test 4: memory stays bounded at <= flush_max_rows during concurrent load
# ---------------------------------------------------------------------------

class TestBufferMemoryBound(unittest.TestCase):
    """The buffer must never hold more than flush_max_rows records because
    flush_async() blocks the training thread when the buffer is at capacity."""

    def test_buffer_never_exceeds_max_rows(self):
        FLUSH_MAX = 8
        TOTAL_SAMPLES = 200

        mgr = _make_mgr(flush_max_rows=FLUSH_MAX, enable_flushing_threads=True)

        max_buffer_seen = {"n": 0}
        original_enqueue = mgr._buffer.__setitem__

        # Track high-water mark inside the buffer lock.
        observation_lock = threading.Lock()

        original_apply = mgr._apply_buffer_records

        def tracking_apply(records):
            # Snapshot buffer size right before drain completes.
            with mgr._buffer_lock:
                with observation_lock:
                    max_buffer_seen["n"] = max(max_buffer_seen["n"], len(mgr._buffer))
            original_apply(records)

        with patch.object(mgr, "_apply_buffer_records", side_effect=tracking_apply):
            for batch_start in range(0, TOTAL_SAMPLES, FLUSH_MAX // 2):
                ids = [str(i) for i in range(batch_start, batch_start + FLUSH_MAX // 2)]
                _enqueue(mgr, ids, step=batch_start)
                with mgr._buffer_lock:
                    current = len(mgr._buffer)
                with observation_lock:
                    max_buffer_seen["n"] = max(max_buffer_seen["n"], current)

        mgr.stop()

        self.assertLessEqual(
            max_buffer_seen["n"],
            FLUSH_MAX,
            f"Buffer reached {max_buffer_seen['n']} records — exceeded flush_max_rows={FLUSH_MAX}.",
        )


# ---------------------------------------------------------------------------
# Test 5: flush() uses blocking _apply_buffer_records (not nonblocking)
# ---------------------------------------------------------------------------

class TestFlushUsesBlockingApply(unittest.TestCase):
    """flush() must call _apply_buffer_records (blocking) so that records are
    guaranteed to land in the DataFrame even under lock contention."""

    def test_flush_calls_blocking_apply(self):
        mgr = _make_mgr(enable_flushing_threads=False)

        with mgr._buffer_lock:
            for i in range(3):
                mgr._buffer[str(i)] = {"sample_id": str(i), "origin": "train"}

        with patch.object(mgr, "_apply_buffer_records", wraps=mgr._apply_buffer_records) as blocking_mock, \
             patch.object(mgr, "_apply_buffer_records_nonblocking", wraps=mgr._apply_buffer_records_nonblocking) as nonblocking_mock:
            mgr.flush()

        blocking_mock.assert_called_once()
        nonblocking_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
