import time
from statistics import mean, stdev

class ScopeTimer:
    def __init__(self, tag=None):
        self.tag = tag
        self.times = []
        self._start_time = None

    def start(self):
        """Start the timer."""
        if self._start_time is not None:
            raise RuntimeError("Timer is already running. Stop it before starting again.")
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer and record the elapsed time."""
        if self._start_time is None:
            raise RuntimeError("Timer is not running. Start it before stopping.")
        elapsed_time = time.perf_counter() - self._start_time
        self.times.append(elapsed_time)
        self._start_time = None
        return elapsed_time

    def reset(self):
        """Reset the timer statistics and clear all recorded times."""
        self.times.clear()

    def __enter__(self):
        """Enter the context, start timing."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, stop timing and record."""
        self.stop()

    @property
    def count(self):
        """Return the number of recorded times."""
        return len(self.times)

    @property
    def total_time(self):
        """Return the total elapsed time across all recorded runs."""
        return sum(self.times)

    @property
    def average_time(self):
        """Return the average time per run, or None if no runs."""
        return mean(self.times) if self.times else None

    @property
    def std_dev(self):
        """Return the standard deviation of the times, or None if insufficient data."""
        return stdev(self.times) if len(self.times) > 1 else None

    def __str__(self):
        """Human-readable format with statistics."""
        tag_info = f"{self.tag}" if self.tag else "NoTag"
        count_info = f"#{self.count}"
        total_info = f"ttl={self.total_time:.4f} s"
        avg_info = f"avg={self.average_time:.4f} s" if self.average_time is not None else "N/A"
        std_dev_info = f"std={self.std_dev:.4f} s" if self.std_dev is not None else "N/A"

        return f"Scope[{tag_info}][{count_info}]: {total_info} | {avg_info} | {std_dev_info} {self.times[:3]}"

# Example Usage

# As a context manager
with ScopeTimer(tag="Example") as timer:
    time.sleep(1)  # Simulate some operation

# Using start and stop manually
timer = ScopeTimer(tag="Manual")
timer.start()
time.sleep(0.5)  # Simulate an operation
timer.stop()

timer.start()
time.sleep(0.2)  # Another operation
timer.stop()

# Printing the timer object shows all information
print(timer)