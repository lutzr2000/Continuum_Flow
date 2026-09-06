"""Per-run wall-clock timings, including completed CUDA work and JIT compilation."""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter

from numba import cuda


class RunTimings:
    def __init__(self):
        self.entries = {}

    @contextmanager
    def section(self, group, name, gpu=False):
        # Drain earlier work so it cannot be charged to this section.
        if gpu:
            cuda.synchronize()
        start = perf_counter()
        try:
            yield
        finally:
            try:
                if gpu:
                    cuda.synchronize()
            finally:
                elapsed = perf_counter() - start
                entry = self.entries.setdefault((group, name), [0, 0.0])
                entry[0] += 1
                entry[1] += elapsed

    def report(self, total, status):
        print(f"Timing report ({status}) - total run: {total:.6f} s")
        width = max(62, max((len(name) for _, name in self.entries), default=0))
        for group in sorted(
            {key[0] for key in self.entries}, key=lambda name: (name != "solver", name)
        ):
            print(f"[{group}]")
            print(
                f"{'Section / method':{width}} {'Calls':>8} {'Total (s)':>12} {'% run':>9} {'Avg (ms)':>12}"
            )
            entries = (
                (name, values)
                for (entry_group, name), values in self.entries.items()
                if entry_group == group
            )
            for name, (calls, elapsed) in sorted(
                entries, key=lambda item: item[1][1], reverse=True
            ):
                percent = 100.0 * elapsed / total if total > 0.0 else 0.0
                print(
                    f"{name:{width}} {calls:8d} {elapsed:12.6f} {percent:8.2f}% {1000.0 * elapsed / calls:12.6f}"
                )


def profiled_run(function):
    """Own a fresh report per run; nested functions can share its collector."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        if kwargs.get("timings") is not None:
            return function(*args, **kwargs)
        timings = RunTimings()
        kwargs["timings"] = timings
        start = perf_counter()
        status = "failed / partial"
        try:
            result = function(*args, **kwargs)
            status = "finished / clean stop"
            return result
        finally:
            timings.report(perf_counter() - start, status)

    return wrapped
