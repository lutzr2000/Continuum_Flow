from collections import defaultdict
from functools import wraps
from time import perf_counter
from typing import Any

import pyopencl as cl


class ProfiledKernel:
    """Forward a PyOpenCL kernel call and retain its profiling event."""

    def __init__(self, kernel: cl.Kernel, timings: "RunTimings", group: str) -> None:
        self._kernel = kernel
        self._timings = timings
        self._group = group

    def __call__(self, *args: Any, **kwargs: Any) -> cl.Event:
        event = self._kernel(*args, **kwargs)
        self._timings.record_event(
            self._group,
            self._kernel.function_name,
            event,
        )
        return event

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


class RunTimings:
    """Accumulate OpenCL device timings without synchronizing every kernel."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0.0])
        self.pending: list[tuple[str, str, cl.Event]] = []
        self.queue: cl.CommandQueue | None = None

    def set_queue(self, queue: cl.CommandQueue) -> None:
        self.queue = queue

    def instrument_kernels(
        self,
        kernels: dict[str, cl.Kernel],
        group: str,
    ) -> None:
        for name, kernel in tuple(kernels.items()):
            kernels[name] = ProfiledKernel(kernel, self, group)  # type: ignore[assignment]

    def record_event(self, group: str, name: str, event: cl.Event) -> None:
        self.pending.append((group, name, event))

    def record_cpu(self, name: str, started: float) -> None:
        """Record host wall time without adding GPU synchronization."""
        elapsed = perf_counter() - started
        entry = self.entries[("cpu", name)]
        entry[0] += 1
        entry[1] += elapsed

    def _record(self, group: str, name: str, event: cl.Event) -> None:
        elapsed = (event.profile.end - event.profile.start) * 1.0e-9
        entry = self.entries[(group, name)]
        entry[0] += 1
        entry[1] += elapsed

    def collect_completed(self) -> None:
        """Collect completed events without introducing a queue wait."""
        remaining = []

        for group, name, event in self.pending:
            if event.command_execution_status == cl.command_execution_status.COMPLETE:
                self._record(group, name, event)
            else:
                remaining.append((group, name, event))

        self.pending = remaining

    def finish(self) -> None:
        """Finish the queue once and collect every remaining event."""
        if self.queue is not None:
            self.queue.finish()

        for group, name, event in self.pending:
            self._record(group, name, event)

        self.pending.clear()

    def report(self, total: float, status: str) -> None:
        """Print the same summary layout used by the Numba GPU solver."""
        print(f"Timing report ({status}) - total run: {total:.6f} s")
        width = max(62, max((len(name) for _, name in self.entries), default=0))

        for group in sorted(
            {key[0] for key in self.entries},
            key=lambda name: (name != "solver", name),
        ):
            print(f"[{group}]")
            print(
                f"{'Section / method':{width}} "
                f"{'Calls':>8} {'Total (s)':>12} {'% run':>9} {'Avg (ms)':>12}"
            )

            entries = (
                (name, values)
                for (entry_group, name), values in self.entries.items()
                if entry_group == group
            )

            for name, (calls, elapsed) in sorted(
                entries,
                key=lambda item: item[1][1],
                reverse=True,
            ):
                calls_int = int(calls)
                percent = 100.0 * elapsed / total if total > 0.0 else 0.0
                print(
                    f"{name:{width}} {calls_int:8d} {elapsed:12.6f} "
                    f"{percent:8.2f}% {1000.0 * elapsed / calls_int:12.6f}"
                )


def profiled_run(function: Any) -> Any:
    """Decorate an OpenCL solver run with device-event timing collection."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("timings") is not None:
            return function(*args, **kwargs)

        timings = RunTimings()
        kwargs["timings"] = timings
        start = perf_counter()

        try:
            result = function(*args, **kwargs)
        except BaseException:
            try:
                timings.finish()
            except Exception:
                # Preserve the solver's original exception. Events completed
                # before the failure remain available in the partial report.
                pass
            timings.report(perf_counter() - start, "failed / partial")
            raise

        try:
            timings.finish()
        except BaseException:
            timings.report(perf_counter() - start, "failed / partial")
            raise

        timings.report(perf_counter() - start, "finished / clean stop")
        return result

    return wrapped
