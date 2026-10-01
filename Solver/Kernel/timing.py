from __future__ import annotations

import inspect
from functools import wraps
from time import perf_counter
from types import ModuleType
from typing import Any, Callable

import pyopencl as cl

_CURRENT_TIMINGS: "RunTimings | None" = None


class TimedKernel:
    """Drop-in ``cl.Kernel`` proxy that records device execution time."""

    def __init__(self, kernel: cl.Kernel, timings: "RunTimings") -> None:
        self.kernel = kernel
        self.timings = timings

    def __getattr__(self, name: str) -> Any:
        return getattr(self.kernel, name)

    def __call__(self, *args: Any, **kwargs: Any) -> cl.Event:
        event = self.kernel(*args, **kwargs)
        if self.timings.active:
            event.wait()
            elapsed = (event.profile.end - event.profile.start) * 1.0e-9
            self.timings.record_kernel(self.kernel.function_name, elapsed)
        return event


class RunTimings:
    """Collect synchronized method timings and OpenCL event timings."""

    def __init__(self) -> None:
        self.entries: dict[str, list[float]] = {}
        self.queue: cl.CommandQueue | None = None
        self.active = False
        self.loop_started: float | None = None
        self.loop_elapsed = 0.0
        self.call_stack: list[list[float]] = []

    def configure(
        self,
        queue: cl.CommandQueue,
        modules: dict[str, ModuleType],
    ) -> None:
        global _CURRENT_TIMINGS
        _CURRENT_TIMINGS = self
        self.queue = queue
        for prefix, module in modules.items():
            self.instrument_module(prefix, module)

    def instrument_module(self, prefix: str, module: ModuleType) -> None:
        """Wrap functions defined by a solver module without changing call sites."""
        for function_name, function in inspect.getmembers(module, inspect.isfunction):
            if function.__module__ != module.__name__ or function_name.startswith("_"):
                continue
            original = getattr(function, "_timing_original", function)
            setattr(
                module, function_name, self.wrap(f"{prefix}.{function_name}", original)
            )

    def instrument_kernel_sets(self, *kernel_sets: dict[str, cl.Kernel]) -> None:
        for kernels in kernel_sets:
            for name, kernel in kernels.items():
                raw_kernel = (
                    kernel.kernel if isinstance(kernel, TimedKernel) else kernel
                )
                kernels[name] = TimedKernel(raw_kernel, self)  # type: ignore[assignment]

    def wrap_kernel(self, kernel: cl.Kernel) -> TimedKernel:
        return TimedKernel(kernel, self)

    def wrap(self, name: str, function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def measured(*args: Any, **kwargs: Any) -> Any:
            if not self.active:
                return function(*args, **kwargs)
            self._synchronize()
            started = perf_counter()
            frame = [0.0]
            self.call_stack.append(frame)
            try:
                return function(*args, **kwargs)
            finally:
                self._synchronize()
                elapsed = perf_counter() - started
                self.call_stack.pop()
                exclusive = max(0.0, elapsed - frame[0])
                self.record(name, exclusive)
                if self.call_stack:
                    self.call_stack[-1][0] += elapsed

        measured._timing_original = function  # type: ignore[attr-defined]
        return measured

    def call(
        self,
        name: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self.wrap(name, function)(*args, **kwargs)

    def start_loop(self) -> None:
        self._synchronize()
        self.loop_started = perf_counter()
        self.active = True

    def stop_loop(self) -> None:
        if not self.active:
            return
        self._synchronize()
        self.loop_elapsed += perf_counter() - (self.loop_started or perf_counter())
        self.loop_started = None
        self.active = False

    def record(self, name: str, elapsed: float) -> None:
        entry = self.entries.setdefault(name, [0.0, 0.0])
        entry[0] += 1
        entry[1] += elapsed

    def record_kernel(self, name: str, elapsed: float) -> None:
        self.record(name, elapsed)
        if self.call_stack:
            self.call_stack[-1][0] += elapsed

    def record_event(self, kernel: cl.Kernel, event: cl.Event) -> None:
        if not self.active:
            return
        event.wait()
        elapsed = (event.profile.end - event.profile.start) * 1.0e-9
        self.record_kernel(kernel.function_name, elapsed)

    def report(self, status: str) -> None:
        self.stop_loop()
        total = self.loop_elapsed
        print(f"Timing report ({status}) - timed loop: {total:.6f} s")
        width = max(20, max(map(len, self.entries), default=0))
        print(
            f"{'Method / kernel':{width}} {'Runtime (s)':>13} "
            f"{'% loop':>9} {'Per call (ms)':>14} {'Ncalls':>8}"
        )
        for name, (calls, elapsed) in sorted(
            self.entries.items(), key=lambda item: item[1][1], reverse=True
        ):
            percent = 100.0 * elapsed / total if total else 0.0
            per_call = 1000.0 * elapsed / calls
            print(
                f"{name:{width}} {elapsed:13.6f} {percent:8.2f}% "
                f"{per_call:14.6f} {int(calls):8d}"
            )

    def _synchronize(self) -> None:
        if self.queue is not None:
            self.queue.finish()


def profiled_run(function: Callable[..., Any]) -> Callable[..., Any]:
    """Create one timing collector and always print its final loop report."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("timings") is not None:
            return function(*args, **kwargs)
        timings = RunTimings()
        kwargs["timings"] = timings
        status = "failed / partial"
        try:
            result = function(*args, **kwargs)
            status = "finished / clean stop"
            return result
        finally:
            timings.report(status)

    return wrapped


def record_kernel_event(kernel: cl.Kernel, event: cl.Event) -> None:
    """Record kernels created dynamically outside the loaded kernel dictionaries."""
    if _CURRENT_TIMINGS is not None:
        _CURRENT_TIMINGS.record_event(kernel, event)
