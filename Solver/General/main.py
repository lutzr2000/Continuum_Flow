from typing import Any, Callable

from time import perf_counter
import threading


MessageCallback = Callable[[dict[str, Any]], None]

_message_callback: MessageCallback | None = None
_message_callback_lock = threading.Lock()


def set_message_callback(callback: MessageCallback | None) -> None:
    """
    Set the callback used for solver runtime messages.
    """
    global _message_callback

    with _message_callback_lock:
        _message_callback = callback


def clear_message_callback() -> None:
    """
    Remove the currently registered solver message callback.
    """
    global _message_callback

    with _message_callback_lock:
        _message_callback = None


def emit_message(message: dict[str, Any]) -> None:
    """
    Forward one solver runtime message to the registered callback.
    """
    with _message_callback_lock:
        callback = _message_callback

    if callback is not None:
        callback(message)


def main(config: dict[str, Any]) -> None:
    """
    Configure the runtime environment and execute the requested solver backend.

    Total bake time is printed from a ``finally`` block, so timing information
    is emitted after both successful runs and failures.
    """
    total_start_time = perf_counter()

    try:
        simulation_cfg = config.get("simulation") or {}
        solver_backend = (
            str((simulation_cfg.get("settings") or {}).get("solver_backend", "GPU"))
            .strip()
            .upper()
        )

        if solver_backend in {"CPU", "GPU"}:
            from ..Kernel import solver as solver_kernel_module

            return solver_kernel_module.solver(config)

        raise ValueError(f"Unsupported solver backend: {solver_backend}")

    finally:
        total_runtime = perf_counter() - total_start_time
        print(f"Bake runtime: {total_runtime:.3f} s")
        print("################################################################")
