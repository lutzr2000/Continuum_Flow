from pathlib import Path
import contextlib
import json
import sys
import traceback


class _JsonLogStream:
    def __init__(self):
        self._buffer = ""

    def write(self, text):
        text = str(text or "")
        if not text:
            return 0

        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip()
            if line:
                _emit_message({"type": "log", "message": line})
        return len(text)

    def flush(self):
        remaining = self._buffer.strip()
        if remaining:
            _emit_message({"type": "log", "message": remaining})
        self._buffer = ""


def _emit_message(message):
    sys.__stdout__.write(json.dumps(message) + "\n")
    sys.__stdout__.flush()


def _write_config_to_disk(config):
    config = config or {}
    bake_directory_value = config.get("bake_directory") or ""
    if not str(bake_directory_value).strip():
        raise ValueError("Missing bake_directory in config.")

    bake_directory = Path(bake_directory_value).resolve()
    bake_directory.mkdir(parents=True, exist_ok=True)

    config_path = bake_directory / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(f"Dummy bake wrote config to: {config_path}")
    return config_path


def main(config=None):
    return _write_config_to_disk(config)


def _run_with_worker_logging(callback, *args, **kwargs):
    logger = _JsonLogStream()
    with contextlib.redirect_stdout(logger), contextlib.redirect_stderr(logger):
        try:
            return callback(*args, **kwargs)
        finally:
            logger.flush()


def _run_worker_loop():
    _emit_message({"type": "ready"})

    for raw_line in sys.stdin:
        payload = raw_line.strip()
        if not payload:
            continue

        try:
            message = json.loads(payload)
        except json.JSONDecodeError:
            _emit_message({
                "type": "error",
                "message": "Received invalid JSON command in solver worker",
            })
            continue

        command = str(message.get("command") or "").strip().lower()

        if command == "shutdown":
            break

        if command == "preload":
            backend = str(message.get("backend") or "GPU").strip().upper()
            _emit_message({
                "type": "preload_complete",
                "backend": backend,
                "success": True,
            })
            continue

        if command == "run_job":
            job_id = int(message.get("job_id", 0) or 0)
            config = message.get("config") or {}
            _emit_message({"type": "job_started", "job_id": job_id})
            try:
                _run_with_worker_logging(main, config)
                _emit_message({
                    "type": "job_finished",
                    "job_id": job_id,
                    "success": True,
                })
            except Exception:
                _emit_message({
                    "type": "job_finished",
                    "job_id": job_id,
                    "success": False,
                    "message": "Dummy solver job failed",
                    "traceback": traceback.format_exc(),
                })
            continue

        _emit_message({
            "type": "error",
            "message": f"Unknown solver worker command: {command}",
        })


if __name__ == "__main__":
    try:
        if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
            _run_worker_loop()
        else:
            if len(sys.argv) < 2:
                raise ValueError("Expected bake directory path as first argument.")

            bake_directory = Path(sys.argv[1]).resolve()
            config = json.load(sys.stdin)
            config["bake_directory"] = str(bake_directory)
            main(config)

    except Exception:
        traceback.print_exc()
        sys.exit(1)
