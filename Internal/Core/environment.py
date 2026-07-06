import shutil
import subprocess
from pathlib import Path


DEFAULT_SOLVER_PYTHON_VERSION = "3.11"
SOLVER_ENV_DIR_NAME = ".venv"
LEGACY_SOLVER_ENV_DIR_NAME = "ContinuumFlow_env"
SOLVER_REQUIREMENTS_FILE = "requirements-solver.txt"


def project_root_directory(anchor_file=None):
    if anchor_file:
        anchor_path = Path(anchor_file).resolve()
    else:
        anchor_path = Path(__file__).resolve()

    candidate_directories = [anchor_path.parent, *anchor_path.parents]
    for candidate in candidate_directories:
        if (candidate / "Internal").exists() and (candidate / "Solver").exists():
            return candidate

    return anchor_path.parent


def solver_environment_directory(anchor_file=None):
    return project_root_directory(anchor_file) / SOLVER_ENV_DIR_NAME


def legacy_solver_environment_directory(anchor_file=None):
    return project_root_directory(anchor_file) / LEGACY_SOLVER_ENV_DIR_NAME


def _python_executable_for_environment(environment_directory):
    return environment_directory / "Scripts" / "python.exe"


def preferred_solver_python_executable(anchor_file=None):
    return _python_executable_for_environment(solver_environment_directory(anchor_file))


def solver_python_executable(anchor_file=None):
    preferred = preferred_solver_python_executable(anchor_file)
    if preferred.exists():
        return preferred

    legacy = _python_executable_for_environment(
        legacy_solver_environment_directory(anchor_file)
    )
    if legacy.exists():
        return legacy

    return preferred


def solver_environment_exists(anchor_file=None):
    return solver_python_executable(anchor_file).exists()


def solver_requirements_path(anchor_file=None):
    return project_root_directory(anchor_file) / SOLVER_REQUIREMENTS_FILE


def find_uv_executable():
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path

    raise FileNotFoundError(
        "uv was not found on PATH. Install uv first, then run the environment setup again."
    )


def _run_checked(command, cwd):
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return result

    output_parts = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(result.stderr.strip())
    output_text = "\n".join(output_parts).strip()
    if output_text:
        raise RuntimeError(output_text)

    raise RuntimeError(f"Command failed with exit code {result.returncode}.")


def install_solver_environment(anchor_file=None, python_version=DEFAULT_SOLVER_PYTHON_VERSION):
    project_root = project_root_directory(anchor_file)
    environment_directory = solver_environment_directory(anchor_file)
    requirements_path = solver_requirements_path(anchor_file)

    if not requirements_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {requirements_path}")

    uv_executable = find_uv_executable()

    _run_checked(
        [
            uv_executable,
            "venv",
            str(environment_directory),
            "--python",
            str(python_version),
        ],
        cwd=project_root,
    )

    python_executable = preferred_solver_python_executable(anchor_file)
    _run_checked(
        [
            uv_executable,
            "pip",
            "install",
            "--python",
            str(python_executable),
            "-r",
            str(requirements_path),
        ],
        cwd=project_root,
    )

    return {
        "project_root": str(project_root),
        "environment_path": str(environment_directory),
        "python_executable": str(python_executable),
    }


def solver_environment_status(anchor_file=None):
    project_root = project_root_directory(anchor_file)
    environment_directory = solver_environment_directory(anchor_file)
    python_executable = solver_python_executable(anchor_file)
    return {
        "project_root": str(project_root),
        "environment_path": str(environment_directory),
        "python_path": str(python_executable),
        "ready": python_executable.exists(),
    }
