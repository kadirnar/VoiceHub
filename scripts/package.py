import subprocess
import sys
from pathlib import Path


def run_command(*command: str) -> None:
    """Executes a given command using the subprocess module.

    Args:
        command: The command and its arguments.
    """
    subprocess.run(command, check=True)


def main() -> None:
    """Build PEP 517 artifacts from pyproject.toml and upload with Twine."""
    run_command(sys.executable, "-m", "build")
    artifacts = sorted(str(path) for pattern in ("*.whl", "*.tar.gz") for path in Path("dist").glob(pattern))
    if not artifacts:
        raise FileNotFoundError("No distributions were created in dist/.")
    run_command(sys.executable, "-m", "twine", "upload", *artifacts)


if __name__ == "__main__":
    main()
