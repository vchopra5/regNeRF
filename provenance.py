"""Record the code revision and source file hashes for an experiment."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def git(*arguments):
    """Run a Git query in the project root. Return "unknown" if Git is not available."""
    try:
        result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def file_hashes(names):
    """Return the SHA-256 of each named file, relative to the project root."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
