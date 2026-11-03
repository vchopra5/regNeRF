"""Open the toy lesson with project imports set on Windows, macOS, or Linux."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true", help="Open the reading view instead of the editor.")
    options, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    return subprocess.call([sys.executable, "-m", "marimo", "run" if options.view else "edit",
                            str(root / "notebooks" / "01_toy.py"), *extra], cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
