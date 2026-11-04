"""Open a saved lesson with project imports set on Windows, macOS, or Linux."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--edit", action="store_true", help="Open the code editor instead of the interactive reading view.")
    mode.add_argument("--view", action="store_true", help="Open the interactive reading view (the default).")
    parser.add_argument("--notebook", type=int, choices=(1, 2, 3), default=1,
                        help="Open lesson 1, 2, or 3 (default: 1).")
    options, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    filename = {1: "01_toy.py", 2: "02_paper.py", 3: "03_extension.py"}[options.notebook]
    return subprocess.call([sys.executable, "-m", "marimo", "edit" if options.edit else "run",
                            str(root / "notebooks" / filename), *extra], cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
