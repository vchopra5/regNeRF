"""Keep the old launcher command working. Use notebooks.open_notebook."""
from notebooks.open_notebook import main


if __name__ == "__main__":
    raise SystemExit(main())
