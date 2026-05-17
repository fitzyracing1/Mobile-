"""Allow `python -m mars_ai` to run the CLI."""

from mars_ai.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
