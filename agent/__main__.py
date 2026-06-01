"""Allow `python -m agent` to run the CLI."""

from agent.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
