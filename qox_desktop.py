"""PyInstaller entry point: a Windows menu, not the CLI."""

from qox.desktop import main

if __name__ == "__main__":
    raise SystemExit(main())
