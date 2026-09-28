"""Compatibility wrapper for the fingertip-train command."""

from fingertip_depth.training_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
