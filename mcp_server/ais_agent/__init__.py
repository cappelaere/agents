"""Standalone copy of `ais_agent` for the MCP bridge.

This package mirrors the minimal files needed from the repository's
`agents/ais_agent` so the bridge can import and run without mounting
the host source tree.
"""

from .ais_agent import app  # re-export the FastAPI app

__all__ = ["app"]
