"""Logic-only AIS implementation for the MCP bridge.

This package mirrors the minimal files needed from the repository's
`agents/ais_agent` so the bridge can import and run without mounting
the host source tree. All FastAPI routes are defined in `mcp_server/app.py`;
this package only provides implementation functions.
"""

from . import ais_agent

__all__ = ["ais_agent"]
