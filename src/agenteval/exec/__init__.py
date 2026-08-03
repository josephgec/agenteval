"""Real execution: containers that tool calls run inside.

Importing this registers the exec tools, the same way `agenteval.world` does
for the simulated ones.
"""

from . import tools  # noqa: F401  (registers exec_bash and friends)
from .environment import (
    DEFAULT_IMAGE,
    Environment,
    EnvironmentSpec,
    ExecResult,
    available,
    image_present,
)
from .tools import EXEC_TOOLS, attach, harvest_into, snapshot

__all__ = [
    "Environment", "EnvironmentSpec", "ExecResult", "DEFAULT_IMAGE",
    "EXEC_TOOLS", "attach", "harvest_into", "snapshot", "available", "image_present",
]
