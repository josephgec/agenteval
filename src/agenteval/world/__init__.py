"""Simulated enterprise systems.

Importing this package registers every tool. `agenteval.registry.REGISTRY` is
empty until this happens, so the harness imports it before building a session.
"""

from . import admin, crm, docs, email, expenses, hr, tickets  # noqa: F401
from ..state import Mutation, World, WorldError

__all__ = ["World", "WorldError", "Mutation"]
