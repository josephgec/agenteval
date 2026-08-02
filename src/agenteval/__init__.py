"""agenteval — an agentic evaluation harness for enterprise workflow tasks.

    from agenteval import discover, run_suite, RunConfig, ClaudeAgent

    tasks = discover("tasks")
    results = await run_suite(tasks, ClaudeAgent(model="claude-opus-5"))

The pieces are independent on purpose: swap the agent to evaluate a different
scaffold, add a task directory to extend coverage, or import `World` and the
tool registry on their own to build something else entirely.
"""

from .agents import Agent, ClaudeAgent, OllamaAgent, ScriptedAgent, build_agent
from .checks import Checks, checks
from .cost import cost_usd
from .grading import LLMJudge
from .registry import P, REGISTRY, ToolSession, all_tools, tool
from .runner import RunConfig, run_one, run_suite
from .tasks import LoadedTask, discover, load_task
from .types import Check, RunResult, Score, TaskSpec, Trajectory, Usage
from .world import World

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "ClaudeAgent",
    "OllamaAgent",
    "ScriptedAgent",
    "build_agent",
    "Check",
    "Checks",
    "checks",
    "cost_usd",
    "LLMJudge",
    "P",
    "REGISTRY",
    "ToolSession",
    "all_tools",
    "tool",
    "RunConfig",
    "run_one",
    "run_suite",
    "LoadedTask",
    "discover",
    "load_task",
    "RunResult",
    "Score",
    "TaskSpec",
    "Trajectory",
    "Usage",
    "World",
]
