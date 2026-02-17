"""Agent capability declarations.

Capabilities let the coordinator route tasks to agents that can handle them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

TaskHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class Capability:
    """A named capability with an associated handler function."""

    name: str
    handler: TaskHandler
    description: str = ""
    concurrency: int = 1
    tags: list[str] = field(default_factory=list)

    def matches(self, task_name: str) -> bool:
        """Check if this capability can handle the given task name."""
        return task_name == self.name or task_name in self.tags
