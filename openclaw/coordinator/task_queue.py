"""Distributed task queue for agent work distribution."""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class TaskStatus(enum.Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A unit of work to be processed by an agent."""

    name: str
    payload: dict[str, Any]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: str | None = None
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    parent_task_id: str | None = None
    subtask_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "payload": self.payload,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "parent_task_id": self.parent_task_id,
            "subtask_ids": self.subtask_ids,
        }


class TaskQueue:
    """Async-safe task queue with priority and cancellation support."""

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, float, Task]] = (
            asyncio.PriorityQueue(maxsize=maxsize)
        )
        self._tasks: dict[str, Task] = {}
        self._lock = asyncio.Lock()

    async def submit(self, task: Task, priority: int = 5) -> str:
        """Add a task to the queue. Lower priority number = higher priority."""
        async with self._lock:
            self._tasks[task.task_id] = task
        await self._queue.put((priority, task.created_at, task))
        return task.task_id

    async def claim(self, agent_id: str) -> Task | None:
        """Claim the next available task for an agent."""
        try:
            _, _, task = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

        async with self._lock:
            task.status = TaskStatus.ASSIGNED
            task.assigned_to = agent_id
            task.started_at = time.time()
        return task

    async def complete(self, task_id: str, result: Any = None) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()

    async def fail(self, task_id: str, error: str) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.FAILED
                task.error = error
                task.completed_at = time.time()

    async def cancel_all(self) -> int:
        """Cancel all pending tasks. Returns the count of cancelled tasks."""
        cancelled = 0
        async with self._lock:
            for task in self._tasks.values():
                if task.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                    task.status = TaskStatus.CANCELLED
                    cancelled += 1
        return cancelled

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def all_tasks(self) -> dict[str, Task]:
        return dict(self._tasks)
