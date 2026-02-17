"""Master Coordinator — the brain of the distributed system.

Owns the kill switch, the task queue, the agent registry, and the
message bus. All agent activity flows through here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .kill_switch import KillSwitch, KillSwitchState
from .task_queue import Task, TaskQueue, TaskStatus
from ..comms.message_bus import MessageBus, Message, MessageType
from ..monitoring.health import HealthMonitor

logger = logging.getLogger(__name__)


@dataclass
class AgentRecord:
    """Metadata about a registered agent."""

    agent_id: str
    capabilities: list[str]
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: str = "idle"
    current_task_id: str | None = None


class MasterCoordinator:
    """Central coordinator for the OpenClaw distributed agent system.

    Responsibilities:
      - Agent registration and discovery
      - Task distribution and load balancing
      - Kill switch management
      - Health monitoring
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        grace_period: float = 30.0,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 15.0,
    ) -> None:
        self._state_dir = state_dir or Path("/tmp/openclaw")

        # Core subsystems
        self.kill_switch = KillSwitch(
            state_file=self._state_dir / "kill_switch.json",
            grace_period_seconds=grace_period,
        )
        self.task_queue = TaskQueue()
        self.message_bus = MessageBus()
        self.health_monitor = HealthMonitor(
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )

        # Agent registry
        self._agents: dict[str, AgentRecord] = {}
        self._lock = asyncio.Lock()
        self._running = False

        # Wire up kill switch hooks
        self.kill_switch.on_shutdown(self._on_kill_switch)

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Boot the coordinator and begin accepting agents."""
        if not self.kill_switch.is_alive:
            logger.error(
                "Cannot start: kill switch is in state %s. "
                "Disengage the kill switch first.",
                self.kill_switch.state.value,
            )
            return

        self._running = True
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # Start the message bus listener
        self.message_bus.subscribe(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.message_bus.subscribe(MessageType.TASK_RESULT, self._handle_task_result)
        self.message_bus.subscribe(MessageType.REGISTER, self._handle_register)

        # Start background loops
        asyncio.create_task(self._dispatch_loop())
        asyncio.create_task(self._health_check_loop())

        logger.info("MasterCoordinator started.")

    async def stop(self, reason: str = "coordinator shutdown") -> None:
        """Gracefully stop the coordinator."""
        self._running = False
        await self.kill_switch.engage(reason=reason, triggered_by="coordinator")

    # -- agent management -----------------------------------------------------

    async def register_agent(
        self,
        agent_id: str,
        capabilities: list[str],
    ) -> bool:
        """Register an agent with the coordinator."""
        if not self.kill_switch.is_alive:
            logger.warning("Rejecting registration: system is not active.")
            return False

        async with self._lock:
            self._agents[agent_id] = AgentRecord(
                agent_id=agent_id,
                capabilities=capabilities,
            )
        self.health_monitor.register(agent_id)
        logger.info("Agent registered: %s (capabilities: %s)", agent_id, capabilities)
        return True

    async def unregister_agent(self, agent_id: str) -> None:
        async with self._lock:
            self._agents.pop(agent_id, None)
        self.health_monitor.unregister(agent_id)
        logger.info("Agent unregistered: %s", agent_id)

    def get_agents(self) -> dict[str, AgentRecord]:
        return dict(self._agents)

    # -- task management ------------------------------------------------------

    async def submit_task(
        self,
        name: str,
        payload: dict[str, Any],
        priority: int = 5,
    ) -> str:
        """Submit a task for distributed processing."""
        if not self.kill_switch.is_alive:
            raise RuntimeError("Cannot submit tasks: system is shut down.")

        task = Task(name=name, payload=payload)
        task_id = await self.task_queue.submit(task, priority=priority)
        logger.info("Task submitted: %s (%s)", task_id, name)
        return task_id

    async def submit_composite_task(
        self,
        name: str,
        subtasks: list[dict[str, Any]],
        priority: int = 5,
    ) -> str:
        """Submit a task that is broken into sub-tasks for parallel processing."""
        parent = Task(name=name, payload={"type": "composite"})
        parent_id = await self.task_queue.submit(parent, priority=priority)

        for sub in subtasks:
            child = Task(
                name=sub.get("name", name),
                payload=sub.get("payload", {}),
                parent_task_id=parent_id,
            )
            child_id = await self.task_queue.submit(child, priority=priority)
            parent.subtask_ids.append(child_id)

        logger.info(
            "Composite task submitted: %s with %d subtasks", parent_id, len(subtasks)
        )
        return parent_id

    # -- status ---------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "kill_switch": self.kill_switch.state.value,
            "agents": {
                aid: {
                    "status": rec.status,
                    "capabilities": rec.capabilities,
                    "last_heartbeat": rec.last_heartbeat,
                    "current_task": rec.current_task_id,
                }
                for aid, rec in self._agents.items()
            },
            "tasks_pending": self.task_queue.pending_count,
            "health": self.health_monitor.summary(),
        }

    # -- internal loops -------------------------------------------------------

    async def _dispatch_loop(self) -> None:
        """Continuously assign pending tasks to idle agents."""
        while self._running and self.kill_switch.is_alive:
            async with self._lock:
                idle_agents = [
                    a for a in self._agents.values() if a.status == "idle"
                ]

            for agent_rec in idle_agents:
                task = await self.task_queue.claim(agent_rec.agent_id)
                if task is None:
                    break
                agent_rec.status = "busy"
                agent_rec.current_task_id = task.task_id

                await self.message_bus.publish(Message(
                    msg_type=MessageType.TASK_ASSIGN,
                    sender="coordinator",
                    recipient=agent_rec.agent_id,
                    payload=task.to_dict(),
                ))

            await asyncio.sleep(0.5)

    async def _health_check_loop(self) -> None:
        """Periodically check agent health."""
        while self._running:
            dead_agents = self.health_monitor.check_all()
            for agent_id in dead_agents:
                logger.warning("Agent %s missed heartbeat — removing.", agent_id)
                await self.unregister_agent(agent_id)
            await asyncio.sleep(self.health_monitor.heartbeat_interval)

    # -- message handlers -----------------------------------------------------

    async def _handle_heartbeat(self, msg: Message) -> None:
        async with self._lock:
            agent = self._agents.get(msg.sender)
            if agent:
                agent.last_heartbeat = time.time()
        self.health_monitor.record_heartbeat(msg.sender)

    async def _handle_task_result(self, msg: Message) -> None:
        task_id = msg.payload.get("task_id")
        success = msg.payload.get("success", False)
        result = msg.payload.get("result")
        error = msg.payload.get("error")

        if success:
            await self.task_queue.complete(task_id, result)
        else:
            await self.task_queue.fail(task_id, error or "unknown error")

        # Mark agent idle again
        async with self._lock:
            agent = self._agents.get(msg.sender)
            if agent:
                agent.status = "idle"
                agent.current_task_id = None

    async def _handle_register(self, msg: Message) -> None:
        caps = msg.payload.get("capabilities", [])
        await self.register_agent(msg.sender, caps)

    # -- kill switch callback -------------------------------------------------

    async def _on_kill_switch(self, ks: KillSwitch) -> None:
        """Called when the kill switch is engaged."""
        logger.warning("Kill switch callback: cancelling all tasks and notifying agents.")
        cancelled = await self.task_queue.cancel_all()
        logger.info("Cancelled %d pending tasks.", cancelled)

        # Broadcast shutdown to all agents
        await self.message_bus.publish(Message(
            msg_type=MessageType.SHUTDOWN,
            sender="coordinator",
            recipient="*",
            payload={"reason": "kill switch engaged", "state": ks.state.value},
        ))
