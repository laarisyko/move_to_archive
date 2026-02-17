"""Agent Worker — the unit of computation in the distributed system.

Each worker registers with the coordinator, receives tasks via the
message bus, executes them, and reports results. Workers respect the
kill switch at all times and will shut down immediately when told to.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from .capabilities import Capability
from ..comms.message_bus import MessageBus, Message, MessageType

logger = logging.getLogger(__name__)


class AgentWorker:
    """A single distributed agent worker.

    Usage::

        async def handle_summarize(payload):
            text = payload["text"]
            return {"summary": text[:100] + "..."}

        worker = AgentWorker(
            message_bus=bus,
            capabilities=[
                Capability(name="summarize", handler=handle_summarize),
            ],
        )
        await worker.start()
    """

    def __init__(
        self,
        message_bus: MessageBus,
        capabilities: list[Capability] | None = None,
        agent_id: str | None = None,
    ) -> None:
        self.agent_id = agent_id or f"agent-{uuid.uuid4().hex[:8]}"
        self.message_bus = message_bus
        self._capabilities = {c.name: c for c in (capabilities or [])}
        self._running = False
        self._current_task: dict[str, Any] | None = None

    @property
    def capability_names(self) -> list[str]:
        return list(self._capabilities.keys())

    def add_capability(self, cap: Capability) -> None:
        self._capabilities[cap.name] = cap

    # -- lifecycle ------------------------------------------------------------

    async def start(self) -> None:
        """Register with the coordinator and start processing."""
        self._running = True

        # Subscribe to messages addressed to us (or broadcast)
        self.message_bus.subscribe(MessageType.TASK_ASSIGN, self._on_task_assigned)
        self.message_bus.subscribe(MessageType.SHUTDOWN, self._on_shutdown)

        # Send registration message
        await self.message_bus.publish(Message(
            msg_type=MessageType.REGISTER,
            sender=self.agent_id,
            recipient="coordinator",
            payload={"capabilities": self.capability_names},
        ))

        # Start heartbeat loop
        asyncio.create_task(self._heartbeat_loop())

        logger.info("Worker %s started with capabilities: %s", self.agent_id, self.capability_names)

    async def stop(self) -> None:
        """Gracefully stop this worker."""
        self._running = False
        logger.info("Worker %s stopping.", self.agent_id)

    # -- task execution -------------------------------------------------------

    async def _on_task_assigned(self, msg: Message) -> None:
        """Handle an incoming task assignment."""
        if msg.recipient != self.agent_id and msg.recipient != "*":
            return  # not for us

        if not self._running:
            logger.warning("Worker %s is stopped; ignoring task.", self.agent_id)
            return

        task_data = msg.payload
        task_id = task_data.get("task_id", "unknown")
        task_name = task_data.get("name", "unknown")

        logger.info("Worker %s received task %s (%s)", self.agent_id, task_id, task_name)

        # Find a matching capability
        cap = self._capabilities.get(task_name)
        if cap is None:
            # Check tag-based matching
            for c in self._capabilities.values():
                if c.matches(task_name):
                    cap = c
                    break

        if cap is None:
            await self._report_result(task_id, success=False, error=f"No capability for '{task_name}'")
            return

        self._current_task = task_data

        try:
            result = await cap.handler(task_data.get("payload", {}))
            await self._report_result(task_id, success=True, result=result)
        except Exception as exc:
            logger.exception("Worker %s failed task %s", self.agent_id, task_id)
            await self._report_result(task_id, success=False, error=str(exc))
        finally:
            self._current_task = None

    async def _report_result(
        self,
        task_id: str,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        await self.message_bus.publish(Message(
            msg_type=MessageType.TASK_RESULT,
            sender=self.agent_id,
            recipient="coordinator",
            payload={
                "task_id": task_id,
                "success": success,
                "result": result,
                "error": error,
            },
        ))

    # -- shutdown handling ----------------------------------------------------

    async def _on_shutdown(self, msg: Message) -> None:
        """Respond to a system-wide shutdown command."""
        logger.warning(
            "Worker %s received SHUTDOWN: %s",
            self.agent_id,
            msg.payload.get("reason", "unknown"),
        )
        await self.stop()

    # -- heartbeat ------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        while self._running:
            await self.message_bus.publish(Message(
                msg_type=MessageType.HEARTBEAT,
                sender=self.agent_id,
                recipient="coordinator",
                payload={"status": "busy" if self._current_task else "idle"},
            ))
            await asyncio.sleep(5.0)
