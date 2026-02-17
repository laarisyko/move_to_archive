"""In-process async message bus for agent ↔ coordinator communication.

This is a lightweight pub/sub bus. In production, swap the transport
layer for something like Redis Streams, NATS, or RabbitMQ — the
interface stays the same.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class MessageType(enum.Enum):
    # Agent ↔ Coordinator
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    SHUTDOWN = "shutdown"

    # Agent ↔ Agent
    PEER_REQUEST = "peer_request"
    PEER_RESPONSE = "peer_response"


@dataclass
class Message:
    """A message exchanged over the bus."""

    msg_type: MessageType
    sender: str
    recipient: str  # agent_id or "*" for broadcast
    payload: dict[str, Any] = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None  # for request/response pairing


MessageHandler = Callable[[Message], Awaitable[None]]


class MessageBus:
    """Simple async pub/sub message bus.

    Subscribers register for specific message types and receive all
    messages of that type. Routing by recipient is the subscriber's
    responsibility (this keeps the bus simple and flexible).
    """

    def __init__(self) -> None:
        self._subscribers: dict[MessageType, list[MessageHandler]] = {}
        self._history: list[Message] = []
        self._max_history = 1000

    def subscribe(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """Subscribe to a message type."""
        self._subscribers.setdefault(msg_type, []).append(handler)

    def unsubscribe(self, msg_type: MessageType, handler: MessageHandler) -> None:
        """Remove a subscription."""
        handlers = self._subscribers.get(msg_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, message: Message) -> None:
        """Publish a message to all subscribers of its type."""
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        handlers = self._subscribers.get(message.msg_type, [])
        if not handlers:
            return

        # Fire all handlers concurrently
        results = await asyncio.gather(
            *(h(message) for h in handlers),
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Handler %s failed for message %s: %s",
                    handlers[i],
                    message.message_id,
                    result,
                )

    async def request(
        self,
        message: Message,
        timeout: float = 30.0,
    ) -> Message | None:
        """Send a request and wait for a correlated response."""
        correlation_id = message.message_id
        message.correlation_id = correlation_id

        future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

        # Determine the expected response type
        response_type = {
            MessageType.PEER_REQUEST: MessageType.PEER_RESPONSE,
            MessageType.TASK_ASSIGN: MessageType.TASK_RESULT,
        }.get(message.msg_type)

        if response_type is None:
            raise ValueError(f"No response type defined for {message.msg_type}")

        async def _catch_response(msg: Message) -> None:
            if msg.correlation_id == correlation_id and not future.done():
                future.set_result(msg)

        self.subscribe(response_type, _catch_response)

        try:
            await self.publish(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Request %s timed out after %.1fs", correlation_id, timeout)
            return None
        finally:
            self.unsubscribe(response_type, _catch_response)

    @property
    def recent_messages(self) -> list[Message]:
        return list(self._history[-50:])
