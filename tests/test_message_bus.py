"""Tests for the message bus."""

import pytest

from openclaw.comms.message_bus import MessageBus, Message, MessageType


@pytest.mark.asyncio
async def test_publish_subscribe():
    bus = MessageBus()
    received = []

    async def handler(msg: Message) -> None:
        received.append(msg)

    bus.subscribe(MessageType.HEARTBEAT, handler)

    msg = Message(
        msg_type=MessageType.HEARTBEAT,
        sender="agent-1",
        recipient="coordinator",
    )
    await bus.publish(msg)

    assert len(received) == 1
    assert received[0].sender == "agent-1"


@pytest.mark.asyncio
async def test_unsubscribe():
    bus = MessageBus()
    received = []

    async def handler(msg: Message) -> None:
        received.append(msg)

    bus.subscribe(MessageType.HEARTBEAT, handler)
    bus.unsubscribe(MessageType.HEARTBEAT, handler)

    await bus.publish(Message(
        msg_type=MessageType.HEARTBEAT,
        sender="a",
        recipient="b",
    ))
    assert len(received) == 0


@pytest.mark.asyncio
async def test_multiple_subscribers():
    bus = MessageBus()
    counts = {"a": 0, "b": 0}

    async def handler_a(msg: Message) -> None:
        counts["a"] += 1

    async def handler_b(msg: Message) -> None:
        counts["b"] += 1

    bus.subscribe(MessageType.SHUTDOWN, handler_a)
    bus.subscribe(MessageType.SHUTDOWN, handler_b)

    await bus.publish(Message(
        msg_type=MessageType.SHUTDOWN,
        sender="coord",
        recipient="*",
    ))

    assert counts["a"] == 1
    assert counts["b"] == 1


@pytest.mark.asyncio
async def test_request_response():
    bus = MessageBus()

    async def echo_handler(msg: Message) -> None:
        response = Message(
            msg_type=MessageType.PEER_RESPONSE,
            sender="responder",
            recipient=msg.sender,
            payload={"echo": msg.payload},
            correlation_id=msg.correlation_id,
        )
        await bus.publish(response)

    bus.subscribe(MessageType.PEER_REQUEST, echo_handler)

    req = Message(
        msg_type=MessageType.PEER_REQUEST,
        sender="requester",
        recipient="responder",
        payload={"data": "hello"},
    )
    resp = await bus.request(req, timeout=5.0)

    assert resp is not None
    assert resp.payload["echo"]["data"] == "hello"
