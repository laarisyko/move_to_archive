"""Tests for agent workers."""

import asyncio

import pytest

from openclaw.agents.worker import AgentWorker
from openclaw.agents.capabilities import Capability
from openclaw.comms.message_bus import MessageBus, Message, MessageType


@pytest.mark.asyncio
async def test_worker_registration():
    bus = MessageBus()
    registered = []

    async def catch_register(msg: Message) -> None:
        registered.append(msg)

    bus.subscribe(MessageType.REGISTER, catch_register)

    worker = AgentWorker(
        message_bus=bus,
        capabilities=[Capability(name="echo", handler=lambda p: p)],
    )
    await worker.start()

    assert len(registered) == 1
    assert registered[0].payload["capabilities"] == ["echo"]

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_processes_task():
    bus = MessageBus()
    results = []

    async def handle_add(payload: dict) -> dict:
        return {"sum": payload["a"] + payload["b"]}

    async def catch_result(msg: Message) -> None:
        results.append(msg)

    bus.subscribe(MessageType.TASK_RESULT, catch_result)

    worker = AgentWorker(
        message_bus=bus,
        capabilities=[Capability(name="add", handler=handle_add)],
        agent_id="worker-test",
    )
    await worker.start()

    # Simulate task assignment
    await bus.publish(Message(
        msg_type=MessageType.TASK_ASSIGN,
        sender="coordinator",
        recipient="worker-test",
        payload={
            "task_id": "t1",
            "name": "add",
            "payload": {"a": 2, "b": 3},
        },
    ))

    await asyncio.sleep(0.1)

    assert len(results) == 1
    assert results[0].payload["success"] is True
    assert results[0].payload["result"]["sum"] == 5

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_handles_shutdown():
    bus = MessageBus()
    worker = AgentWorker(message_bus=bus, agent_id="w1")
    await worker.start()
    assert worker._running is True

    await bus.publish(Message(
        msg_type=MessageType.SHUTDOWN,
        sender="coordinator",
        recipient="*",
        payload={"reason": "test"},
    ))

    await asyncio.sleep(0.1)
    assert worker._running is False
