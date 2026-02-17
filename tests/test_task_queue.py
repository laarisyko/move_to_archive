"""Tests for the task queue."""

import pytest

from openclaw.coordinator.task_queue import TaskQueue, Task, TaskStatus


@pytest.mark.asyncio
async def test_submit_and_claim():
    q = TaskQueue()
    task = Task(name="test_task", payload={"key": "value"})
    tid = await q.submit(task)

    assert q.pending_count == 1

    claimed = await q.claim("agent-1")
    assert claimed is not None
    assert claimed.task_id == tid
    assert claimed.status == TaskStatus.ASSIGNED
    assert claimed.assigned_to == "agent-1"


@pytest.mark.asyncio
async def test_complete_task():
    q = TaskQueue()
    task = Task(name="t", payload={})
    tid = await q.submit(task)
    await q.claim("a1")
    await q.complete(tid, result={"done": True})

    t = q.get_task(tid)
    assert t.status == TaskStatus.COMPLETED
    assert t.result == {"done": True}


@pytest.mark.asyncio
async def test_fail_task():
    q = TaskQueue()
    task = Task(name="t", payload={})
    tid = await q.submit(task)
    await q.claim("a1")
    await q.fail(tid, "something broke")

    t = q.get_task(tid)
    assert t.status == TaskStatus.FAILED
    assert t.error == "something broke"


@pytest.mark.asyncio
async def test_cancel_all():
    q = TaskQueue()
    await q.submit(Task(name="t1", payload={}))
    await q.submit(Task(name="t2", payload={}))
    await q.submit(Task(name="t3", payload={}))

    count = await q.cancel_all()
    assert count == 3


@pytest.mark.asyncio
async def test_priority_ordering():
    q = TaskQueue()
    low = Task(name="low", payload={})
    high = Task(name="high", payload={})

    await q.submit(low, priority=10)
    await q.submit(high, priority=1)

    first = await q.claim("a1")
    assert first.name == "high"
