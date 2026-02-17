"""Example: spin up a coordinator with two workers, submit tasks, then kill."""

import asyncio
import logging

from openclaw.coordinator.master import MasterCoordinator
from openclaw.agents.worker import AgentWorker
from openclaw.agents.capabilities import Capability
from openclaw.monitoring.dashboard import Dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ── capability handlers ─────────────────────────────────────────────

async def handle_summarize(payload: dict) -> dict:
    text = payload.get("text", "")
    return {"summary": text[:80] + ("..." if len(text) > 80 else "")}


async def handle_classify(payload: dict) -> dict:
    text = payload.get("text", "")
    # Toy classifier
    if "error" in text.lower():
        return {"label": "error"}
    return {"label": "info"}


# ── main ─────────────────────────────────────────────────────────────

async def main() -> None:
    coord = MasterCoordinator(grace_period=5.0)
    dashboard = Dashboard(coord)

    await coord.start()

    # Create two workers with different capabilities
    worker_a = AgentWorker(
        message_bus=coord.message_bus,
        capabilities=[Capability(name="summarize", handler=handle_summarize)],
    )
    worker_b = AgentWorker(
        message_bus=coord.message_bus,
        capabilities=[Capability(name="classify", handler=handle_classify)],
    )

    await worker_a.start()
    await worker_b.start()

    # Submit some tasks
    t1 = await coord.submit_task("summarize", {"text": "OpenClaw is a distributed multi-agent system with a master kill switch for safe operation."})
    t2 = await coord.submit_task("classify", {"text": "Error: connection timed out"})
    t3 = await coord.submit_task("summarize", {"text": "Agents coordinate through an async message bus."})

    # Let tasks process
    await asyncio.sleep(2)

    print("\n" + dashboard.render())

    # Print task results
    for tid in [t1, t2, t3]:
        task = coord.task_queue.get_task(tid)
        if task:
            print(f"  Task {task.task_id} ({task.name}): {task.status.value} → {task.result}")

    # Now pull the kill switch
    print("\n>>> Engaging kill switch...")
    await coord.kill_switch.engage(reason="demo complete", triggered_by="example")

    print("\n" + dashboard.render())

    # Re-enable
    print(">>> Disengaging kill switch...")
    await coord.kill_switch.disengage(triggered_by="example")

    print(f"\nFinal state: {coord.kill_switch.state.value}")


if __name__ == "__main__":
    asyncio.run(main())
