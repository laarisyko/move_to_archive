"""Text-based dashboard for system status."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..coordinator.master import MasterCoordinator


class Dashboard:
    """Renders a text snapshot of system state.

    Designed for CLI or logging output. Plug a web framework on top
    for a real UI.
    """

    def __init__(self, coordinator: MasterCoordinator) -> None:
        self._coord = coordinator

    def render(self) -> str:
        status = self._coord.status()
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  OpenClaw System Dashboard")
        lines.append("=" * 60)
        lines.append("")

        # Kill switch
        ks_state = status["kill_switch"]
        indicator = {
            "active": "[LIVE]",
            "draining": "[DRAINING]",
            "killed": "[KILLED]",
        }.get(ks_state, "[???]")
        lines.append(f"  Kill Switch:   {indicator} {ks_state}")
        lines.append(f"  Tasks Pending: {status['tasks_pending']}")
        lines.append("")

        # Agents
        agents = status["agents"]
        lines.append(f"  Agents ({len(agents)}):")
        if not agents:
            lines.append("    (none)")
        for aid, info in agents.items():
            hb_ago = time.time() - info["last_heartbeat"]
            lines.append(
                f"    {aid:20s}  status={info['status']:6s}  "
                f"heartbeat={hb_ago:.0f}s ago  "
                f"task={info['current_task'] or '-'}"
            )
        lines.append("")

        # Health
        health = status["health"]
        lines.append(f"  Health ({len(health)}):")
        if not health:
            lines.append("    (none)")
        for aid, h in health.items():
            alive = "ALIVE" if h["is_alive"] else "DEAD"
            lines.append(
                f"    {aid:20s}  {alive:5s}  "
                f"beats={h['heartbeat_count']}  "
                f"missed={h['missed_heartbeats']}"
            )

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
