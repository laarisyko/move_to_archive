"""Health monitoring for distributed agents."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AgentHealth:
    """Health state for a single agent."""

    agent_id: str
    last_heartbeat: float = field(default_factory=time.time)
    heartbeat_count: int = 0
    missed_heartbeats: int = 0
    is_alive: bool = True


class HealthMonitor:
    """Tracks agent liveness via heartbeats.

    If an agent fails to send a heartbeat within ``heartbeat_timeout``
    seconds, it is considered dead and flagged for removal.
    """

    def __init__(
        self,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 15.0,
    ) -> None:
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._agents: dict[str, AgentHealth] = {}

    def register(self, agent_id: str) -> None:
        self._agents[agent_id] = AgentHealth(agent_id=agent_id)

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def record_heartbeat(self, agent_id: str) -> None:
        health = self._agents.get(agent_id)
        if health:
            health.last_heartbeat = time.time()
            health.heartbeat_count += 1
            health.is_alive = True

    def check_all(self) -> list[str]:
        """Check all agents and return IDs of dead ones."""
        now = time.time()
        dead: list[str] = []
        for agent_id, health in self._agents.items():
            elapsed = now - health.last_heartbeat
            if elapsed > self.heartbeat_timeout:
                health.is_alive = False
                health.missed_heartbeats += 1
                dead.append(agent_id)
                logger.warning(
                    "Agent %s: no heartbeat for %.1fs (missed %d)",
                    agent_id,
                    elapsed,
                    health.missed_heartbeats,
                )
        return dead

    def get_health(self, agent_id: str) -> AgentHealth | None:
        return self._agents.get(agent_id)

    def summary(self) -> dict[str, dict]:
        return {
            agent_id: {
                "is_alive": h.is_alive,
                "last_heartbeat": h.last_heartbeat,
                "heartbeat_count": h.heartbeat_count,
                "missed_heartbeats": h.missed_heartbeats,
            }
            for agent_id, h in self._agents.items()
        }
