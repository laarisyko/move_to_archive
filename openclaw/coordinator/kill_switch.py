"""Master Kill Switch — the central safety mechanism.

The kill switch is a global, authoritative control that can immediately
halt all agent activity across the entire distributed system. It is
designed so that:

  1. Any authorized operator can trigger it.
  2. Once triggered, ALL agents must stop accepting new work and drain
     their current tasks within a configurable grace period.
  3. Agents that fail to stop within the grace period are force-terminated.
  4. The switch state is persisted so that restarted agents cannot resume
     work while the switch is engaged.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class KillSwitchState(enum.Enum):
    """Possible states of the master kill switch."""

    ACTIVE = "active"          # System is running normally
    DRAINING = "draining"      # Graceful shutdown in progress
    KILLED = "killed"          # Full stop — nothing should run


@dataclass
class KillSwitchEvent:
    """Record of a kill switch state transition."""

    from_state: KillSwitchState
    to_state: KillSwitchState
    reason: str
    triggered_by: str
    timestamp: float = field(default_factory=time.time)


ShutdownHook = Callable[["KillSwitch"], Awaitable[None]]


class KillSwitch:
    """Master kill switch for the entire distributed agent system.

    Usage::

        ks = KillSwitch(state_file=Path("/var/run/openclaw/kill_switch.json"))

        # Register callbacks that fire when the switch is pulled
        ks.on_shutdown(my_cleanup_callback)

        # Pull the switch
        await ks.engage(reason="maintenance window", triggered_by="ops-team")

        # Later, re-enable the system
        await ks.disengage(triggered_by="ops-team")
    """

    def __init__(
        self,
        state_file: Path | None = None,
        grace_period_seconds: float = 30.0,
    ) -> None:
        self._state = KillSwitchState.ACTIVE
        self._state_file = state_file
        self._grace_period = grace_period_seconds
        self._hooks: list[ShutdownHook] = []
        self._history: list[KillSwitchEvent] = []
        self._lock = asyncio.Lock()

        # Restore persisted state if available
        if state_file and state_file.exists():
            self._load_state()

    # -- public queries -------------------------------------------------------

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_alive(self) -> bool:
        return self._state == KillSwitchState.ACTIVE

    @property
    def history(self) -> list[KillSwitchEvent]:
        return list(self._history)

    # -- hooks ----------------------------------------------------------------

    def on_shutdown(self, hook: ShutdownHook) -> None:
        """Register an async callback invoked when the switch is engaged."""
        self._hooks.append(hook)

    # -- state transitions ----------------------------------------------------

    async def engage(
        self,
        reason: str = "manual",
        triggered_by: str = "operator",
        force: bool = False,
    ) -> None:
        """Pull the kill switch — begin graceful (or forced) shutdown."""
        async with self._lock:
            if self._state == KillSwitchState.KILLED and not force:
                logger.info("Kill switch already engaged; nothing to do.")
                return

            prev = self._state

            if force:
                self._transition(KillSwitchState.KILLED, reason, triggered_by)
            else:
                # Graceful: go to DRAINING first
                self._transition(KillSwitchState.DRAINING, reason, triggered_by)

            self._persist_state()

        logger.warning(
            "KILL SWITCH ENGAGED (%s → %s) by %s: %s",
            prev.value,
            self._state.value,
            triggered_by,
            reason,
        )

        # Fire shutdown hooks (best-effort, with timeout)
        await self._fire_hooks()

        if not force:
            # Wait for the grace period then force-kill
            await asyncio.sleep(self._grace_period)
            async with self._lock:
                if self._state == KillSwitchState.DRAINING:
                    self._transition(
                        KillSwitchState.KILLED,
                        f"grace period ({self._grace_period}s) expired",
                        "system",
                    )
                    self._persist_state()

    async def disengage(self, triggered_by: str = "operator") -> None:
        """Re-enable the system after a kill switch event."""
        async with self._lock:
            if self._state == KillSwitchState.ACTIVE:
                logger.info("System already active; nothing to do.")
                return

            self._transition(
                KillSwitchState.ACTIVE,
                "system re-enabled",
                triggered_by,
            )
            self._persist_state()

        logger.info("System re-enabled by %s.", triggered_by)

    # -- internals ------------------------------------------------------------

    def _transition(
        self,
        to: KillSwitchState,
        reason: str,
        triggered_by: str,
    ) -> None:
        event = KillSwitchEvent(
            from_state=self._state,
            to_state=to,
            reason=reason,
            triggered_by=triggered_by,
        )
        self._history.append(event)
        self._state = to

    async def _fire_hooks(self) -> None:
        for hook in self._hooks:
            try:
                await asyncio.wait_for(hook(self), timeout=10.0)
            except Exception:
                logger.exception("Shutdown hook %s failed", hook)

    def _persist_state(self) -> None:
        if self._state_file is None:
            return
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": self._state.value,
            "updated_at": time.time(),
            "history": [
                {
                    "from": e.from_state.value,
                    "to": e.to_state.value,
                    "reason": e.reason,
                    "triggered_by": e.triggered_by,
                    "timestamp": e.timestamp,
                }
                for e in self._history[-50:]  # keep last 50 events
            ],
        }
        self._state_file.write_text(json.dumps(payload, indent=2))

    def _load_state(self) -> None:
        try:
            data = json.loads(self._state_file.read_text())
            self._state = KillSwitchState(data["state"])
            logger.info("Restored kill switch state: %s", self._state.value)
        except Exception:
            logger.exception("Failed to load kill switch state; defaulting to KILLED for safety")
            self._state = KillSwitchState.KILLED
