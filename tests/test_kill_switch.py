"""Tests for the master kill switch."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from openclaw.coordinator.kill_switch import KillSwitch, KillSwitchState


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    return tmp_path / "ks.json"


@pytest.mark.asyncio
async def test_initial_state():
    ks = KillSwitch()
    assert ks.state == KillSwitchState.ACTIVE
    assert ks.is_alive is True


@pytest.mark.asyncio
async def test_engage_and_disengage():
    ks = KillSwitch(grace_period_seconds=0.1)
    await ks.engage(reason="test", triggered_by="pytest")
    # After grace period it should be KILLED
    await asyncio.sleep(0.2)
    assert ks.state == KillSwitchState.KILLED
    assert ks.is_alive is False

    await ks.disengage(triggered_by="pytest")
    assert ks.state == KillSwitchState.ACTIVE
    assert ks.is_alive is True


@pytest.mark.asyncio
async def test_force_engage():
    ks = KillSwitch(grace_period_seconds=60.0)
    await ks.engage(reason="force test", triggered_by="pytest", force=True)
    assert ks.state == KillSwitchState.KILLED


@pytest.mark.asyncio
async def test_state_persistence(state_file: Path):
    ks = KillSwitch(state_file=state_file, grace_period_seconds=0.0)
    await ks.engage(reason="persist test", triggered_by="pytest", force=True)

    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["state"] == "killed"

    # New instance should load persisted state
    ks2 = KillSwitch(state_file=state_file)
    assert ks2.state == KillSwitchState.KILLED


@pytest.mark.asyncio
async def test_hooks_fire_on_engage():
    called = False

    async def my_hook(ks: KillSwitch) -> None:
        nonlocal called
        called = True

    ks = KillSwitch(grace_period_seconds=0.1)
    ks.on_shutdown(my_hook)
    await ks.engage(reason="hook test", triggered_by="pytest")
    assert called is True


@pytest.mark.asyncio
async def test_history_tracking():
    ks = KillSwitch(grace_period_seconds=0.0)
    await ks.engage(reason="r1", triggered_by="u1", force=True)
    await ks.disengage(triggered_by="u2")

    assert len(ks.history) >= 2
    assert ks.history[0].to_state == KillSwitchState.KILLED
    assert ks.history[-1].to_state == KillSwitchState.ACTIVE
