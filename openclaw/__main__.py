"""CLI entry point for OpenClaw.

Usage:
    python -m openclaw start          # Start the coordinator
    python -m openclaw status         # Print system status
    python -m openclaw kill [reason]  # Engage the kill switch
    python -m openclaw revive         # Disengage the kill switch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .coordinator.kill_switch import KillSwitch


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(prog="openclaw", description="OpenClaw Agent System")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("/tmp/openclaw"),
        help="Directory for persistent state (default: /tmp/openclaw)",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start the coordinator")
    sub.add_parser("status", help="Print system status")

    kill_parser = sub.add_parser("kill", help="Engage the master kill switch")
    kill_parser.add_argument("reason", nargs="?", default="manual CLI kill")
    kill_parser.add_argument("--force", action="store_true")

    sub.add_parser("revive", help="Disengage the kill switch")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "start":
        _run_coordinator(args.state_dir)
    elif args.command == "status":
        _print_status(args.state_dir)
    elif args.command == "kill":
        asyncio.run(_kill(args.state_dir, args.reason, args.force))
    elif args.command == "revive":
        asyncio.run(_revive(args.state_dir))


def _run_coordinator(state_dir: Path) -> None:
    from .coordinator.master import MasterCoordinator
    from .monitoring.dashboard import Dashboard

    async def _main() -> None:
        coord = MasterCoordinator(state_dir=state_dir)
        dashboard = Dashboard(coord)
        await coord.start()

        print(dashboard.render())
        print("\nCoordinator running. Press Ctrl+C to stop.\n")

        try:
            while True:
                await asyncio.sleep(10)
                print(dashboard.render())
        except KeyboardInterrupt:
            print("\nShutting down...")
            await coord.stop()

    asyncio.run(_main())


def _print_status(state_dir: Path) -> None:
    ks_file = state_dir / "kill_switch.json"
    if not ks_file.exists():
        print("No state found. System has not been started yet.")
        return
    data = json.loads(ks_file.read_text())
    print(json.dumps(data, indent=2))


async def _kill(state_dir: Path, reason: str, force: bool) -> None:
    ks = KillSwitch(
        state_file=state_dir / "kill_switch.json",
        grace_period_seconds=0.0 if force else 30.0,
    )
    await ks.engage(reason=reason, triggered_by="cli", force=force)
    print(f"Kill switch engaged: {ks.state.value}")


async def _revive(state_dir: Path) -> None:
    ks = KillSwitch(state_file=state_dir / "kill_switch.json")
    await ks.disengage(triggered_by="cli")
    print(f"System state: {ks.state.value}")


if __name__ == "__main__":
    main()
