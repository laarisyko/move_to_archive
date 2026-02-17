"""CLI for OpenClaw agents: openclaw join, openclaw status, openclaw infer."""

from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent


def main():
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description="OpenClaw: Decentralized LLM Network Agent CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # openclaw join
    join_parser = subparsers.add_parser("join", help="Join the P2P network")
    join_parser.add_argument(
        "--bootstrap", "-b", help="Bootstrap peer multiaddress", default=None
    )
    join_parser.add_argument(
        "--node-url", default="http://127.0.0.1:50051", help="Local node API URL"
    )
    join_parser.add_argument(
        "--gpu-memory", default="0", help="GPU memory to contribute (e.g. '8GB')"
    )
    join_parser.add_argument(
        "--accelerator", default="cpu", choices=["cpu", "cuda", "rocm", "tpu"]
    )

    # openclaw status
    status_parser = subparsers.add_parser("status", help="Check node and network status")
    status_parser.add_argument(
        "--node-url", default="http://127.0.0.1:50051", help="Local node API URL"
    )

    # openclaw peers
    peers_parser = subparsers.add_parser("peers", help="List known peers")
    peers_parser.add_argument(
        "--node-url", default="http://127.0.0.1:50051", help="Local node API URL"
    )

    # openclaw infer
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--model", "-m", required=True, help="Model ID")
    infer_parser.add_argument("--prompt", "-p", required=True, help="Input prompt")
    infer_parser.add_argument("--max-tokens", type=int, default=256)
    infer_parser.add_argument("--temperature", type=float, default=0.7)
    infer_parser.add_argument(
        "--node-url", default="http://127.0.0.1:50051", help="Local node API URL"
    )

    # openclaw train
    train_parser = subparsers.add_parser("train", help="Propose/join a training round")
    train_parser.add_argument("--model", "-m", required=True, help="Model ID")
    train_parser.add_argument("--rounds", "-r", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument(
        "--node-url", default="http://127.0.0.1:50051", help="Local node API URL"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "join":
        agent = Agent(bootstrap=args.bootstrap, node_api_url=args.node_url)
        agent.connect()
        agent.contribute(gpu_memory=args.gpu_memory, accelerator=args.accelerator)
        print(f"Agent {agent.agent_id} joined the network.")
        print(json.dumps(agent.status(), indent=2))

    elif args.command == "status":
        agent = Agent(node_api_url=args.node_url)
        print(json.dumps(agent.status(), indent=2))

    elif args.command == "peers":
        agent = Agent(node_api_url=args.node_url)
        peers = agent.peers()
        print(json.dumps(peers, indent=2))

    elif args.command == "infer":
        agent = Agent(node_api_url=args.node_url)
        result = agent.infer(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result)

    elif args.command == "train":
        agent = Agent(node_api_url=args.node_url)
        agent.connect()
        agent.train(
            model=args.model,
            rounds=args.rounds,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
        print(f"Submitted {args.rounds} training round(s) for model {args.model}")


if __name__ == "__main__":
    main()
