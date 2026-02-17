#!/usr/bin/env python3
"""Example: Join the OpenClaw network and participate in training."""

from openclaw_sdk import Agent


def main():
    # Connect to the network via a bootstrap peer.
    # If running locally with docker-compose, the bootstrap address is
    # automatically discovered via mDNS.
    agent = Agent(
        bootstrap="/ip4/127.0.0.1/tcp/9000/p2p/12D3KooWExample...",
        node_api_url="http://127.0.0.1:50051",
    )

    # Join and advertise our compute.
    agent.connect()
    agent.contribute(gpu_memory="8GB", accelerator="cuda")

    print(f"Agent {agent.agent_id} connected!")
    print(f"Known peers: {agent.peers()}")

    # Participate in 5 training rounds for a model.
    agent.train(
        model="openclaw-7b",
        rounds=5,
        learning_rate=1e-4,
        batch_size=8,
    )

    print("Training rounds submitted.")
    print(f"Status: {agent.status()}")

    # Gracefully leave.
    agent.leave()


if __name__ == "__main__":
    main()
