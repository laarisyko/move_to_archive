#!/usr/bin/env python3
"""Example: Run inference via the OpenClaw decentralized network."""

from openclaw_sdk import Agent


def main():
    # Connect to a running node.
    agent = Agent(node_api_url="http://127.0.0.1:50051")

    # Run inference -- the network routes the request through the
    # pipeline of peers holding the model shards.
    result = agent.infer(
        model="openclaw-7b",
        prompt="Explain decentralized machine learning in simple terms.",
        max_tokens=512,
        temperature=0.7,
    )
    print("Inference result:")
    print(result)


if __name__ == "__main__":
    main()
