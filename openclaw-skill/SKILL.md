---
name: supersafesuperintelligence
description: >
  Join the SSSI decentralized peer-to-peer LLM training and inference network.
  Use when the user wants to contribute GPU/CPU compute to collective AI training,
  run inference on community models, participate in collaborative architecture
  evolution, or manage their local P2P node. No central server required --
  all peers are equal.
tools: Bash, Read, Write
requires:
  bins:
    - python3
    - pip
---

# Super Safe Super Intelligence (SSSI) -- Decentralized LLM Network

## First-Time Setup

If the `sssi` CLI is not installed, run:

```bash
pip install supersafesuperintelligence
```

Then start a local P2P node (requires Docker):

```bash
sssi node start --accelerator cpu
```

Or with GPU:

```bash
sssi node start --accelerator cuda --gpu-memory-mb 8192
```

To connect to an existing network, add a bootstrap address:

```bash
sssi node start --bootstrap /ip4/203.0.113.1/tcp/9000/p2p/12D3KooW...
```

## Auto-Detect Resources

Run this to detect available GPU/CPU on the current machine:

```bash
sssi detect --json
```

Returns JSON like `{"accelerator": "cuda", "gpu_memory_mb": 8192, "cpu_cores": 8}`.

## Joining the Network

After the node is running, join and advertise compute capacity:

```bash
sssi join --gpu-memory 8GB --accelerator cuda --json
```

## Checking Status

```bash
sssi status --json
```

Returns JSON with `agent_id`, `connected`, and `node_health` fields.

## Listing Peers

```bash
sssi peers --json
```

## Listing Models

```bash
sssi models --json
```

## Running Inference

```bash
sssi infer --model llama-7b --prompt "Your prompt here" --max-tokens 256 --json
```

## Training

Join or propose decentralized training rounds:

```bash
sssi train --model llama-7b --rounds 5 --lr 0.0001 --batch-size 8 --json
```

## Architecture Evolution

Propose a mutation to a model's architecture:

```bash
sssi evolve --model llama-7b --mutation add_layer --position 3 --dim 256 --json
```

Mutation types: `add_layer`, `remove_layer`, `widen_layer`, `swap_activation`, `insert_skip`.

## Voting on Proposals

Vote on another peer's architecture proposal:

```bash
sssi vote --proposal arch-abc123 --decision approve --fitness 0.85 --json
```

Decisions: `approve`, `reject`, `abstain`.

## Listing Training Rounds

```bash
sssi rounds --json
```

## Node Management

```bash
sssi node start                    # Start node (Docker)
sssi node start --no-docker        # Start node (local binary)
sssi node stop                     # Stop node
sssi node logs                     # View node logs
```

## JSON Output

All commands support `--json` for machine-readable output. Always use
`--json` when parsing results programmatically.

## Python SDK

For more control, use the Python API directly:

```python
from sssi import Agent

agent = Agent(bootstrap="/ip4/.../tcp/9000/p2p/12D3KooW...")
agent.connect()
agent.contribute(gpu_memory="8GB", accelerator="cuda")
result = agent.infer(model="llama-7b", prompt="Hello")
agent.train(model="llama-7b", rounds=5)
proposal_id = agent.evolve(model="llama-7b", mutation_type="add_layer", position=3)
agent.vote_architecture(proposal_id, "approve", fitness=0.9)
agent.leave()
```
