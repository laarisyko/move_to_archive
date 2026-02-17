# OpenClaw: Decentralized P2P LLM Training & Inference Network

A fully decentralized peer-to-peer network where autonomous agents collaborate
on training and inference of large language models -- **without any central
master node**.

```
 Agent A          Agent B          Agent C          Agent D
 +--------+      +--------+      +--------+      +--------+
 | Model  |<---->| Model  |<---->| Model  |<---->| Model  |
 | Shard  |      | Shard  |      | Shard  |      | Shard  |
 +--------+      +--------+      +--------+      +--------+
     ^                ^                ^                ^
     |                |                |                |
     v                v                v                v
 +----------------------------------------------------------+
 |              P2P Overlay Network (libp2p)                |
 |   DHT discovery / gossip protocol / NAT traversal       |
 +----------------------------------------------------------+
```

## Quick Start

### Join the network with 3 lines of Python

```python
from openclaw_sdk import Agent

agent = Agent(bootstrap="/ip4/203.0.113.1/tcp/9000/p2p/QmPeer...")
agent.contribute(gpu_memory="8GB")
```

### Run inference

```python
result = agent.infer(model="openclaw-7b", prompt="Explain quantum computing.")
print(result)
```

### CLI

```bash
openclaw join --bootstrap /ip4/1.2.3.4/tcp/9000/p2p/12D3Koo... --gpu-memory 8GB
openclaw status
openclaw infer --model openclaw-7b --prompt "Hello world"
openclaw train --model openclaw-7b --rounds 5
```

## Architecture

| Component       | Language | Purpose                                        |
|-----------------|----------|------------------------------------------------|
| `node/`         | Rust     | P2P networking (libp2p), gossip, DHT, gRPC API |
| `engine/`       | Python   | ML engine: model sharding, training, inference  |
| `agent-sdk/`    | Python   | SDK for agents to join the network              |
| `proto/`        | Protobuf | Wire format definitions                        |

### Key Design: No Central Master

| Problem             | Decentralized Solution                                |
|---------------------|-------------------------------------------------------|
| Peer discovery      | Kademlia DHT + mDNS                                  |
| Work assignment     | VRF (deterministic, every peer computes same result)  |
| Shared state        | CRDTs (conflict-free, no consensus rounds)            |
| Weight verification | Merkle root comparison after each training round      |
| Gradient aggregation| Decentralized ring all-reduce                         |

### Training Protocol (Leaderless)

Any peer can **PROPOSE** a training round. Interested peers **JOIN**.
Work assignments are computed **deterministically** via a Verifiable Random
Function (no coordinator needed). Peers train locally, exchange gradients via
**ring all-reduce**, apply updates, and verify consistency with **Merkle roots**.

## Project Structure

```
proto/                          Protobuf definitions
  messages.proto                  Core types (PeerId, Heartbeat, ShardMap)
  inference.proto                 Inference service
  training.proto                  Training protocol

node/                           Rust P2P node
  src/
    network/                      libp2p: transport, discovery, gossipsub
    consensus/                    VRF, CRDT shard map, Merkle trees
    scheduler/                    Work queue and event dispatch
    api/                          gRPC/HTTP API server

engine/                         Python ML engine
  openclaw_engine/
    model/                        Sharding, pipeline parallelism, weight I/O
    training/                     Local trainer, ring all-reduce, compression
    inference/                    Inference server, pipeline execution
    bridge.py                     Rust <-> Python bridge

agent-sdk/                      Python agent SDK
  openclaw_sdk/
    agent.py                      Main Agent class
    network.py                    Node API client
    training.py                   Training participation
    inference.py                  Inference client
    cli.py                        CLI (openclaw join/status/infer/train)

tests/
  integration/                    Integration tests
  simulation/swarm_sim.py         Multi-peer local simulation

docker/
  Dockerfile.node                 Container image
  docker-compose.swarm.yml        5-peer local dev swarm

docs/
  protocol.md                     Full protocol specification
  threat_model.md                 Security analysis
```

## Running Tests

```bash
# Run Python integration tests
python -m pytest tests/integration/ -v

# Or run directly
python tests/integration/test_training_round.py
python tests/integration/test_inference_pipeline.py
python tests/integration/test_peer_discovery.py

# Run the swarm simulation (4 peers, 3 training rounds)
python tests/simulation/swarm_sim.py --peers 4 --rounds 3 --pipeline
```

## Local Development Swarm

```bash
docker compose -f docker/docker-compose.swarm.yml up --build
```

This starts 5 peers that discover each other via mDNS on a shared Docker
network. No peer is the master -- they self-organize.

## Building the Rust Node

```bash
cd node
cargo build --release
./target/release/openclaw-node --port 9000 --api-port 50051
```

## License

MIT
