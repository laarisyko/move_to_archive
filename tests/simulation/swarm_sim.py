#!/usr/bin/env python3
"""Swarm simulation -- simulate N peers doing training and inference locally.

This simulates the full decentralized training lifecycle without needing
actual networking:
1. N peers each hold a shard of the model (data parallelism).
2. Each peer does a local training step.
3. Peers aggregate gradients via ring all-reduce.
4. All peers verify weight consistency via Merkle roots.
5. Each peer serves inference requests.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn

from openclaw_engine.model.shard import split_model, ModelShard, ShardConfig
from openclaw_engine.model.pipeline import PipelineExecutor
from openclaw_engine.training.trainer import LocalTrainer, TrainingConfig
from openclaw_engine.training.allreduce import RingAllReduce
from openclaw_engine.training.compression import TopKCompressor
from openclaw_engine.inference.server import InferenceServer, InferenceRequest


def make_model(n_layers: int = 8, hidden_dim: int = 64) -> nn.Module:
    wrapper = nn.Module()
    wrapper.layers = nn.ModuleList(
        [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(n_layers)]
    )
    return wrapper


def run_simulation(
    n_peers: int = 4,
    n_layers: int = 8,
    hidden_dim: int = 64,
    training_steps: int = 5,
    training_rounds: int = 3,
):
    print(f"=== OpenClaw Swarm Simulation ===")
    print(f"Peers: {n_peers}, Layers: {n_layers}, Hidden dim: {hidden_dim}")
    print(f"Training: {training_rounds} rounds x {training_steps} steps")
    print()

    # --- Phase 1: Model Creation & Sharding ---
    print("[1/5] Creating model and distributing shards...")
    model = make_model(n_layers, hidden_dim)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Data parallelism: each peer gets a full copy (same shard).
    peer_shards = [split_model(model, "sim-model", 1)[0] for _ in range(n_peers)]
    print(f"  Each peer holds {peer_shards[0].num_parameters():,} parameters")
    print()

    # --- Phase 2: Training Rounds ---
    config = TrainingConfig(
        learning_rate=1e-3,
        num_steps=training_steps,
        optimizer="adamw",
    )
    trainers = [LocalTrainer(shard, config) for shard in peer_shards]
    rings = RingAllReduce.local_ring(n_peers)
    compressor = TopKCompressor(ratio=0.1)

    for round_idx in range(training_rounds):
        round_start = time.monotonic()
        print(f"[2/5] Training round {round_idx + 1}/{training_rounds}...")

        # Each peer trains on different data.
        all_grads = []
        for peer_idx, trainer in enumerate(trainers):
            for step in range(training_steps):
                x = torch.randn(4, 8, hidden_dim) * (peer_idx + 1)
                metrics = trainer.train_step(x)

            grads = trainer.get_gradients()
            all_grads.append(grads)
            print(f"    Peer {peer_idx}: grad_norm={metrics['grad_norm']:.4f}")

        # Ring all-reduce.
        print(f"  [3/5] Ring all-reduce across {n_peers} peers...")
        aggregated = RingAllReduce.reduce_all(rings, all_grads)

        # Apply aggregated gradients.
        for i, trainer in enumerate(trainers):
            trainer.set_gradients(aggregated[i])
            trainer.apply_gradients()

        # Verify Merkle roots.
        roots = [shard.merkle_root().hex()[:16] for shard in peer_shards]
        consistent = len(set(roots)) == 1
        round_ms = (time.monotonic() - round_start) * 1000
        print(f"  [4/5] Merkle verification: {'CONSISTENT' if consistent else 'DIVERGENT'}")
        print(f"    Root: {roots[0]}...")
        print(f"    Round completed in {round_ms:.1f}ms")
        print()

    # --- Phase 3: Inference ---
    print("[5/5] Running inference on each peer...")
    for peer_idx, shard in enumerate(peer_shards):
        server = InferenceServer()
        server.register_shard("sim-model", shard)
        request = InferenceRequest(model_id="sim-model", prompt="Hello from simulation")
        response = server.infer(request)
        print(f"  Peer {peer_idx}: {response.text} (latency: {response.latency_ms:.2f}ms)")

    print()
    print("=== Simulation Complete ===")


def run_pipeline_simulation(n_peers: int = 4, n_layers: int = 8, hidden_dim: int = 64):
    """Simulate pipeline-parallel inference across peers."""
    print(f"\n=== Pipeline Parallelism Simulation ===")
    print(f"Peers: {n_peers}, Layers: {n_layers}")

    model = make_model(n_layers, hidden_dim)
    shards = split_model(model, "pipeline-model", n_peers)

    for i, shard in enumerate(shards):
        print(
            f"  Peer {i}: layers [{shard.config.layer_start}, {shard.config.layer_end})"
            f" ({shard.num_parameters():,} params)"
        )

    pipeline = PipelineExecutor.local(shards)
    x = torch.randn(1, 16, hidden_dim)

    start = time.monotonic()
    output = pipeline.forward(x)
    elapsed_ms = (time.monotonic() - start) * 1000

    print(f"\n  Input shape:  {list(x.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Pipeline latency: {elapsed_ms:.2f}ms")
    print(f"  Stages: {pipeline.num_stages}")
    print("=== Pipeline Simulation Complete ===\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw swarm simulation")
    parser.add_argument("--peers", type=int, default=4, help="Number of simulated peers")
    parser.add_argument("--layers", type=int, default=8, help="Number of model layers")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--steps", type=int, default=5, help="Training steps per round")
    parser.add_argument("--rounds", type=int, default=3, help="Number of training rounds")
    parser.add_argument(
        "--pipeline", action="store_true", help="Also run pipeline parallelism demo"
    )
    args = parser.parse_args()

    run_simulation(
        n_peers=args.peers,
        n_layers=args.layers,
        hidden_dim=args.hidden_dim,
        training_steps=args.steps,
        training_rounds=args.rounds,
    )

    if args.pipeline:
        run_pipeline_simulation(
            n_peers=args.peers,
            n_layers=args.layers,
            hidden_dim=args.hidden_dim,
        )
