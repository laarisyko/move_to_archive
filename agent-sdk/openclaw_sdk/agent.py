"""Base agent class -- the main entry point for OpenClaw agents."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

from .network import NetworkClient
from .training import TrainingParticipant
from .inference import InferenceClient
from .architecture import ArchitectureEvolver

logger = logging.getLogger(__name__)


class Agent:
    """An OpenClaw agent that participates in the decentralized LLM network.

    Usage::

        from openclaw_sdk import Agent

        agent = Agent(bootstrap="/ip4/203.0.113.1/tcp/9000/p2p/QmPeer...")
        agent.contribute(gpu_memory="8GB")

        # Run inference
        result = agent.infer(model="llama-7b", prompt="Hello world")

        # Participate in training
        agent.train(model="llama-7b", rounds=10)

        # Gracefully leave
        agent.leave()
    """

    def __init__(
        self,
        bootstrap: Optional[str] = None,
        node_api_url: str = "http://127.0.0.1:50051",
        agent_id: Optional[str] = None,
    ):
        """Initialize an OpenClaw agent.

        Args:
            bootstrap: Multiaddress of a bootstrap peer. If provided, the SDK
                will start a local node and connect to the network. If not
                provided, assumes a node is already running locally.
            node_api_url: URL of the local node's API endpoint.
            agent_id: Unique identifier for this agent. Auto-generated if not set.
        """
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        self.bootstrap = bootstrap
        self.node_api_url = node_api_url
        self._connected = False

        self.network = NetworkClient(node_api_url)
        self.training = TrainingParticipant(self.network, self.agent_id)
        self.inference = InferenceClient(self.network)
        self.architecture = ArchitectureEvolver(self.network, self.agent_id)

        logger.info("Agent %s initialized (node: %s)", self.agent_id, node_api_url)

    def connect(self) -> "Agent":
        """Connect to the P2P network.

        If a bootstrap address was provided, this will instruct the local
        node to dial the bootstrap peer.
        """
        if self.bootstrap:
            self.network.dial(self.bootstrap)

        health = self.network.health()
        if health.get("status") == "ok":
            self._connected = True
            logger.info("Agent %s connected to network", self.agent_id)
        else:
            logger.warning("Agent %s: node health check failed: %s", self.agent_id, health)

        return self

    def contribute(self, gpu_memory: str = "0", accelerator: str = "cpu") -> "Agent":
        """Advertise this agent's compute capacity to the network.

        Args:
            gpu_memory: GPU memory available (e.g. "8GB", "16GB").
            accelerator: Type of accelerator ("cpu", "cuda", "rocm").
        """
        capacity = {
            "agent_id": self.agent_id,
            "gpu_memory": gpu_memory,
            "accelerator": accelerator,
            "status": "available",
        }
        self.network.publish("openclaw/heartbeat", capacity)
        logger.info(
            "Agent %s contributing: %s %s",
            self.agent_id,
            gpu_memory,
            accelerator,
        )
        return self

    def infer(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Run inference on a model via the decentralized network.

        Args:
            model: Model identifier (e.g. "llama-7b").
            prompt: Input prompt.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Generated text.
        """
        return self.inference.infer(
            model_id=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def infer_async(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Async inference."""
        return await self.inference.infer_async(
            model_id=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def train(
        self,
        model: str,
        rounds: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
    ):
        """Participate in decentralized training rounds.

        Args:
            model: Model to train.
            rounds: Number of training rounds to participate in.
            learning_rate: Learning rate.
            batch_size: Batch size per step.
        """
        self.training.join_training(
            model_id=model,
            num_rounds=rounds,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    def evolve(
        self,
        model: str,
        mutation_type: str,
        position: int = 0,
        **kwargs,
    ) -> str:
        """Propose an architecture mutation for collaborative evolution.

        Args:
            model: Model to evolve.
            mutation_type: "add_layer", "remove_layer", "widen_layer",
                "swap_activation", or "insert_skip".
            position: Layer position for the mutation.
            **kwargs: Additional mutation parameters (new_output_dim,
                new_activation, layer_type, input_dim, output_dim).

        Returns:
            The proposal_id.
        """
        return self.architecture.propose_mutation(
            model_id=model,
            mutation_type=mutation_type,
            position=position,
            **kwargs,
        )

    def vote_architecture(
        self,
        proposal_id: str,
        decision: str,
        fitness: float = 0.0,
    ):
        """Vote on an architecture proposal from another peer.

        Args:
            proposal_id: The proposal to vote on.
            decision: "approve", "reject", or "abstain".
            fitness: Locally measured fitness score.
        """
        self.architecture.vote(proposal_id, decision, fitness)

    def peers(self) -> list:
        """List known peers in the network."""
        return self.network.peers()

    def status(self) -> dict:
        """Get current agent and network status."""
        health = self.network.health()
        return {
            "agent_id": self.agent_id,
            "connected": self._connected,
            "node_health": health,
        }

    def leave(self):
        """Gracefully leave the network."""
        logger.info("Agent %s leaving network", self.agent_id)
        self._connected = False

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.leave()

    def __repr__(self):
        return f"Agent(id={self.agent_id}, connected={self._connected})"
