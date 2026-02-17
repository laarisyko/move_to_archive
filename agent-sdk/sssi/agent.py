"""Base agent class -- the main entry point for SSSI agents."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from .network import NetworkClient
from .training import TrainingParticipant
from .inference import InferenceClient
from .architecture import ArchitectureEvolver

logger = logging.getLogger(__name__)


class Agent:
    """An SSSI agent that participates in the decentralized LLM network.

    Usage::

        from sssi import Agent

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
        """Connect to the P2P network."""
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
        """Advertise this agent's compute capacity to the network."""
        capacity = {
            "agent_id": self.agent_id,
            "gpu_memory": gpu_memory,
            "accelerator": accelerator,
            "status": "available",
        }
        self.network.publish("sssi/heartbeat", capacity)
        logger.info("Agent %s contributing: %s %s", self.agent_id, gpu_memory, accelerator)
        return self

    def infer(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """Run inference on a model via the decentralized network."""
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
        """Participate in decentralized training rounds."""
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
        """Vote on an architecture proposal from another peer."""
        self.architecture.vote(proposal_id, decision, fitness)

    def peers(self) -> list:
        """List known peers in the network."""
        return self.network.peers()

    def models(self) -> list:
        """List available models on the network."""
        return self.inference.list_models()

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
