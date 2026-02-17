"""Training participation API -- lets agents join decentralized training rounds."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from .network import NetworkClient

logger = logging.getLogger(__name__)


class TrainingParticipant:
    """Manages this agent's participation in decentralized training rounds."""

    def __init__(self, network: NetworkClient, agent_id: str):
        self.network = network
        self.agent_id = agent_id
        self._active_round: Optional[str] = None

    def propose_round(
        self,
        model_id: str,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        num_steps: int = 100,
    ) -> str:
        """Propose a new training round to the network.

        Returns:
            The round_id of the proposed round.
        """
        round_id = f"round-{uuid.uuid4().hex[:12]}"
        proposal = {
            "type": "proposal",
            "round_id": round_id,
            "model_id": model_id,
            "proposer": self.agent_id,
            "hyper_params": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_steps": num_steps,
            },
        }
        self.network.publish("sssi/training", proposal)
        self._active_round = round_id
        logger.info("Proposed training round %s for model %s", round_id, model_id)
        return round_id

    def join_round(self, round_id: str):
        """Join an existing training round."""
        join_msg = {
            "type": "join",
            "round_id": round_id,
            "peer_id": self.agent_id,
        }
        self.network.publish("sssi/training", join_msg)
        self._active_round = round_id
        logger.info("Joined training round %s", round_id)

    def join_training(
        self,
        model_id: str,
        num_rounds: int = 1,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
    ):
        """Convenience: propose and participate in training rounds."""
        for i in range(num_rounds):
            round_id = self.propose_round(
                model_id=model_id,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            logger.info("Training round %d/%d: %s", i + 1, num_rounds, round_id)

    def announce_gradient_ready(self, round_id: str, merkle_root: str):
        """Announce that local gradients are ready for aggregation."""
        msg = {
            "type": "gradient_ready",
            "round_id": round_id,
            "peer_id": self.agent_id,
            "merkle_root": merkle_root,
        }
        self.network.publish("sssi/gradient", msg)

    def announce_checkpoint(self, round_id: str, weights_merkle_root: str, cid: str = ""):
        """Announce a completed checkpoint after training."""
        msg = {
            "type": "checkpoint",
            "round_id": round_id,
            "peer_id": self.agent_id,
            "weights_merkle_root": weights_merkle_root,
            "checkpoint_cid": cid,
        }
        self.network.publish("sssi/checkpoint", msg)

    @property
    def active_round(self) -> Optional[str]:
        return self._active_round
