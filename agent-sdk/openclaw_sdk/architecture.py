"""Architecture evolution API for OpenClaw agents.

Lets agents propose architectural mutations, vote on proposals from other
peers, and query the current genome of a model.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, List, Optional

from .network import NetworkClient

logger = logging.getLogger(__name__)


class ArchitectureEvolver:
    """Client for collaborative architecture evolution.

    Enables agents to participate in decentralized neural architecture search:
    - Propose mutations to model architectures
    - Receive and vote on proposals from other peers
    - Track the evolution history of a model
    """

    def __init__(self, network: NetworkClient, agent_id: str):
        self.network = network
        self.agent_id = agent_id
        self._proposals_sent: List[str] = []

    def propose_mutation(
        self,
        model_id: str,
        mutation_type: str,
        position: int = 0,
        new_output_dim: int = 0,
        new_activation: str = "",
        layer_type: str = "linear",
        input_dim: int = 0,
        output_dim: int = 0,
    ) -> str:
        """Propose an architecture mutation to the network.

        Args:
            model_id: Model to mutate.
            mutation_type: One of "add_layer", "remove_layer", "widen_layer",
                "swap_activation", "insert_skip".
            position: Layer position to apply the mutation.
            new_output_dim: For widen_layer mutations.
            new_activation: For swap_activation mutations.
            layer_type: For add_layer, the type of layer to add.
            input_dim: For add_layer, the input dimension.
            output_dim: For add_layer, the output dimension.

        Returns:
            The proposal_id.
        """
        proposal_id = f"arch-{uuid.uuid4().hex[:12]}"

        mutation = {"type": mutation_type, "position": position}
        if mutation_type == "add_layer":
            mutation["gene"] = {
                "layer_type": layer_type,
                "input_dim": input_dim,
                "output_dim": output_dim,
            }
        elif mutation_type == "widen_layer":
            mutation["new_output_dim"] = new_output_dim
        elif mutation_type == "swap_activation":
            mutation["new_activation"] = new_activation

        proposal = {
            "type": "architecture_proposal",
            "proposal_id": proposal_id,
            "proposer_id": self.agent_id,
            "model_id": model_id,
            "mutation": mutation,
        }

        self.network.publish("openclaw/architecture", proposal)
        self._proposals_sent.append(proposal_id)
        logger.info(
            "Proposed %s mutation for model %s (proposal: %s)",
            mutation_type,
            model_id,
            proposal_id,
        )
        return proposal_id

    def vote(
        self,
        proposal_id: str,
        decision: str,
        measured_fitness: float = 0.0,
    ):
        """Cast a vote on an architecture proposal.

        Args:
            proposal_id: The proposal to vote on.
            decision: "approve", "reject", or "abstain".
            measured_fitness: Fitness score measured locally (optional).
        """
        vote = {
            "type": "architecture_vote",
            "proposal_id": proposal_id,
            "voter_id": self.agent_id,
            "decision": decision,
            "measured_fitness": measured_fitness,
        }
        self.network.publish("openclaw/architecture", vote)
        logger.info("Voted %s on proposal %s", decision, proposal_id)

    def propose_add_layer(
        self,
        model_id: str,
        position: int,
        layer_type: str = "linear",
        dim: int = 256,
    ) -> str:
        """Convenience: propose adding a layer."""
        return self.propose_mutation(
            model_id=model_id,
            mutation_type="add_layer",
            position=position,
            layer_type=layer_type,
            input_dim=dim,
            output_dim=dim,
        )

    def propose_widen(
        self,
        model_id: str,
        position: int,
        new_dim: int,
    ) -> str:
        """Convenience: propose widening a layer."""
        return self.propose_mutation(
            model_id=model_id,
            mutation_type="widen_layer",
            position=position,
            new_output_dim=new_dim,
        )

    def propose_remove_layer(self, model_id: str, position: int) -> str:
        """Convenience: propose removing a layer."""
        return self.propose_mutation(
            model_id=model_id,
            mutation_type="remove_layer",
            position=position,
        )

    def propose_swap_activation(
        self, model_id: str, position: int, activation: str
    ) -> str:
        """Convenience: propose changing an activation function."""
        return self.propose_mutation(
            model_id=model_id,
            mutation_type="swap_activation",
            position=position,
            new_activation=activation,
        )
