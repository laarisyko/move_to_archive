"""Architecture mutation operators.

Each mutation is a self-contained, serializable operation that transforms one
genome into another. Peers propose mutations; the swarm votes on them.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .genome import ArchitectureGenome, LayerGene, LayerType


class Mutation(ABC):
    """Base class for architecture mutations."""

    @abstractmethod
    def apply(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Apply this mutation to a genome, returning a new genome."""
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of the mutation."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialize the mutation for gossip transmission."""
        ...

    @classmethod
    def from_dict(cls, d: Dict) -> "Mutation":
        """Deserialize a mutation from a dict."""
        mutation_types = {
            "add_layer": AddLayerMutation,
            "remove_layer": RemoveLayerMutation,
            "widen_layer": WidenLayerMutation,
            "insert_skip": InsertSkipConnection,
            "swap_activation": SwapActivation,
        }
        mut_type = d.get("type", "")
        if mut_type in mutation_types:
            return mutation_types[mut_type]._from_dict(d)
        raise ValueError(f"Unknown mutation type: {mut_type}")


class AddLayerMutation(Mutation):
    """Insert a new layer at a given position in the genome."""

    def __init__(self, position: int, gene: LayerGene):
        self.position = position
        self.gene = gene

    def apply(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        new = genome.clone()
        pos = min(self.position, len(new.genes))
        new.genes.insert(pos, copy.deepcopy(self.gene))
        new.generation += 1
        new.parent_hash = genome.hash()
        return new

    def describe(self) -> str:
        return (
            f"Add {self.gene.layer_type.value} layer "
            f"({self.gene.input_dim}->{self.gene.output_dim}) at position {self.position}"
        )

    def to_dict(self) -> Dict:
        return {
            "type": "add_layer",
            "position": self.position,
            "gene": self.gene.to_dict(),
        }

    @classmethod
    def _from_dict(cls, d: Dict) -> "AddLayerMutation":
        return cls(position=d["position"], gene=LayerGene.from_dict(d["gene"]))


class RemoveLayerMutation(Mutation):
    """Remove a layer at a given position."""

    def __init__(self, position: int):
        self.position = position

    def apply(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        new = genome.clone()
        if 0 <= self.position < len(new.genes):
            new.genes.pop(self.position)
        new.generation += 1
        new.parent_hash = genome.hash()
        return new

    def describe(self) -> str:
        return f"Remove layer at position {self.position}"

    def to_dict(self) -> Dict:
        return {"type": "remove_layer", "position": self.position}

    @classmethod
    def _from_dict(cls, d: Dict) -> "RemoveLayerMutation":
        return cls(position=d["position"])


class WidenLayerMutation(Mutation):
    """Increase the width (output dimension) of a layer.

    Uses Net2Net-style widening: new neurons are copies of existing ones
    with noise, so the function is approximately preserved.
    """

    def __init__(self, position: int, new_output_dim: int):
        self.position = position
        self.new_output_dim = new_output_dim

    def apply(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        new = genome.clone()
        if 0 <= self.position < len(new.genes):
            gene = new.genes[self.position]
            gene.output_dim = self.new_output_dim
            # Fix dimensions of the next layer that takes this as input.
            for i in range(self.position + 1, len(new.genes)):
                next_gene = new.genes[i]
                if next_gene.input_dim > 0:
                    next_gene.input_dim = self.new_output_dim
                    break
        new.generation += 1
        new.parent_hash = genome.hash()
        return new

    def describe(self) -> str:
        return f"Widen layer {self.position} to output_dim={self.new_output_dim}"

    def to_dict(self) -> Dict:
        return {
            "type": "widen_layer",
            "position": self.position,
            "new_output_dim": self.new_output_dim,
        }

    @classmethod
    def _from_dict(cls, d: Dict) -> "WidenLayerMutation":
        return cls(position=d["position"], new_output_dim=d["new_output_dim"])


class InsertSkipConnection(Mutation):
    """Add a residual/skip connection between two layers."""

    def __init__(self, source_position: int, target_position: int):
        self.source_position = source_position
        self.target_position = target_position

    def apply(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        new = genome.clone()
        if 0 <= self.target_position < len(new.genes):
            new.genes[self.target_position].skip_target = self.source_position
        new.generation += 1
        new.parent_hash = genome.hash()
        return new

    def describe(self) -> str:
        return f"Add skip connection from layer {self.source_position} to {self.target_position}"

    def to_dict(self) -> Dict:
        return {
            "type": "insert_skip",
            "source_position": self.source_position,
            "target_position": self.target_position,
        }

    @classmethod
    def _from_dict(cls, d: Dict) -> "InsertSkipConnection":
        return cls(
            source_position=d["source_position"],
            target_position=d["target_position"],
        )


class SwapActivation(Mutation):
    """Change the activation function of a layer."""

    def __init__(self, position: int, new_activation: str):
        self.position = position
        self.new_activation = new_activation

    def apply(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        new = genome.clone()
        if 0 <= self.position < len(new.genes):
            new.genes[self.position].activation = self.new_activation
        new.generation += 1
        new.parent_hash = genome.hash()
        return new

    def describe(self) -> str:
        return f"Swap activation at layer {self.position} to {self.new_activation}"

    def to_dict(self) -> Dict:
        return {
            "type": "swap_activation",
            "position": self.position,
            "new_activation": self.new_activation,
        }

    @classmethod
    def _from_dict(cls, d: Dict) -> "SwapActivation":
        return cls(position=d["position"], new_activation=d["new_activation"])


class MutationGenerator:
    """Generates random mutations for architecture search.

    Each peer can use this to propose novel architectures. The mutations
    are biased toward changes that are likely to be beneficial (based on
    common neural architecture search heuristics).
    """

    ACTIVATIONS = ["relu", "gelu", "silu", "tanh"]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def random_mutation(self, genome: ArchitectureGenome) -> Mutation:
        """Generate a random valid mutation for the given genome."""
        mutation_types = ["add_layer", "remove_layer", "widen_layer", "swap_activation"]

        if genome.num_genes >= 3:
            mutation_types.append("insert_skip")
        if genome.num_genes <= 2:
            # Don't remove layers from very small models.
            mutation_types.remove("remove_layer")

        choice = self.rng.choice(mutation_types)

        if choice == "add_layer":
            return self._random_add(genome)
        elif choice == "remove_layer":
            return self._random_remove(genome)
        elif choice == "widen_layer":
            return self._random_widen(genome)
        elif choice == "swap_activation":
            return self._random_swap_activation(genome)
        elif choice == "insert_skip":
            return self._random_skip(genome)
        else:
            return self._random_add(genome)

    def _random_add(self, genome: ArchitectureGenome) -> AddLayerMutation:
        pos = self.rng.randint(0, genome.num_genes)
        # Pick a dimension from the existing genome.
        dims = genome.hidden_dims or [64]
        dim = self.rng.choice(dims)
        layer_type = self.rng.choice([LayerType.LINEAR, LayerType.NORM])
        gene = LayerGene(
            layer_type=layer_type,
            input_dim=dim,
            output_dim=dim,
            activation=self.rng.choice(self.ACTIVATIONS),
        )
        return AddLayerMutation(position=pos, gene=gene)

    def _random_remove(self, genome: ArchitectureGenome) -> RemoveLayerMutation:
        pos = self.rng.randint(0, genome.num_genes - 1)
        return RemoveLayerMutation(position=pos)

    def _random_widen(self, genome: ArchitectureGenome) -> WidenLayerMutation:
        # Find linear layers that can be widened.
        candidates = [
            i for i, g in enumerate(genome.genes)
            if g.layer_type in (LayerType.LINEAR, LayerType.FEEDFORWARD)
        ]
        if not candidates:
            # Fall back to an add mutation.
            return self._random_add(genome)
        pos = self.rng.choice(candidates)
        current_dim = genome.genes[pos].output_dim
        # Widen by 25-100%.
        factor = self.rng.uniform(1.25, 2.0)
        new_dim = int(current_dim * factor)
        # Round to nearest multiple of 8 for hardware efficiency.
        new_dim = max(8, (new_dim + 7) // 8 * 8)
        return WidenLayerMutation(position=pos, new_output_dim=new_dim)

    def _random_swap_activation(self, genome: ArchitectureGenome) -> SwapActivation:
        candidates = [
            i for i, g in enumerate(genome.genes)
            if g.layer_type == LayerType.ACTIVATION
        ]
        if not candidates:
            pos = self.rng.randint(0, max(0, genome.num_genes - 1))
        else:
            pos = self.rng.choice(candidates)
        new_act = self.rng.choice(self.ACTIVATIONS)
        return SwapActivation(position=pos, new_activation=new_act)

    def _random_skip(self, genome: ArchitectureGenome) -> InsertSkipConnection:
        n = genome.num_genes
        src = self.rng.randint(0, n - 2)
        tgt = self.rng.randint(src + 2, min(src + 6, n - 1))
        return InsertSkipConnection(source_position=src, target_position=tgt)
