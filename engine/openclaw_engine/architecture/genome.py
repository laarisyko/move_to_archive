"""Architecture genome -- a serializable, mutable blueprint for model structure.

The genome represents a model's architecture as an ordered list of LayerGenes.
It can be serialized for gossip, mutated by peers, and compiled into a live
PyTorch model. Think of it as DNA for neural networks.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class LayerType(str, Enum):
    LINEAR = "linear"
    ATTENTION = "attention"
    FEEDFORWARD = "feedforward"
    NORM = "norm"
    EMBEDDING = "embedding"
    CONV1D = "conv1d"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    SKIP = "skip"  # skip/residual connection marker


@dataclass
class LayerGene:
    """A single gene in the architecture genome -- describes one layer."""

    layer_type: LayerType
    input_dim: int
    output_dim: int
    params: Dict = field(default_factory=dict)
    # Optional fields for richer architectures.
    num_heads: int = 1           # for attention layers
    activation: str = "relu"     # activation function name
    dropout: float = 0.0
    skip_target: int = -1        # index of layer this connects back to (-1 = none)
    gene_id: str = ""            # unique identifier for tracking lineage

    def __post_init__(self):
        if not self.gene_id:
            self.gene_id = hashlib.sha256(
                f"{self.layer_type}:{self.input_dim}:{self.output_dim}:{id(self)}".encode()
            ).hexdigest()[:12]

    def to_dict(self) -> Dict:
        return {
            "layer_type": self.layer_type.value,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "params": self.params,
            "num_heads": self.num_heads,
            "activation": self.activation,
            "dropout": self.dropout,
            "skip_target": self.skip_target,
            "gene_id": self.gene_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LayerGene":
        return cls(
            layer_type=LayerType(d["layer_type"]),
            input_dim=d["input_dim"],
            output_dim=d["output_dim"],
            params=d.get("params", {}),
            num_heads=d.get("num_heads", 1),
            activation=d.get("activation", "relu"),
            dropout=d.get("dropout", 0.0),
            skip_target=d.get("skip_target", -1),
            gene_id=d.get("gene_id", ""),
        )


class ArchitectureGenome:
    """A full model architecture described as an ordered list of layer genes.

    The genome is the unit of collaborative evolution. Peers can:
    - Propose mutations (add/remove/widen layers, change activations, etc.)
    - Vote on proposals based on local validation performance
    - Apply accepted mutations to produce a new generation

    Genomes are content-addressed by their hash, so peers can verify they
    are talking about the same architecture.
    """

    def __init__(
        self,
        model_id: str,
        genes: Optional[List[LayerGene]] = None,
        generation: int = 0,
        parent_hash: str = "",
    ):
        self.model_id = model_id
        self.genes: List[LayerGene] = genes or []
        self.generation = generation
        self.parent_hash = parent_hash

    @classmethod
    def from_model(cls, model: nn.Module, model_id: str) -> "ArchitectureGenome":
        """Reverse-engineer a genome from an existing PyTorch model."""
        genes = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                genes.append(LayerGene(
                    layer_type=LayerType.LINEAR,
                    input_dim=module.in_features,
                    output_dim=module.out_features,
                ))
            elif isinstance(module, nn.MultiheadAttention):
                genes.append(LayerGene(
                    layer_type=LayerType.ATTENTION,
                    input_dim=module.embed_dim,
                    output_dim=module.embed_dim,
                    num_heads=module.num_heads,
                ))
            elif isinstance(module, nn.LayerNorm):
                dim = module.normalized_shape[0] if module.normalized_shape else 0
                genes.append(LayerGene(
                    layer_type=LayerType.NORM,
                    input_dim=dim,
                    output_dim=dim,
                ))
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU)):
                act_name = type(module).__name__.lower()
                genes.append(LayerGene(
                    layer_type=LayerType.ACTIVATION,
                    input_dim=0,
                    output_dim=0,
                    activation=act_name,
                ))
            elif isinstance(module, nn.Dropout):
                genes.append(LayerGene(
                    layer_type=LayerType.DROPOUT,
                    input_dim=0,
                    output_dim=0,
                    dropout=module.p,
                ))
        return cls(model_id=model_id, genes=genes)

    @classmethod
    def simple_transformer(
        cls,
        model_id: str,
        n_layers: int,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
    ) -> "ArchitectureGenome":
        """Create a genome for a standard transformer architecture."""
        ff_dim = ff_dim or hidden_dim * 4
        genes = []

        for i in range(n_layers):
            # Self-attention.
            genes.append(LayerGene(
                layer_type=LayerType.ATTENTION,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_heads=num_heads,
            ))
            # Layer norm after attention.
            genes.append(LayerGene(
                layer_type=LayerType.NORM,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
            ))
            # Feedforward: up-project.
            genes.append(LayerGene(
                layer_type=LayerType.LINEAR,
                input_dim=hidden_dim,
                output_dim=ff_dim,
            ))
            # Activation.
            genes.append(LayerGene(
                layer_type=LayerType.ACTIVATION,
                input_dim=ff_dim,
                output_dim=ff_dim,
                activation="gelu",
            ))
            # Feedforward: down-project.
            genes.append(LayerGene(
                layer_type=LayerType.LINEAR,
                input_dim=ff_dim,
                output_dim=hidden_dim,
            ))
            # Layer norm after FFN.
            genes.append(LayerGene(
                layer_type=LayerType.NORM,
                input_dim=hidden_dim,
                output_dim=hidden_dim,
            ))

        return cls(model_id=model_id, genes=genes)

    def compile(self) -> nn.Module:
        """Compile the genome into a live PyTorch model.

        Each gene is translated into its corresponding nn.Module.
        """
        layers = []
        for gene in self.genes:
            module = _gene_to_module(gene)
            if module is not None:
                layers.append(module)

        model = nn.Sequential(*layers)
        # Attach genome metadata.
        model._genome_hash = self.hash()
        model._genome_generation = self.generation
        return model

    def hash(self) -> str:
        """Content-addressable hash of the genome."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:24]

    @property
    def num_genes(self) -> int:
        return len(self.genes)

    @property
    def hidden_dims(self) -> List[int]:
        """All unique hidden dimensions in the genome."""
        dims = set()
        for g in self.genes:
            if g.input_dim > 0:
                dims.add(g.input_dim)
            if g.output_dim > 0:
                dims.add(g.output_dim)
        return sorted(dims)

    def estimated_parameters(self) -> int:
        """Estimate total parameter count from the genome."""
        total = 0
        for g in self.genes:
            if g.layer_type == LayerType.LINEAR:
                total += g.input_dim * g.output_dim + g.output_dim
            elif g.layer_type == LayerType.ATTENTION:
                # Q, K, V projections + output projection.
                total += 4 * g.input_dim * g.output_dim
            elif g.layer_type == LayerType.NORM:
                total += 2 * g.input_dim  # scale + bias
        return total

    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "genes": [g.to_dict() for g in self.genes],
            "generation": self.generation,
            "parent_hash": self.parent_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ArchitectureGenome":
        genes = [LayerGene.from_dict(g) for g in d["genes"]]
        return cls(
            model_id=d["model_id"],
            genes=genes,
            generation=d.get("generation", 0),
            parent_hash=d.get("parent_hash", ""),
        )

    def to_bytes(self) -> bytes:
        return json.dumps(self.to_dict()).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ArchitectureGenome":
        return cls.from_dict(json.loads(data))

    def clone(self) -> "ArchitectureGenome":
        return ArchitectureGenome(
            model_id=self.model_id,
            genes=copy.deepcopy(self.genes),
            generation=self.generation,
            parent_hash=self.parent_hash,
        )

    def diff(self, other: "ArchitectureGenome") -> List[str]:
        """Describe the differences between two genomes."""
        changes = []
        if len(self.genes) != len(other.genes):
            changes.append(f"layer count: {len(self.genes)} -> {len(other.genes)}")
        for i, (a, b) in enumerate(zip(self.genes, other.genes)):
            if a.to_dict() != b.to_dict():
                changes.append(f"gene[{i}]: {a.layer_type.value} modified")
        if len(other.genes) > len(self.genes):
            for i in range(len(self.genes), len(other.genes)):
                changes.append(f"gene[{i}]: {other.genes[i].layer_type.value} added")
        return changes

    def __repr__(self):
        return (
            f"ArchitectureGenome(model={self.model_id}, "
            f"genes={self.num_genes}, gen={self.generation}, "
            f"hash={self.hash()}, ~{self.estimated_parameters():,} params)"
        )


def _gene_to_module(gene: LayerGene) -> Optional[nn.Module]:
    """Convert a LayerGene into a PyTorch module."""
    if gene.layer_type == LayerType.LINEAR:
        return nn.Linear(gene.input_dim, gene.output_dim)
    elif gene.layer_type == LayerType.ATTENTION:
        return nn.MultiheadAttention(
            embed_dim=gene.input_dim,
            num_heads=gene.num_heads,
            batch_first=True,
        )
    elif gene.layer_type == LayerType.NORM:
        return nn.LayerNorm(gene.input_dim)
    elif gene.layer_type == LayerType.ACTIVATION:
        activations = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        return activations.get(gene.activation, nn.ReLU)()
    elif gene.layer_type == LayerType.DROPOUT:
        return nn.Dropout(gene.dropout)
    elif gene.layer_type == LayerType.FEEDFORWARD:
        return nn.Sequential(
            nn.Linear(gene.input_dim, gene.output_dim),
            nn.ReLU(),
            nn.Linear(gene.output_dim, gene.input_dim),
        )
    return None
