"""Integration tests for collaborative architecture evolution.

Tests the full lifecycle: genome creation, mutations, proposal/voting,
and weight migration -- all without any central coordinator.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "engine"))

import torch
import torch.nn as nn

from openclaw_engine.architecture.genome import (
    ArchitectureGenome,
    LayerGene,
    LayerType,
)
from openclaw_engine.architecture.mutations import (
    AddLayerMutation,
    RemoveLayerMutation,
    WidenLayerMutation,
    SwapActivation,
    InsertSkipConnection,
    MutationGenerator,
    Mutation,
)
from openclaw_engine.architecture.evolution import (
    ArchitectureProposal,
    ProposalVote,
    EvolutionProtocol,
    FitnessEvaluator,
    VoteDecision,
)
from openclaw_engine.architecture.migration import WeightMigrator


# ---- Genome tests ----


def test_genome_creation_and_hash():
    """Verify genome creation and content-addressable hashing."""
    genome = ArchitectureGenome.simple_transformer(
        model_id="test-model", n_layers=2, hidden_dim=64, num_heads=4
    )
    assert genome.num_genes > 0
    assert genome.generation == 0

    h1 = genome.hash()
    h2 = genome.hash()
    assert h1 == h2, "Hash must be deterministic"
    assert len(h1) == 24


def test_genome_serialization_roundtrip():
    """Verify genome can be serialized and deserialized."""
    genome = ArchitectureGenome.simple_transformer("test", 2, 64)
    data = genome.to_bytes()
    restored = ArchitectureGenome.from_bytes(data)

    assert restored.model_id == genome.model_id
    assert restored.num_genes == genome.num_genes
    assert restored.hash() == genome.hash()


def test_genome_compile():
    """Verify a genome can be compiled into a live PyTorch model."""
    genome = ArchitectureGenome(
        model_id="simple",
        genes=[
            LayerGene(LayerType.LINEAR, 64, 128),
            LayerGene(LayerType.ACTIVATION, 0, 0, activation="relu"),
            LayerGene(LayerType.LINEAR, 128, 64),
        ],
    )
    model = genome.compile()
    x = torch.randn(2, 64)
    output = model(x)
    assert output.shape == (2, 64)


def test_genome_estimated_parameters():
    """Verify parameter estimation matches compiled model."""
    genome = ArchitectureGenome(
        model_id="count-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )
    estimated = genome.estimated_parameters()
    model = genome.compile()
    actual = sum(p.numel() for p in model.parameters())
    assert estimated == actual


def test_genome_diff():
    """Verify diff between two genomes."""
    genome_a = ArchitectureGenome(
        model_id="test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )
    mutation = AddLayerMutation(
        position=1,
        gene=LayerGene(LayerType.NORM, 64, 64),
    )
    genome_b = mutation.apply(genome_a)
    diffs = genome_a.diff(genome_b)
    assert len(diffs) > 0
    assert any("layer count" in d for d in diffs)


# ---- Mutation tests ----


def test_add_layer_mutation():
    """Verify adding a layer increases gene count."""
    genome = ArchitectureGenome(
        model_id="test",
        genes=[LayerGene(LayerType.LINEAR, 32, 32)],
    )
    mutation = AddLayerMutation(
        position=1,
        gene=LayerGene(LayerType.LINEAR, 32, 32),
    )
    new_genome = mutation.apply(genome)
    assert new_genome.num_genes == 2
    assert new_genome.generation == 1
    assert new_genome.parent_hash == genome.hash()


def test_remove_layer_mutation():
    """Verify removing a layer decreases gene count."""
    genome = ArchitectureGenome(
        model_id="test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.ACTIVATION, 0, 0),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )
    mutation = RemoveLayerMutation(position=1)
    new_genome = mutation.apply(genome)
    assert new_genome.num_genes == 2


def test_widen_layer_mutation():
    """Verify widening changes the output dimension."""
    genome = ArchitectureGenome(
        model_id="test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )
    mutation = WidenLayerMutation(position=0, new_output_dim=128)
    new_genome = mutation.apply(genome)
    assert new_genome.genes[0].output_dim == 128
    # The next layer's input should be updated too.
    assert new_genome.genes[1].input_dim == 128


def test_swap_activation_mutation():
    """Verify activation swap changes the activation name."""
    genome = ArchitectureGenome(
        model_id="test",
        genes=[
            LayerGene(LayerType.ACTIVATION, 0, 0, activation="relu"),
        ],
    )
    mutation = SwapActivation(position=0, new_activation="gelu")
    new_genome = mutation.apply(genome)
    assert new_genome.genes[0].activation == "gelu"


def test_insert_skip_connection():
    """Verify skip connection is recorded in the gene."""
    genome = ArchitectureGenome(
        model_id="test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 32),
            LayerGene(LayerType.LINEAR, 32, 32),
            LayerGene(LayerType.LINEAR, 32, 32),
        ],
    )
    mutation = InsertSkipConnection(source_position=0, target_position=2)
    new_genome = mutation.apply(genome)
    assert new_genome.genes[2].skip_target == 0


def test_mutation_serialization():
    """Verify mutations can be serialized and deserialized."""
    mutations = [
        AddLayerMutation(1, LayerGene(LayerType.LINEAR, 32, 64)),
        RemoveLayerMutation(2),
        WidenLayerMutation(0, 128),
        SwapActivation(1, "gelu"),
        InsertSkipConnection(0, 3),
    ]
    for mut in mutations:
        d = mut.to_dict()
        restored = Mutation.from_dict(d)
        assert restored.describe() == mut.describe()


def test_random_mutation_generator():
    """Verify the mutation generator produces valid mutations."""
    genome = ArchitectureGenome.simple_transformer("test", 2, 64)
    gen = MutationGenerator(seed=42)

    for _ in range(10):
        mutation = gen.random_mutation(genome)
        new_genome = mutation.apply(genome)
        assert new_genome.generation == genome.generation + 1
        assert new_genome.parent_hash == genome.hash()


# ---- Evolution protocol tests ----


def test_proposal_and_voting():
    """Simulate 3 peers proposing and voting on architecture changes."""
    genome = ArchitectureGenome(
        model_id="evolve-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.ACTIVATION, 0, 0, activation="relu"),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )

    # Create 3 peers with independent evolution protocols.
    peers = []
    for i in range(3):
        protocol = EvolutionProtocol(
            peer_id=f"peer-{i}",
            current_genome=genome.clone(),
            evaluator=FitnessEvaluator(),
            approval_quorum=0.6,
            min_voters=2,
        )
        peers.append(protocol)

    # Peer 0 proposes a mutation.
    mutation = AddLayerMutation(
        position=1,
        gene=LayerGene(LayerType.NORM, 64, 64),
    )
    proposal = peers[0].propose_mutation(mutation)

    # Peers 1 and 2 receive and vote.
    vote1 = peers[1].receive_proposal(proposal)
    vote2 = peers[2].receive_proposal(proposal)

    # Peer 0 also receives the votes.
    peers[0].receive_vote(vote1)
    peers[0].receive_vote(vote2)

    # All peers tally independently -- deterministic outcome.
    results = [p.tally_votes(proposal.proposal_id) for p in peers]

    # Check that all peers reached the same decision.
    non_none = [r for r in results if r is not None]
    if non_none:
        hashes = [r.hash() for r in non_none]
        assert len(set(hashes)) == 1, "All peers must agree on the new genome"


def test_proposal_rejection():
    """Verify a proposal can be rejected if peers vote against it."""
    genome = ArchitectureGenome(
        model_id="reject-test",
        genes=[LayerGene(LayerType.LINEAR, 32, 32)],
    )

    protocol = EvolutionProtocol(
        peer_id="peer-0",
        current_genome=genome.clone(),
        approval_quorum=0.8,
        min_voters=2,
    )

    mutation = RemoveLayerMutation(position=0)
    proposal = protocol.propose_mutation(mutation)

    # Simulate two reject votes.
    protocol.receive_vote(ProposalVote(
        proposal_id=proposal.proposal_id,
        voter_id="peer-1",
        decision=VoteDecision.REJECT,
    ))
    protocol.receive_vote(ProposalVote(
        proposal_id=proposal.proposal_id,
        voter_id="peer-2",
        decision=VoteDecision.REJECT,
    ))

    result = protocol.tally_votes(proposal.proposal_id)
    assert result is None, "Proposal should be rejected"


def test_evolution_history():
    """Verify genome history is tracked across generations."""
    genome = ArchitectureGenome(
        model_id="history-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )

    protocol = EvolutionProtocol(
        peer_id="peer-0",
        current_genome=genome,
        approval_quorum=0.5,
        min_voters=1,
    )

    # Apply two accepted mutations.
    for _ in range(2):
        mutation = AddLayerMutation(
            position=1,
            gene=LayerGene(LayerType.NORM, 64, 64),
        )
        proposal = protocol.propose_mutation(mutation)
        protocol.receive_vote(ProposalVote(
            proposal_id=proposal.proposal_id,
            voter_id="peer-1",
            decision=VoteDecision.APPROVE,
        ))
        protocol.tally_votes(proposal.proposal_id)

    assert len(protocol.genome_history) == 3  # initial + 2 mutations
    assert protocol.current_genome.generation == 2


# ---- Weight migration tests ----


def test_weight_migration_same_shape():
    """Verify weights are preserved when architecture doesn't change shape."""
    genome = ArchitectureGenome(
        model_id="migrate-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )

    old_model = genome.compile()
    old_state = old_model.state_dict()

    migrator = WeightMigrator()
    new_state = migrator.migrate_state_dict(old_state, genome, genome)

    for key in old_state:
        assert torch.equal(old_state[key], new_state[key])


def test_weight_migration_added_layer():
    """Verify weight migration when a layer is added."""
    old_genome = ArchitectureGenome(
        model_id="test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )

    mutation = AddLayerMutation(
        position=1,
        gene=LayerGene(LayerType.NORM, 64, 64),
    )
    new_genome = mutation.apply(old_genome)

    old_model = old_genome.compile()
    migrator = WeightMigrator()

    new_model = migrator.migrate(old_model, old_genome, new_genome)
    assert new_model is not None

    # The new model should have more modules.
    old_params = sum(p.numel() for p in old_model.parameters())
    new_params = sum(p.numel() for p in new_model.parameters())
    assert new_params >= old_params


def test_weight_migration_widened():
    """Verify Net2Net-style weight migration for widened layers."""
    old_genome = ArchitectureGenome(
        model_id="widen-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64, gene_id="layer-a"),
            LayerGene(LayerType.LINEAR, 64, 32, gene_id="layer-b"),
        ],
    )

    new_genome = ArchitectureGenome(
        model_id="widen-test",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 128, gene_id="layer-a"),
            LayerGene(LayerType.LINEAR, 128, 32, gene_id="layer-b"),
        ],
        generation=1,
        parent_hash=old_genome.hash(),
    )

    old_model = old_genome.compile()
    migrator = WeightMigrator(noise_scale=0.001)
    new_model = migrator.migrate(old_model, old_genome, new_genome)

    # The new model should have more parameters.
    new_params = sum(p.numel() for p in new_model.parameters())
    old_params = sum(p.numel() for p in old_model.parameters())
    assert new_params > old_params


# ---- Full lifecycle test ----


def test_full_evolution_cycle():
    """End-to-end: create genome, mutate, vote, accept, migrate weights."""
    # Start with a small model.
    genome = ArchitectureGenome(
        model_id="full-cycle",
        genes=[
            LayerGene(LayerType.LINEAR, 32, 64),
            LayerGene(LayerType.ACTIVATION, 0, 0, activation="relu"),
            LayerGene(LayerType.LINEAR, 64, 32),
        ],
    )
    old_model = genome.compile()

    # Peer proposes adding a normalization layer.
    mutation = AddLayerMutation(
        position=1,
        gene=LayerGene(LayerType.NORM, 64, 64),
    )
    new_genome = mutation.apply(genome)

    # Vote: approved.
    protocol = EvolutionProtocol(
        peer_id="peer-0",
        current_genome=genome,
        approval_quorum=0.5,
        min_voters=1,
    )
    proposal = protocol.propose_mutation(mutation)
    protocol.receive_vote(ProposalVote(
        proposal_id=proposal.proposal_id,
        voter_id="peer-1",
        decision=VoteDecision.APPROVE,
        measured_fitness=-0.5,
    ))
    accepted = protocol.tally_votes(proposal.proposal_id)
    assert accepted is not None
    assert accepted.generation == 1

    # Migrate weights.
    migrator = WeightMigrator()
    new_model = migrator.migrate(old_model, genome, accepted)

    # Verify the new model works.
    x = torch.randn(2, 32)
    output = new_model(x)
    assert output.shape == (2, 32)


if __name__ == "__main__":
    tests = [
        test_genome_creation_and_hash,
        test_genome_serialization_roundtrip,
        test_genome_compile,
        test_genome_estimated_parameters,
        test_genome_diff,
        test_add_layer_mutation,
        test_remove_layer_mutation,
        test_widen_layer_mutation,
        test_swap_activation_mutation,
        test_insert_skip_connection,
        test_mutation_serialization,
        test_random_mutation_generator,
        test_proposal_and_voting,
        test_proposal_rejection,
        test_evolution_history,
        test_weight_migration_same_shape,
        test_weight_migration_added_layer,
        test_weight_migration_widened,
        test_full_evolution_cycle,
    ]
    for test in tests:
        test()
        print(f"  [PASS] {test.__name__}")
    print(f"\nAll {len(tests)} architecture evolution tests passed!")
