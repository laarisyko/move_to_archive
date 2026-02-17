"""Decentralized training: local training loops, gradient aggregation, compression."""

from .trainer import LocalTrainer, TrainingConfig
from .allreduce import RingAllReduce
from .compression import TopKCompressor, FP16Compressor, CompressorChain
from .hierarchical import (
    HierarchicalAllReduce,
    ClusterConfig,
    ClusterTopology,
    PeerClusterAssignment,
    assign_clusters_vrf,
    compute_scaling_stats,
)
from .cluster import ClusterManager, ClusterMembership, PeerCapacity, PeerRole
