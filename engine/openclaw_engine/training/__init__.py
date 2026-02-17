"""Decentralized training: local training loops, gradient aggregation, compression."""

from .trainer import LocalTrainer, TrainingConfig
from .allreduce import RingAllReduce
from .compression import TopKCompressor, FP16Compressor, CompressorChain
