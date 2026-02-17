"""Model management: sharding, pipeline parallelism, weight loading."""

from .shard import ModelShard, ShardConfig
from .pipeline import PipelineStage, PipelineExecutor
from .loader import WeightLoader, WeightSaver
