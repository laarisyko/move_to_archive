"""SSSI -- Super Safe Super Intelligence SDK.

pip install supersafesuperintelligence
CLI: sssi join | sssi status | sssi infer | sssi train | sssi evolve | sssi vote
"""

__version__ = "0.1.0"

from .agent import Agent
from .network import NetworkClient
from .training import TrainingParticipant
from .inference import InferenceClient
from .architecture import ArchitectureEvolver
from .node_manager import NodeManager
