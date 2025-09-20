"""
ATCOT: Adaptive Tool-Augmented Chain of Thought Reasoning Framework

This package implements the ATCOT framework for adaptive reasoning with tool feedback,
enabling LLMs to revise their reasoning trajectories when tool invocations reveal errors.
"""

from .core.state_representation import ATCOTState, PlanningStructure, ReasoningTrace, ToolHistory, CorrectionLog
from .core.framework import ATCOTFramework
from .core.planning import AdaptivePlanner
from .core.execution import ToolAugmentedExecutor
from .core.correction import AdaptiveCorrectionMechanism
from .tools.base import BaseTool, ToolRegistry
from .utils.consistency import ConsistencyChecker
from .utils.config import ATCOTConfig

__version__ = "1.0.0"
__author__ = "Lei Wei, Xu Dong, Xiao Peng, Niantao Xie, Bin Wang"

__all__ = [
    "ATCOTState",
    "PlanningStructure", 
    "ReasoningTrace",
    "ToolHistory",
    "CorrectionLog",
    "ATCOTFramework",
    "AdaptivePlanner",
    "ToolAugmentedExecutor", 
    "AdaptiveCorrectionMechanism",
    "BaseTool",
    "ToolRegistry",
    "ConsistencyChecker",
    "ATCOTConfig"
]
