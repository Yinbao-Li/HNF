"""Foveated active perception for long-window waveform processing.

This subpackage implements a "global scan -> local fovea -> causal memory" loop
on top of the existing HNF backbone (Huygens-based picking + optional inversion).
"""

from .peripheral_scanner import CandidateRegion, PeripheralScanner
from .fovea_processor import FoveaOutput, FoveaProcessor, SUPPORTED_WINDOW_SIZES
from .causal_memory import (
    CausalEdge,
    CausalGraph,
    CausalMemory,
    CausalNode,
    Scheduler,
    SchedulerDecision,
    WindowSelector,
)
from .engine import FoveatedEngine, FoveatedEngineOutput, GazeStep, visualize_trajectory_ascii
from .training import (
    FoveatedTrainConfig,
    pick_bce_loss,
    stage1_behavior_cloning_loss,
    stage2_joint_loss,
)

__all__ = [
    "CandidateRegion",
    "PeripheralScanner",
    "FoveaProcessor",
    "FoveaOutput",
    "SUPPORTED_WINDOW_SIZES",
    "CausalNode",
    "CausalEdge",
    "CausalGraph",
    "CausalMemory",
    "Scheduler",
    "SchedulerDecision",
    "WindowSelector",
    "FoveatedEngine",
    "FoveatedEngineOutput",
    "GazeStep",
    "visualize_trajectory_ascii",
    "FoveatedTrainConfig",
    "pick_bce_loss",
    "stage1_behavior_cloning_loss",
    "stage2_joint_loss",
]
