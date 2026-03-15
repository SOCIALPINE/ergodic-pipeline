"""
ERGODIC — Emergent Recursive Generation Over Distributed Interpretation Cycles
"""

from .pipeline import ErgodicConfig, ErgodicPipeline, generate_noise

__version__ = "0.9.0"
__all__ = ["ErgodicConfig", "ErgodicPipeline", "generate_noise"]