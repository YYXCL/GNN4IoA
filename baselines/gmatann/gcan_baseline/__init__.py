"""GMATANN/GCAN 单子任务分配基线。"""

from .data import GCANCase, load_cases
from .model import GCANConfig, GraphConvolutionalAssignmentNetwork

__all__ = ["GCANCase", "GCANConfig", "GraphConvolutionalAssignmentNetwork", "load_cases"]
