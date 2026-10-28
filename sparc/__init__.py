"""SPARC: Sparse-Product And Reservoir Cloze - a deep, closed-form (backprop-free) contextual encoder."""
from .model import SPARC
from .utils import Bank, nz
__all__ = ["SPARC", "Bank", "nz"]
