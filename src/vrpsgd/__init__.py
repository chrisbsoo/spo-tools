"""vrpsgd: variance-reduced proximal stochastic gradient methods for sparse portfolio optimisation."""

__version__ = "0.1.0"

from vrpsgd.objective import ObjectiveConfig
from vrpsgd.optimizers import REGISTRY, SPGD, ProxSARAH, ProxSTORM, ProxSVRG

__all__ = ["ObjectiveConfig", "SPGD", "ProxSVRG", "ProxSARAH", "ProxSTORM", "REGISTRY", "__version__"]
