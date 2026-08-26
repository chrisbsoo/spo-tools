from vrpsgd.optimizers.base import OptimizationHistory, OptimizationResult, ProximalStochasticOptimizer
from vrpsgd.optimizers.prox_sarah import ProxSARAH
from vrpsgd.optimizers.prox_storm import ProxSTORM
from vrpsgd.optimizers.prox_svrg import ProxSVRG
from vrpsgd.optimizers.spgd import SPGD

REGISTRY: dict[str, type[ProximalStochasticOptimizer]] = {
    "spgd": SPGD,
    "prox-svrg": ProxSVRG,
    "prox-sarah": ProxSARAH,
    "prox-storm": ProxSTORM,
}

__all__ = [
    "SPGD",
    "ProxSVRG",
    "ProxSARAH",
    "ProxSTORM",
    "ProximalStochasticOptimizer",
    "OptimizationResult",
    "OptimizationHistory",
    "REGISTRY",
]
