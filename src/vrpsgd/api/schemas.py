from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

AlgorithmName = Literal["spgd", "prox-svrg", "prox-sarah", "prox-storm"]


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=2, examples=[["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]])
    start: date = Field(..., examples=["2020-01-01"])
    end: date = Field(..., examples=["2025-01-01"])
    algorithm: AlgorithmName = "prox-svrg"
    gamma: float = Field(2.0, description="Risk-aversion parameter.")
    lam: float = Field(0.01, ge=0.0, description="L1 penalty strength controlling sparsity.")
    t_max: int = Field(3000, ge=10, le=50_000, description="Iteration budget.")
    seed: int = 42


class AssetWeight(BaseModel):
    ticker: str
    weight: float


class OptimizeResponse(BaseModel):
    algorithm: str
    weights: list[AssetWeight]
    n_assets: int
    n_nonzero: int
    sparsity_pct: float
    final_objective: float
    n_iterations: int
    wall_time_seconds: float


class CompareResponse(BaseModel):
    results: list[OptimizeResponse]
