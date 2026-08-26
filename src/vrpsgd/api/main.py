"""FastAPI service exposing the vrpsgd optimisers over HTTP.

Run locally:
    uvicorn vrpsgd.api.main:app --reload

Endpoints:
    GET  /health              liveness probe
    POST /optimize            run one algorithm, return the sparse portfolio
    POST /compare              run all four algorithms on the same data, for comparison
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from vrpsgd import __version__
from vrpsgd.api.schemas import AssetWeight, CompareResponse, OptimizeRequest, OptimizeResponse
from vrpsgd.data import load_return_matrix
from vrpsgd.objective import ObjectiveConfig
from vrpsgd.optimizers import REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vrpsgd.api")

app = FastAPI(
    title="VR-PSGD Sparse Portfolio Optimiser",
    description="Variance-reduced proximal stochastic gradient methods for sparse, "
    "long-only portfolio optimisation under an L1-regularised mean-variance objective.",
    version=__version__,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


def _run_single(request: OptimizeRequest) -> OptimizeResponse:
    try:
        R, tickers = load_return_matrix(request.tickers, str(request.start), str(request.end))
    except Exception as exc:  # noqa: BLE001 - surface as a clean HTTP error
        logger.exception("Failed to load market data")
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {exc}") from exc

    if R.shape[0] < 30:
        raise HTTPException(status_code=422, detail="Fewer than 30 trading days after alignment/cleaning.")

    cfg = ObjectiveConfig(gamma=request.gamma, lam=request.lam)
    optimizer_cls = REGISTRY[request.algorithm]
    result = optimizer_cls(cfg, seed=request.seed).fit(R, T_max=request.t_max)
    payload = result.as_dict()

    return OptimizeResponse(
        algorithm=payload["algorithm"],
        weights=[AssetWeight(ticker=t, weight=w) for t, w in zip(tickers, payload["weights"], strict=True)],
        n_assets=payload["n_assets"],
        n_nonzero=payload["n_nonzero"],
        sparsity_pct=payload["sparsity_pct"],
        final_objective=payload["final_objective"],
        n_iterations=payload["n_iterations"],
        wall_time_seconds=payload["wall_time_seconds"],
    )


@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    return _run_single(request)


@app.post("/compare", response_model=CompareResponse)
def compare(request: OptimizeRequest) -> CompareResponse:
    results = []
    for algo in REGISTRY:
        sub_request = request.model_copy(update={"algorithm": algo})
        results.append(_run_single(sub_request))
    return CompareResponse(results=results)
