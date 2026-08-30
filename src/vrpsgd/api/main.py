"""FastAPI service exposing the vrpsgd optimisers over HTTP.

This is the primary way to use vrpsgd: a live, rate-limited, monitored API,
not a library you import. See README.md for the deployed URL.

Run locally:
    uvicorn vrpsgd.api.main:app --reload

Endpoints:
    GET  /health              liveness probe
    GET  /metrics             Prometheus exposition format
    POST /optimize            run one algorithm, return the sparse portfolio
    POST /compare              run all four algorithms on the same data, for comparison
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from vrpsgd import __version__
from vrpsgd.api.schemas import (
    AssetWeight,
    CompareResponse,
    OptimizeRequest,
    OptimizeResponse,
    ReturnRow,
    ReturnsRequest,
    ReturnsResponse,
)
from vrpsgd.data import load_dated_returns, load_return_matrix
from vrpsgd.metrics import OPTIMIZATION_ITERATIONS, REQUEST_LATENCY, REQUESTS_TOTAL, metrics_response
from vrpsgd.objective import ObjectiveConfig
from vrpsgd.optimizers import REGISTRY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vrpsgd.api")

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="VR-PSGD Sparse Portfolio Optimiser",
    description="Variance-reduced proximal stochastic gradient methods for sparse, "
    "long-only portfolio optimisation under an L1-regularised mean-variance objective. "
    "Free to call — rate-limited to keep it usable for everyone.",
    version=__version__,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Public demo API called from arbitrary front-ends/scripts: allow any origin.
# Tighten this to specific origins if this ever serves anything beyond a
# public read-mostly demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/metrics")
def metrics() -> Response:
    body, content_type = metrics_response()
    return Response(content=body, media_type=content_type)


def _run_single(request: OptimizeRequest) -> OptimizeResponse:
    t0 = time.perf_counter()
    try:
        R, tickers = load_return_matrix(request.tickers, str(request.start), str(request.end))
    except Exception as exc:  # noqa: BLE001 - surface as a clean HTTP error
        logger.exception("Failed to load market data")
        REQUESTS_TOTAL.labels(endpoint="optimize", algorithm=request.algorithm, status="data_error").inc()
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {exc}") from exc

    if R.shape[0] < 30:
        REQUESTS_TOTAL.labels(endpoint="optimize", algorithm=request.algorithm, status="insufficient_data").inc()
        raise HTTPException(status_code=422, detail="Fewer than 30 trading days after alignment/cleaning.")

    cfg = ObjectiveConfig(gamma=request.gamma, lam=request.lam)
    optimizer_cls = REGISTRY[request.algorithm]

    try:
        result = optimizer_cls(cfg, seed=request.seed).fit(R, T_max=request.t_max)
    except Exception as exc:  # noqa: BLE001
        REQUESTS_TOTAL.labels(endpoint="optimize", algorithm=request.algorithm, status="optimizer_error").inc()
        raise HTTPException(status_code=500, detail=f"Optimisation failed: {exc}") from exc

    payload = result.as_dict()
    OPTIMIZATION_ITERATIONS.labels(algorithm=request.algorithm).observe(payload["wall_time_seconds"])
    REQUESTS_TOTAL.labels(endpoint="optimize", algorithm=request.algorithm, status="ok").inc()
    REQUEST_LATENCY.labels(endpoint="optimize", algorithm=request.algorithm).observe(time.perf_counter() - t0)

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
@limiter.limit("20/minute")
def optimize(request: Request, body: OptimizeRequest) -> OptimizeResponse:
    return _run_single(body)


@app.post("/compare", response_model=CompareResponse)
@limiter.limit("5/minute")
def compare(request: Request, body: OptimizeRequest) -> CompareResponse:
    results = []
    for algo in REGISTRY:
        sub_request = body.model_copy(update={"algorithm": algo})
        results.append(_run_single(sub_request))
    return CompareResponse(results=results)


@app.post("/returns", response_model=ReturnsResponse)
@limiter.limit("30/minute")
def returns(request: Request, body: ReturnsRequest) -> ReturnsResponse:
    """Date-aligned log returns for a set of tickers — the raw data behind
    /optimize, exposed directly. Used by spo-web's drift detection, so both
    services share one source of truth for market data (fetching, caching,
    alignment) instead of spo-web re-implementing it against Yahoo directly.
    """
    try:
        returns_df = load_dated_returns(body.tickers, str(body.start), str(body.end))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load market data")
        REQUESTS_TOTAL.labels(endpoint="returns", algorithm="n/a", status="data_error").inc()
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {exc}") from exc

    if returns_df.empty:
        REQUESTS_TOTAL.labels(endpoint="returns", algorithm="n/a", status="insufficient_data").inc()
        raise HTTPException(status_code=422, detail="No overlapping trading days for the given tickers/date range.")

    REQUESTS_TOTAL.labels(endpoint="returns", algorithm="n/a", status="ok").inc()
    rows = [ReturnRow(date=idx.date(), returnsByTicker=row.to_dict()) for idx, row in returns_df.iterrows()]
    return ReturnsResponse(rows=rows)