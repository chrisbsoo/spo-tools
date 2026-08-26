# vrpsgd — Variance-Reduced Proximal Stochastic Gradient Methods for Sparse Portfolio Optimisation

[![CI](https://github.com/<your-username>/vrpsgd/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/vrpsgd/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A tested, typed, Dockerised implementation of four proximal stochastic gradient
algorithms — **SPGD**, **Prox-SVRG**, **Prox-SARAH**, **Prox-STORM** — for
L1-regularised, simplex-constrained (long-only, fully invested) portfolio
optimisation, served over a small HTTP API.

This is the production-code counterpart to my undergraduate dissertation,
*[From Theory to Trading: A Comparative Study of Variance-Reduced Proximal
Stochastic Gradient Algorithms for Real-Time S&P 500 Sparse Portfolio
Optimisation](docs/thesis.pdf)* (University of Bristol, MATH30009, 2026,
supervised by Prof. Christophe Andrieu). The dissertation derives and proves
convergence for each algorithm; this repo turns that math into software that
runs, is tested, and can be called by something other than a notebook.

## Why this exists

The dissertation's original code was a single ~700-line script: hardcoded
hyperparameters, four near-identical copy-pasted training loops, results
dumped to CSV, no tests. It worked well enough to produce the dissertation's
results, but it isn't something you'd trust in a pipeline. This repo is a
from-scratch rewrite that:

- factors the four algorithms into one shared interface so they're easy to
  compare, extend, and unit test individually,
- adds tests that actually check numerical correctness — including a
  finite-difference check of the analytic gradient, which is the single
  test most likely to catch a silent bug in optimisation code,
- fixes a real bug the original code had (see [Deviations from the thesis](#deviations-from-the-thesis-and-why)),
- wraps it in a FastAPI service + Dockerfile so "run the optimiser" doesn't
  require cloning a repo and reading a script.

## The problem

Given daily returns for a set of assets, find portfolio weights `w` that
minimise

```
f(w) + λ‖w‖₁     subject to   Σwᵢ = 1,  wᵢ ≥ 0
```

where `f` is a mean-variance objective and the L1 penalty drives small,
unprofitable positions to exactly zero — producing a portfolio you could
actually trade without the fixed costs of tiny positions swamping their
returns. `f` is a finite sum over trading days, so it's cheap to get a noisy
gradient estimate from one day and expensive to get the exact gradient from
all of them — the standard setting where variance-reduced stochastic methods
(SVRG, SARAH, STORM) earn their keep over plain SGD.

| Algorithm | Idea | Convergence (strongly convex) |
|---|---|---|
| SPGD | one random sample, diminishing step size | sub-linear, `O(1/√k)` |
| Prox-SVRG | periodic full-gradient "anchor" corrects sampled gradients | linear |
| Prox-SARAH | recursively updates the gradient estimate step-to-step (biased, lower variance) | linear |
| Prox-STORM | momentum-style recursive estimator, no periodic full gradient | linear (non-convex guarantees too) |

Full derivations, strong-convexity/L-smoothness proofs for the objective, and
convergence proofs for all four algorithms are in the [dissertation](docs/thesis.pdf), §2.9 and §3.

## Quickstart

```bash
git clone https://github.com/<your-username>/vrpsgd.git
cd vrpsgd
pip install -e ".[dev]"
pytest                      # 37 tests, ~85% coverage
```

Run locally without Docker:

```bash
uvicorn vrpsgd.api.main:app --reload
# -> http://localhost:8000/docs for interactive OpenAPI docs
```

Or with Docker:

```bash
docker compose up --build
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
        "tickers": ["AAPL","MSFT","GOOGL","AMZN","NVDA","JPM","XOM","KO"],
        "start": "2020-01-01",
        "end": "2025-01-01",
        "algorithm": "prox-svrg",
        "gamma": 2.0,
        "lam": 0.01
      }'
```

Or from the CLI:

```bash
vrpsgd optimize AAPL MSFT GOOGL AMZN NVDA --start 2020-01-01 --end 2025-01-01
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | liveness probe |
| `/optimize` | POST | run one algorithm, return sparse weights + diagnostics |
| `/compare` | POST | run all four algorithms on the same data for side-by-side comparison |

Full request/response schemas: `/docs` (Swagger) or `/redoc` once the server
is running — generated automatically from the Pydantic models in
`src/vrpsgd/api/schemas.py`.

## Repository layout

```
src/vrpsgd/
  proximal.py          soft-thresholding, simplex projection (sparsity-preserving variant)
  objective.py          objective + analytic gradients (covariance and legacy second-moment modes)
  optimizers/
    base.py              shared ProximalStochasticOptimizer interface
    spgd.py, prox_svrg.py, prox_sarah.py, prox_storm.py
  data.py                Yahoo Finance loader with disk caching
  api/                   FastAPI app + Pydantic schemas
  cli.py                 command-line entry point
tests/                   unit + API tests (pytest)
docs/                    dissertation PDF, architecture notes
.github/workflows/ci.yml lint (ruff) + type-check (mypy) + test + docker build
```

## Deviations from the thesis, and why

Writing tests surfaced a real discrepancy between the dissertation's derived
objective and the original implementation, which I'm documenting rather than
quietly fixing, because it's a more honest signal of what happened than
pretending the first version was correct:

- **Covariance vs. second moment.** The thesis derives
  `f(w) = ½wᵀΣw − γμᵀw` using the *covariance* matrix `Σ`. The original code
  instead used `RᵀR/T` (the *second-moment* matrix, i.e. covariance plus the
  outer product of the mean) — correct only if returns had zero mean, which
  they don't. This repo's default (`ObjectiveConfig(mode="covariance")`) is
  the thesis-correct version; the original behaviour is preserved as
  `mode="second_moment"` so the dissertation's published numbers stay
  reproducible.
- **Sign convention on `gamma`.** With the original second-moment objective,
  increasing `gamma` pushes toward *more* return-seeking (less conservative)
  portfolios — the opposite of "risk aversion" as the comment claimed. Under
  the corrected covariance-based objective, `gamma` behaves as documented.

See `CHANGELOG.md` for the full list, including limitations not yet
addressed (no transaction costs, no cross-validated hyperparameters).

## Roadmap

- [ ] `v0.2` — turnover/transaction-cost penalty in the objective; walk-forward
      backtest as an API endpoint (`/backtest`), reproducing dissertation Ch. 6.
- [ ] `v0.3` — Prox-SAGA (memory-based variant, proved in the dissertation but
      not yet in this package); Nesterov-accelerated variants (§2.7).
- [ ] `v0.4` — hosted demo (Fly.io/Render) + a minimal front-end.

## Provenance

Dissertation co-authored with Chloe Wong, Keyaan Miah, and Zoya Li, supervised
by Prof. Christophe Andrieu, University of Bristol. This repository is my own
independent re-implementation and extension of the parts of the project I was
responsible for (algorithms, algorithmic framework, references); it is not a
copy of the original team codebase.

## License

MIT — see [LICENSE](LICENSE).
