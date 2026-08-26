# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/), versioning follows [SemVer](https://semver.org/).

## [0.1.0] - Unreleased

### Added
- `vrpsgd` package: `SPGD`, `ProxSVRG`, `ProxSARAH`, `ProxSTORM` as a common
  `ProximalStochasticOptimizer` interface, rewritten from the MATH30009
  dissertation's monolithic script into a tested, typed package.
- FastAPI service (`/health`, `/optimize`, `/compare`) with Docker + docker-compose.
- CLI (`vrpsgd optimize ...`) for local experimentation.
- Unit tests: proximal-operator correctness, analytic-vs-finite-difference
  gradient checks, per-optimizer convergence/determinism/contract tests, API tests.
- GitHub Actions CI: ruff, mypy, pytest with coverage, Docker build.

### Changed
- **Objective function default** now uses the true covariance matrix
  (`mode="covariance"`), matching the mean-variance derivation in the thesis.
  The original script's `R^T R / T` second-moment formulation is preserved as
  `mode="second_moment"` for reproducing the original thesis numbers exactly.
- Simplex projection keeps the "sparsity-preserving" restricted-support variant
  from the thesis (`project_simplex_sparse`), since plain projection can
  reintroduce mass onto zeroed weights and silently defeat the L1 penalty.

### Known limitations (carried over from the thesis, tracked for v0.2)
- No transaction-cost / turnover penalty in the objective.
- No cross-validated hyperparameter selection for `gamma`, `lambda`, `eta`.
- Portfolio return is approximated as `w^T r` rather than `log(sum_i w_i exp(r_i))`.
