"""Prometheus metrics for the vrpsgd API: request counts, latency, and
per-algorithm optimisation performance (final objective, wall-clock time).

Exposed at GET /metrics in the standard Prometheus text exposition format.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS_TOTAL = Counter(
    "vrpsgd_requests_total", "Total requests to /optimize or /compare.", ["endpoint", "algorithm", "status"]
)

REQUEST_LATENCY = Histogram(
    "vrpsgd_request_latency_seconds", "End-to-end request latency (data fetch + optimisation).",
    ["endpoint", "algorithm"],
)

OPTIMIZATION_ITERATIONS = Histogram(
    "vrpsgd_optimization_wall_time_seconds", "Optimiser wall-clock time (excludes data fetch/network).",
    ["algorithm"],
)


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST