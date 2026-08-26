from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import vrpsgd.api.main as api_main


@pytest.fixture
def client(monkeypatch):
    def fake_load_return_matrix(tickers, start, end, use_cache=True):
        rng = np.random.RandomState(0)
        R = rng.randn(250, len(tickers)) * 0.01
        return R.astype(np.float64), list(tickers)

    monkeypatch.setattr(api_main, "load_return_matrix", fake_load_return_matrix)
    return TestClient(api_main.app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_optimize_returns_valid_portfolio(client):
    body = {
        "tickers": ["AAA", "BBB", "CCC", "DDD"],
        "start": "2020-01-01",
        "end": "2021-01-01",
        "algorithm": "prox-svrg",
        "t_max": 200,
    }
    resp = client.post("/optimize", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["algorithm"] == "Prox-SVRG"
    assert len(data["weights"]) == 4
    total = sum(w["weight"] for w in data["weights"])
    assert total == pytest.approx(1.0, abs=1e-4)


def test_optimize_rejects_unknown_algorithm(client):
    body = {
        "tickers": ["AAA", "BBB"],
        "start": "2020-01-01",
        "end": "2021-01-01",
        "algorithm": "not-a-real-algorithm",
    }
    resp = client.post("/optimize", json=body)
    assert resp.status_code == 422  # pydantic literal validation


def test_compare_runs_all_algorithms(client):
    body = {
        "tickers": ["AAA", "BBB", "CCC"],
        "start": "2020-01-01",
        "end": "2021-01-01",
        "t_max": 150,
    }
    resp = client.post("/compare", json=body)
    assert resp.status_code == 200
    algos = {r["algorithm"] for r in resp.json()["results"]}
    assert algos == {"SPGD", "Prox-SVRG", "Prox-SARAH", "Prox-STORM"}
