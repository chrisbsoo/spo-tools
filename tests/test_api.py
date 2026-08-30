from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import vrpsgd.api.main as api_main


@pytest.fixture
def client(monkeypatch):
    def fake_load_return_matrix(tickers, start, end, use_cache=True):
        rng = np.random.RandomState(0)
        R = rng.randn(250, len(tickers)) * 0.01
        return R.astype(np.float64), list(tickers)

    def fake_load_dated_returns(tickers, start, end, use_cache=True):
        rng = np.random.RandomState(0)
        dates = pd.date_range(start, periods=40, freq="B")
        data = {t: rng.randn(40) * 0.01 for t in tickers}
        return pd.DataFrame(data, index=dates)

    monkeypatch.setattr(api_main, "load_return_matrix", fake_load_return_matrix)
    monkeypatch.setattr(api_main, "load_dated_returns", fake_load_dated_returns)
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

def test_returns_gives_date_aligned_rows(client):
    body = {"tickers": ["AAA", "BBB"], "start": "2024-01-01", "end": "2024-03-01"}
    resp = client.post("/returns", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rows"]) == 40
    first_row = data["rows"][0]
    assert set(first_row["returnsByTicker"].keys()) == {"AAA", "BBB"}
    # rows must be in chronological order — downstream drift logic assumes this
    dates = [row["date"] for row in data["rows"]]
    assert dates == sorted(dates)


def test_returns_rejects_empty_ticker_list(client):
    body = {"tickers": [], "start": "2024-01-01", "end": "2024-03-01"}
    resp = client.post("/returns", json=body)
    assert resp.status_code == 422


def test_returns_surfaces_data_fetch_failures_as_502(client, monkeypatch):
    def failing_load_dated_returns(tickers, start, end, use_cache=True):
        raise RuntimeError("simulated upstream failure")

    monkeypatch.setattr(api_main, "load_dated_returns", failing_load_dated_returns)

    body = {"tickers": ["AAA"], "start": "2024-01-01", "end": "2024-03-01"}
    resp = client.post("/returns", json=body)
    assert resp.status_code == 502
