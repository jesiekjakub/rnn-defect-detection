"""Smoke tests for every dashboard endpoint."""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient

from rnn_defect_detection.config import NUM_DEFECT_TYPES, NUM_SENSORS


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["models_loaded"] is True
    assert body["cache_ready"] is True


def test_sample_generates_valid_sequence(client: TestClient) -> None:
    response = client.post(
        "/api/sample",
        json={
            "defects": [True, False, False, False, False],
            "seq_len": 40,
            "seed": 123,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["x"]) == 40
    assert all(len(row) == NUM_SENSORS for row in body["x"])
    assert body["y_true"] == [1, 0, 0, 0, 0]


def test_sample_rejects_wrong_defect_length(client: TestClient) -> None:
    response = client.post(
        "/api/sample",
        json={"defects": [True, False], "seq_len": 30, "seed": 1},
    )
    assert response.status_code == 422


def test_predict_compare_returns_both_approaches(client: TestClient) -> None:
    seq_resp = client.post(
        "/api/sample",
        json={"defects": [True, False, False, False, False], "seq_len": 32, "seed": 7},
    )
    sequence = seq_resp.json()

    response = client.post("/api/predict/compare", json=sequence)
    assert response.status_code == 200
    body = response.json()
    assert len(body["attention"]["probs"]) == NUM_DEFECT_TYPES
    assert len(body["seq2seq"]["probs"]) == NUM_DEFECT_TYPES
    assert len(body["agreement"]) == NUM_DEFECT_TYPES


def test_predict_stream_returns_snapshots(client: TestClient) -> None:
    seq_resp = client.post(
        "/api/sample",
        json={"defects": [False, False, False, False, False], "seq_len": 30, "seed": 2},
    )
    sequence = seq_resp.json()
    response = client.post(
        "/api/predict/stream",
        json={"sequence": sequence, "window_size": 10, "stride": 5},
    )
    assert response.status_code == 200
    snapshots = response.json()["snapshots"]
    assert len(snapshots) >= 2
    assert snapshots[0]["t"] == 9


def test_upload_csv_parses_single_sequence(client: TestClient) -> None:
    csv = "t,sensor_0,sensor_1,sensor_2\n"
    for t in range(20):
        csv += f"{t},{t * 0.1},{1.0},{-1.0}\n"
    files = {"file": ("sample.csv", csv.encode(), "text/csv")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
    body = response.json()
    assert len(body["sequences"]) == 1
    assert len(body["sequences"][0]["x"]) == 20


def test_upload_json_parses_list(client: TestClient) -> None:
    payload = json.dumps(
        [
            {"x": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], "y_true": [1, 0, 0, 0, 0]},
            {"x": [[0.1, 0.2, 0.3]] * 15},
        ]
    )
    files = {"file": ("sample.json", payload.encode(), "application/json")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 200
    sequences = response.json()["sequences"]
    assert len(sequences) == 2
    assert sequences[0]["y_true"] == [1, 0, 0, 0, 0]


def test_upload_rejects_too_large(client: TestClient) -> None:
    blob = io.BytesIO(b"x" * (6 * 1024 * 1024))
    response = client.post("/api/upload", files={"file": ("big.csv", blob, "text/csv")})
    assert response.status_code == 413


def test_threshold_evaluate_returns_metrics(client: TestClient) -> None:
    response = client.post(
        "/api/threshold/evaluate",
        json={"thresholds": [0.5] * NUM_DEFECT_TYPES, "approach": "attention"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["per_class"]) == NUM_DEFECT_TYPES
    assert 0.0 <= body["macro_f1"] <= 1.0


def test_threshold_curves_match_grid(client: TestClient) -> None:
    response = client.get("/api/threshold/curves")
    assert response.status_code == 200
    curves = response.json()["per_class_curves"]
    assert len(curves) == NUM_DEFECT_TYPES
    assert len(curves[0]) == 51  # threshold grid of 0..1 in 0.02 steps


def test_latent_returns_points(client: TestClient) -> None:
    response = client.get("/api/latent")
    assert response.status_code == 200
    body = response.json()
    assert len(body["points"]) == 24


def test_latent_get_sample_runs_inference(client: TestClient) -> None:
    response = client.get("/api/latent/sample/3")
    assert response.status_code == 200
    body = response.json()
    assert body["sequence"]["origin"] == "latent"
    assert len(body["attention"]["probs"]) == NUM_DEFECT_TYPES


def test_batch_paginates(client: TestClient) -> None:
    response = client.get("/api/batch?limit=5&offset=0")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 0
    assert len(body["rows"]) <= 5


def test_architecture_lists_two_approaches(client: TestClient) -> None:
    response = client.get("/api/architecture")
    assert response.status_code == 200
    approaches = response.json()["approaches"]
    assert len(approaches) == 2


def test_metrics_returns_empty_when_no_file(client: TestClient) -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["attention"] is None
    assert body["seq2seq"] is None
