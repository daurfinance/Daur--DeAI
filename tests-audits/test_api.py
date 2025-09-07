"""
test_api.py — Тесты для FastAPI API Daur-AI
"""
import pytest
from fastapi.testclient import TestClient
from api-marketplace.api import app

client = TestClient(app)

def test_create_task():
    response = client.post("/tasks", json={
        "target_result": "generate_code",
        "data": "dGVzdF9kYXRh",  # base64 test data
        "reward": 1000
    })
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "agg_hash" in data

def test_get_task():
    # Сначала создаём задачу
    response = client.post("/tasks", json={
        "target_result": "generate_code",
        "data": "dGVzdF9kYXRh",
        "reward": 1000
    })
    task_id = response.json()["task_id"]
    # Получаем результат
    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "agg_hash" in data
