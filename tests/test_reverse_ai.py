"""
Тесты для reverse_ai.py — ядра реверсивного ИИ Daur-AI
"""
import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_core.reverse_ai import generate_paths, analyze_patterns, mini_llm_planner, PathHypothesis, TaskPlan

def test_generate_paths_empty():
    result = generate_paths("target", b"data")
    assert isinstance(result, list)
    assert all(isinstance(p, PathHypothesis) for p in result)

def test_analyze_patterns_empty():
    paths = []
    patterns = analyze_patterns(paths)
    assert isinstance(patterns, dict)

def test_mini_llm_planner_empty():
    plan = mini_llm_planner({"task": "test"})
    assert isinstance(plan, TaskPlan)
    assert hasattr(plan, "chunks")
    assert hasattr(plan, "strategy")
