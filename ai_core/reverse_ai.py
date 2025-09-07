"""
reverse_ai.py — Ядро реверсивного ИИ для Daur-AI

Функции:
- generate_paths: генерация гипотез путей достижения результата
- analyze_patterns: анализ закономерностей между путями
- mini_llm_planner: планирование задачи через мини-LLM
"""
from typing import List, Dict, Any

class PathHypothesis:
    def __init__(self, steps: List[str], result: str, metrics: Dict[str, float]):
        self.steps = steps
        self.result = result
        self.metrics = metrics

class TaskPlan:
    def __init__(self, chunks: List[Any], strategy: str):
        self.chunks = chunks
        self.strategy = strategy

from ai_core.model_utils import load_lora_model, prepare_qat_model

import random
import string

def generate_paths(target_result: str, dataset_chunk: bytes) -> List[PathHypothesis]:
    """
    Генерирует 5–10 вариантов путей достижения целевого результата.
    Имитация: случайные шаги, метрики.
    """
    from llama_cpp import Llama
    import json
    import concurrent.futures
    # Путь к вашей квантованной модели GGML
    model_path = "mini-llm-ggml.bin"
    llm = Llama(model_path=model_path, n_ctx=2048)
    prompt = (
        f"You are an AI planner. Given the target result: '{target_result}' and a dataset chunk, "
        "generate a step-by-step hypothesis (path) to achieve the result. "
        "Return a JSON with: steps (list of strings), output (string)."
    )
    N = 10
    def gen_one():
        response = llm(prompt)
        try:
            data = json.loads(response["choices"][0]["text"])
            steps = data.get("steps", [])
            output = data.get("output", "")
            metrics = {}
            return PathHypothesis(steps, output, metrics)
        except Exception:
            return None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda _: gen_one(), range(N)))
    return [r for r in results if r]

def analyze_patterns(paths: List[PathHypothesis]) -> Dict[str, float]:
    """
    Анализирует сходства, различия, повторяемость между путями.
    Использует SentenceTransformers и Levenshtein.
    """
    if not paths:
        return {}
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import Levenshtein

    outputs = [p.result for p in paths]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(outputs)
    sim_matrix = cosine_similarity(embeddings)
    similarities = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            similarities.append(sim_matrix[i][j])
    avg_similarity = float(np.mean(similarities)) if similarities else 0.0
    differences = []
    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            differences.append(Levenshtein.distance(outputs[i], outputs[j]))
    avg_difference = float(np.mean(differences)) if differences else 0.0
    step_sets = [tuple(p.steps) for p in paths]
    repeat_steps = len(set(step_sets)) < len(paths)
    # Novelty scoring: энтропия эмбеддингов
    def entropy(vectors):
        # Простая оценка энтропии по распределению значений
        flat = np.concatenate(vectors)
        hist, _ = np.histogram(flat, bins=20)
        probs = hist / np.sum(hist)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])
    novelty_score = entropy(embeddings)
    return {
        "avg_similarity": avg_similarity,
        "avg_difference": avg_difference,
        "repeat_steps": repeat_steps,
        "novelty_score": novelty_score
    }

def mini_llm_planner(task_spec: Dict) -> TaskPlan:
    """
    Использует мини-LLM для разбиения задачи на чанки и выбора стратегии.
    """
    from llama_cpp import Llama
    import json
    model_path = "mini-llm-ggml.bin"
    llm = Llama(model_path=model_path, n_ctx=2048)
    prompt = (
        f"You are a task planner. Given the task spec: {json.dumps(task_spec)}, "
        "split the data into optimal chunks (max_size=1024 bytes), and choose a strategy ('fast' or 'deep') based on complexity. "
        "Return a JSON: {chunks: [chunk1, chunk2, ...], strategy: 'fast' or 'deep'}"
    )
    response = llm(prompt)
    try:
        data = json.loads(response["choices"][0]["text"])
        chunks = data.get("chunks", [])
        strategy = data.get("strategy", "fast")
        return TaskPlan(chunks, strategy)
    except Exception as e:
        return TaskPlan([], "fast")
