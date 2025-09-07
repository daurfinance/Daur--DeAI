"""
privacy_utils.py — Дифференциальная приватность для Daur-AI
"""
import numpy as np

def add_differential_privacy(embeddings, epsilon=1.0):
    """
    Добавляет дифференциальный шум к эмбеддингам для анонимизации.
    Args:
        embeddings: np.ndarray, shape (N, D)
        epsilon: float, параметр приватности
    Returns:
        np.ndarray, эмбеддинги с шумом
    """
    sensitivity = np.max(np.abs(embeddings))
    noise = np.random.laplace(loc=0.0, scale=sensitivity/epsilon, size=embeddings.shape)
    return embeddings + noise

# Пример интеграции:
# from ai_core.privacy_utils import add_differential_privacy
# private_embeddings = add_differential_privacy(embeddings, epsilon=0.5)
