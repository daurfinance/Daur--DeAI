"""
node_client.py — Реальная интеграция ноды с ядром ИИ и блокчейном
"""
from ai_core.reverse_ai import generate_paths, analyze_patterns
from ai_core.blockchain_client import BlockchainClient
import hashlib

class NodeClient:
    def __init__(self, blockchain: BlockchainClient, private_key: str):
        self.blockchain = blockchain
        self.private_key = private_key

    def process_task(self, target_result: str, dataset_chunk: bytes, reward: int):
        # 1. Генерация путей
        paths = generate_paths(target_result, dataset_chunk)
        # 2. Анализ закономерностей
        concept = analyze_patterns(paths)
        # 3. Хэш результата
        partial_hash = hashlib.sha256(str(concept).encode()).hexdigest()
        # 4. Создание задачи в блокчейне
        task_hash = hashlib.sha256(dataset_chunk).hexdigest()
        tx_task = self.blockchain.create_task(task_hash, reward, self.private_key)
        print(f"Task created: {tx_task}")
        # 5. Отправка partial
        tx_partial = self.blockchain.submit_partial(0, 0, partial_hash, b"zk_proof", self.private_key)
        print(f"Partial submitted: {tx_partial}")
        return tx_task, tx_partial
