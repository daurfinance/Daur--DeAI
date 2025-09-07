"""
orchestrator.py — Реальный оркестратор для Daur-AI
"""

from ai_core.reverse_ai import mini_llm_planner
from nodes_sdk.node_client import NodeClient
from ai_core.blockchain_client import BlockchainClient
import hashlib
from prometheus_client import Counter, start_http_server

tasks_processed = Counter('tasks_processed_total', 'Total tasks processed')
node_active_count = Counter('node_active_count', 'Active node count')
token_transfer_total = Counter('token_transfer_total', 'Total token transfers')

start_http_server(9100)

class Orchestrator:
    def __init__(self, node_clients):
        self.node_clients = node_clients  # список NodeClient

    def process_task(self, task_spec: dict, dataset: bytes, reward: int):
        tasks_processed.inc()
        node_active_count.inc(len(self.node_clients))
        # 1. Планирование: разбить на чанки и выбрать стратегию
        plan = mini_llm_planner(task_spec)
        chunks = plan.chunks
        strategy = plan.strategy
        print(f"Orchestrator: strategy={strategy}, chunks={len(chunks)}")
        # 2. Распределение чанков по нодам
        partials = []
        for i, chunk in enumerate(chunks):
            node = self.node_clients[i % len(self.node_clients)]
            tx_task, tx_partial = node.process_task(task_spec['target_result'], chunk.encode() if isinstance(chunk, str) else chunk, reward)
            partials.append(tx_partial)
        # 3. Агрегация partials
        agg_hash = hashlib.sha256(''.join(partials).encode()).hexdigest()
        # 4. Верификация и распределение наград
        node = self.node_clients[0]
        node.blockchain.aggregate_and_verify(0, agg_hash, node.private_key)
        node.blockchain.distribute_rewards(0, node.private_key)
        print(f"Orchestrator: Aggregation and rewards done.")
        return agg_hash
