"""
Task Orchestrator - Управление распределением и выполнением задач в сети Daur-AI
Автор: Дауиржан Нуридинулы
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import hashlib
import random


class NodeStatus(Enum):
    """Статусы узлов в сети"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Статусы задач"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class NodeCapabilities:
    """Возможности узла"""
    cpu_cores: int
    memory_mb: int
    gpu_available: bool
    gpu_memory_mb: int
    storage_mb: int
    network_speed_mbps: float
    battery_level: int  # Для мобильных устройств
    device_type: str  # smartphone, tablet, desktop
    
    def can_handle_task(self, requirements) -> bool:
        """Проверяет, может ли узел выполнить задачу с данными требованиями"""
        return (
            self.cpu_cores >= requirements.cpu_cores and
            self.memory_mb >= requirements.memory_mb and
            self.storage_mb >= requirements.storage_mb and
            self.network_speed_mbps >= requirements.network_bandwidth_mbps and
            (not requirements.gpu_required or self.gpu_available) and
            self.battery_level > 20  # Минимальный уровень батареи
        )


@dataclass
class Node:
    """Узел в сети Daur-AI"""
    id: str
    address: str  # TON адрес
    status: NodeStatus
    capabilities: NodeCapabilities
    reputation: float
    stake_amount: float
    last_seen: datetime
    current_tasks: Set[str]
    completed_tasks: int
    failed_tasks: int
    total_earnings: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['last_seen'] = self.last_seen.isoformat()
        data['current_tasks'] = list(self.current_tasks)
        return data


@dataclass
class TaskAssignment:
    """Назначение задачи узлу"""
    task_id: str
    subtask_id: str
    node_id: str
    assigned_at: datetime
    deadline: datetime
    reward: float
    status: TaskStatus
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['assigned_at'] = self.assigned_at.isoformat()
        data['deadline'] = self.deadline.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class Bid:
    """Ставка узла на выполнение задачи"""
    node_id: str
    subtask_id: str
    bid_amount: float
    estimated_time: int
    confidence: float
    submitted_at: datetime


class TaskOrchestrator:
    """Основной класс оркестратора задач"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.assignments: Dict[str, TaskAssignment] = {}
        self.active_auctions: Dict[str, List[Bid]] = {}
        self.task_queues: Dict[str, List[str]] = {
            "high": [],
            "normal": [],
            "low": []
        }
        
        # Настройки
        self.auction_duration = 30  # секунд
        self.max_concurrent_tasks_per_node = 3
        self.reputation_threshold = 0.7
        self.min_stake_amount = 10.0
        
        # Логирование
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def register_node(self, node_data: Dict[str, Any]) -> str:
        """Регистрирует новый узел в сети"""
        
        node_id = str(uuid.uuid4())
        
        capabilities = NodeCapabilities(
            cpu_cores=node_data.get('cpu_cores', 1),
            memory_mb=node_data.get('memory_mb', 512),
            gpu_available=node_data.get('gpu_available', False),
            gpu_memory_mb=node_data.get('gpu_memory_mb', 0),
            storage_mb=node_data.get('storage_mb', 1000),
            network_speed_mbps=node_data.get('network_speed_mbps', 1.0),
            battery_level=node_data.get('battery_level', 100),
            device_type=node_data.get('device_type', 'smartphone')
        )
        
        node = Node(
            id=node_id,
            address=node_data['address'],
            status=NodeStatus.ONLINE,
            capabilities=capabilities,
            reputation=0.8,  # Начальная репутация
            stake_amount=node_data.get('stake_amount', 0.0),
            last_seen=datetime.now(),
            current_tasks=set(),
            completed_tasks=0,
            failed_tasks=0,
            total_earnings=0.0
        )
        
        self.nodes[node_id] = node
        self.logger.info(f"Node {node_id} registered successfully")
        
        return node_id
    
    async def update_node_status(self, node_id: str, status: str, 
                               capabilities: Optional[Dict] = None) -> bool:
        """Обновляет статус узла"""
        
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        node.status = NodeStatus(status)
        node.last_seen = datetime.now()
        
        if capabilities:
            # Обновляем возможности узла
            for key, value in capabilities.items():
                if hasattr(node.capabilities, key):
                    setattr(node.capabilities, key, value)
        
        self.logger.info(f"Node {node_id} status updated to {status}")
        return True
    
    async def submit_task_graph(self, task_graph: Dict[str, Any]) -> bool:
        """Принимает граф задач для выполнения"""
        
        task_id = task_graph['task_id']
        subtasks = task_graph['subtasks']
        execution_order = task_graph['execution_order']
        
        self.logger.info(f"Received task graph {task_id} with {len(subtasks)} subtasks")
        
        # Добавляем задачи в очереди по приоритету
        for level in execution_order:
            for subtask_id in level:
                subtask = next(st for st in subtasks if st['id'] == subtask_id)
                priority = subtask.get('priority', 2)  # По умолчанию normal
                
                if priority >= 3:
                    self.task_queues["high"].append(subtask_id)
                elif priority == 2:
                    self.task_queues["normal"].append(subtask_id)
                else:
                    self.task_queues["low"].append(subtask_id)
        
        # Запускаем процесс распределения задач
        asyncio.create_task(self._process_task_queue())
        
        return True
    
    async def _process_task_queue(self):
        """Обрабатывает очередь задач и запускает аукционы"""
        
        # Обрабатываем очереди по приоритету
        for priority in ["high", "normal", "low"]:
            while self.task_queues[priority]:
                subtask_id = self.task_queues[priority].pop(0)
                
                # Запускаем аукцион для подзадачи
                await self._start_auction(subtask_id)
                
                # Небольшая задержка между аукционами
                await asyncio.sleep(1)
    
    async def _start_auction(self, subtask_id: str):
        """Запускает аукцион для подзадачи"""
        
        self.logger.info(f"Starting auction for subtask {subtask_id}")
        
        # Инициализируем список ставок
        self.active_auctions[subtask_id] = []
        
        # Уведомляем подходящие узлы об аукционе
        await self._notify_eligible_nodes(subtask_id)
        
        # Ждем окончания аукциона
        await asyncio.sleep(self.auction_duration)
        
        # Выбираем победителя
        await self._select_auction_winner(subtask_id)
    
    async def _notify_eligible_nodes(self, subtask_id: str):
        """Уведомляет подходящие узлы о новом аукционе"""
        
        # Здесь должна быть логика получения требований подзадачи
        # Для примера используем базовые требования
        
        eligible_nodes = []
        
        for node_id, node in self.nodes.items():
            if (node.status == NodeStatus.ONLINE and
                len(node.current_tasks) < self.max_concurrent_tasks_per_node and
                node.reputation >= self.reputation_threshold and
                node.stake_amount >= self.min_stake_amount):
                
                eligible_nodes.append(node_id)
        
        self.logger.info(f"Found {len(eligible_nodes)} eligible nodes for subtask {subtask_id}")
        
        # В реальной реализации здесь будет отправка уведомлений узлам
        # Для демонстрации симулируем автоматические ставки
        for node_id in eligible_nodes[:5]:  # Ограничиваем количество участников
            await self._simulate_bid(node_id, subtask_id)
    
    async def _simulate_bid(self, node_id: str, subtask_id: str):
        """Симулирует ставку узла (для демонстрации)"""
        
        node = self.nodes[node_id]
        
        # Генерируем случайную ставку на основе репутации узла
        base_reward = 10.0  # Базовое вознаграждение
        bid_multiplier = 0.8 + (node.reputation * 0.4)  # 0.8 - 1.2
        bid_amount = base_reward * bid_multiplier * random.uniform(0.9, 1.1)
        
        estimated_time = random.randint(30, 300)  # 30 секунд - 5 минут
        confidence = node.reputation * random.uniform(0.8, 1.0)
        
        bid = Bid(
            node_id=node_id,
            subtask_id=subtask_id,
            bid_amount=bid_amount,
            estimated_time=estimated_time,
            confidence=confidence,
            submitted_at=datetime.now()
        )
        
        await self.submit_bid(bid)
    
    async def submit_bid(self, bid: Bid) -> bool:
        """Принимает ставку от узла"""
        
        if bid.subtask_id not in self.active_auctions:
            return False
        
        self.active_auctions[bid.subtask_id].append(bid)
        self.logger.info(f"Received bid from node {bid.node_id} for subtask {bid.subtask_id}: {bid.bid_amount}")
        
        return True
    
    async def _select_auction_winner(self, subtask_id: str):
        """Выбирает победителя аукциона"""
        
        if subtask_id not in self.active_auctions:
            return
        
        bids = self.active_auctions[subtask_id]
        
        if not bids:
            self.logger.warning(f"No bids received for subtask {subtask_id}")
            return
        
        # Алгоритм выбора победителя:
        # Комбинируем цену, репутацию, время выполнения и уверенность
        best_bid = None
        best_score = -1
        
        for bid in bids:
            node = self.nodes[bid.node_id]
            
            # Нормализуем метрики (0-1)
            price_score = 1.0 / (1.0 + bid.bid_amount / 10.0)  # Чем меньше цена, тем лучше
            reputation_score = node.reputation
            time_score = 1.0 / (1.0 + bid.estimated_time / 100.0)  # Чем быстрее, тем лучше
            confidence_score = bid.confidence
            
            # Взвешенная сумма
            total_score = (
                price_score * 0.3 +
                reputation_score * 0.4 +
                time_score * 0.2 +
                confidence_score * 0.1
            )
            
            if total_score > best_score:
                best_score = total_score
                best_bid = bid
        
        if best_bid:
            await self._assign_task(best_bid)
        
        # Очищаем аукцион
        del self.active_auctions[subtask_id]
    
    async def _assign_task(self, winning_bid: Bid):
        """Назначает задачу победителю аукциона"""
        
        assignment_id = str(uuid.uuid4())
        deadline = datetime.now() + timedelta(seconds=winning_bid.estimated_time * 2)
        
        assignment = TaskAssignment(
            task_id=assignment_id,
            subtask_id=winning_bid.subtask_id,
            node_id=winning_bid.node_id,
            assigned_at=datetime.now(),
            deadline=deadline,
            reward=winning_bid.bid_amount,
            status=TaskStatus.ASSIGNED
        )
        
        self.assignments[assignment_id] = assignment
        
        # Обновляем статус узла
        node = self.nodes[winning_bid.node_id]
        node.current_tasks.add(winning_bid.subtask_id)
        
        self.logger.info(f"Task {winning_bid.subtask_id} assigned to node {winning_bid.node_id}")
        
        # В реальной реализации здесь будет отправка задачи узлу
        # Для демонстрации симулируем выполнение
        asyncio.create_task(self._simulate_task_execution(assignment_id))
    
    async def _simulate_task_execution(self, assignment_id: str):
        """Симулирует выполнение задачи (для демонстрации)"""
        
        assignment = self.assignments[assignment_id]
        node = self.nodes[assignment.node_id]
        
        # Обновляем статус на "выполняется"
        assignment.status = TaskStatus.RUNNING
        
        # Симулируем время выполнения
        execution_time = random.randint(10, 60)  # 10-60 секунд
        await asyncio.sleep(execution_time)
        
        # Симулируем результат (успех/неудача)
        success_probability = node.reputation * 0.9  # Высокая репутация = высокая вероятность успеха
        success = random.random() < success_probability
        
        if success:
            await self._complete_task(assignment_id, "QmResultHash123")
        else:
            await self._fail_task(assignment_id, "Execution error")
    
    async def _complete_task(self, assignment_id: str, result_cid: str):
        """Завершает выполнение задачи успешно"""
        
        assignment = self.assignments[assignment_id]
        node = self.nodes[assignment.node_id]
        
        # Обновляем статус
        assignment.status = TaskStatus.COMPLETED
        
        # Обновляем статистику узла
        node.current_tasks.discard(assignment.subtask_id)
        node.completed_tasks += 1
        node.total_earnings += assignment.reward
        
        # Обновляем репутацию (небольшое увеличение)
        node.reputation = min(1.0, node.reputation + 0.01)
        
        self.logger.info(f"Task {assignment.subtask_id} completed by node {assignment.node_id}")
        
        # В реальной реализации здесь будет запись в блокчейн и выплата вознаграждения
    
    async def _fail_task(self, assignment_id: str, error_message: str):
        """Обрабатывает неудачное выполнение задачи"""
        
        assignment = self.assignments[assignment_id]
        node = self.nodes[assignment.node_id]
        
        # Обновляем статус
        assignment.status = TaskStatus.FAILED
        
        # Обновляем статистику узла
        node.current_tasks.discard(assignment.subtask_id)
        node.failed_tasks += 1
        
        # Снижаем репутацию
        node.reputation = max(0.0, node.reputation - 0.05)
        
        self.logger.warning(f"Task {assignment.subtask_id} failed on node {assignment.node_id}: {error_message}")
        
        # Возвращаем задачу в очередь для повторного назначения
        self.task_queues["normal"].append(assignment.subtask_id)
    
    async def get_network_stats(self) -> Dict[str, Any]:
        """Возвращает статистику сети"""
        
        online_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.ONLINE)
        total_tasks = sum(len(queue) for queue in self.task_queues.values())
        active_assignments = sum(1 for a in self.assignments.values() 
                               if a.status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING])
        
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": online_nodes,
            "pending_tasks": total_tasks,
            "active_assignments": active_assignments,
            "total_assignments": len(self.assignments),
            "active_auctions": len(self.active_auctions)
        }
    
    async def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию об узле"""
        
        if node_id not in self.nodes:
            return None
        
        return self.nodes[node_id].to_dict()
    
    async def get_assignment_info(self, assignment_id: str) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о назначении задачи"""
        
        if assignment_id not in self.assignments:
            return None
        
        return self.assignments[assignment_id].to_dict()
    
    async def cleanup_inactive_nodes(self):
        """Очищает неактивные узлы"""
        
        cutoff_time = datetime.now() - timedelta(minutes=10)
        inactive_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.last_seen < cutoff_time and node.status != NodeStatus.OFFLINE:
                node.status = NodeStatus.OFFLINE
                inactive_nodes.append(node_id)
        
        if inactive_nodes:
            self.logger.info(f"Marked {len(inactive_nodes)} nodes as offline")


async def main():
    """Пример использования TaskOrchestrator"""
    
    orchestrator = TaskOrchestrator()
    
    # Регистрируем несколько узлов
    nodes = []
    for i in range(5):
        node_data = {
            'address': f'EQTest{i}Address',
            'cpu_cores': random.randint(1, 4),
            'memory_mb': random.randint(512, 4096),
            'gpu_available': random.choice([True, False]),
            'storage_mb': random.randint(1000, 10000),
            'network_speed_mbps': random.uniform(1.0, 10.0),
            'battery_level': random.randint(50, 100),
            'stake_amount': random.uniform(10.0, 100.0)
        }
        
        node_id = await orchestrator.register_node(node_data)
        nodes.append(node_id)
    
    # Симулируем граф задач
    task_graph = {
        'task_id': 'test_task_001',
        'subtasks': [
            {'id': 'subtask_1', 'priority': 2},
            {'id': 'subtask_2', 'priority': 2},
            {'id': 'subtask_3', 'priority': 2},
        ],
        'execution_order': [['subtask_1'], ['subtask_2', 'subtask_3']]
    }
    
    # Отправляем задачу
    await orchestrator.submit_task_graph(task_graph)
    
    # Ждем выполнения
    await asyncio.sleep(10)
    
    # Выводим статистику
    stats = await orchestrator.get_network_stats()
    print("Network Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Выводим информацию об узлах
    print("\nNode Information:")
    for node_id in nodes:
        node_info = await orchestrator.get_node_info(node_id)
        print(f"Node {node_id}: {node_info['status']}, Reputation: {node_info['reputation']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
