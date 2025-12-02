"""
Daur-AI Task Orchestrator - Полнофункциональная реализация
Управление распределением задач, аукционами и координацией узлов сети
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import hmac
import secrets
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import numpy as np
import sqlite3
import redis
import websockets
import aiohttp
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import requests
from datetime import datetime, timedelta
import schedule
import asyncpg
import motor.motor_asyncio
from pymongo import MongoClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Статусы узлов сети"""
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    BANNED = "banned"

class TaskStatus(Enum):
    """Статусы задач"""
    PENDING = "pending"
    BIDDING = "bidding"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AuctionStatus(Enum):
    """Статусы аукционов"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"

@dataclass
class NodeCapabilities:
    """Возможности узла сети"""
    node_id: str
    address: str
    cpu_cores: int
    memory_mb: int
    gpu_available: bool
    gpu_memory_mb: int
    storage_mb: int
    network_speed_mbps: float
    battery_level: int
    device_type: str
    os_type: str
    last_seen: float
    reputation: float
    total_tasks_completed: int
    success_rate: float
    average_response_time: float
    stake_amount: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Bid:
    """Ставка узла на выполнение подзадачи"""
    bid_id: str
    node_id: str
    subtask_id: str
    bid_amount: float
    estimated_time: int
    confidence: float
    node_reputation: float
    timestamp: float
    signature: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TaskAssignment:
    """Назначение задачи узлу"""
    assignment_id: str
    subtask_id: str
    node_id: str
    assigned_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    status: TaskStatus
    result_hash: Optional[str]
    verification_count: int
    reward_amount: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        return result

@dataclass
class Auction:
    """Аукцион для подзадачи"""
    auction_id: str
    subtask_id: str
    min_bid: float
    max_bid: float
    duration_seconds: int
    started_at: float
    ends_at: float
    status: AuctionStatus
    bids: List[Bid]
    winner_bid_id: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['status'] = self.status.value
        result['bids'] = [bid.to_dict() for bid in self.bids]
        return result

class SecurityManager:
    """Менеджер безопасности для подписи и верификации"""
    
    def __init__(self, master_key: str = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = secrets.token_bytes(32)
        
        # Генерация ключа шифрования
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'daur_ai_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher = Fernet(key)
    
    def sign_data(self, data: Dict[str, Any], private_key: str) -> str:
        """Подпись данных"""
        message = json.dumps(data, sort_keys=True).encode()
        signature = hmac.new(
            private_key.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_signature(self, data: Dict[str, Any], signature: str, public_key: str) -> bool:
        """Проверка подписи"""
        expected_signature = self.sign_data(data, public_key)
        return hmac.compare_digest(signature, expected_signature)
    
    def encrypt_data(self, data: str) -> str:
        """Шифрование данных"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Расшифровка данных"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def generate_node_keypair(self) -> Tuple[str, str]:
        """Генерация пары ключей для узла"""
        private_key = secrets.token_hex(32)
        public_key = hashlib.sha256(private_key.encode()).hexdigest()
        return private_key, public_key

class DatabaseManager:
    """Менеджер базы данных для оркестратора"""
    
    def __init__(self, db_path: str = "orchestrator.db"):
        self.db_path = db_path
        self.redis_client = None
        self.postgres_pool = None
        self.mongo_client = None
        self._init_sqlite()
        self._init_redis()
        asyncio.create_task(self._init_postgres())
        self._init_mongodb()
    
    def _init_sqlite(self):
        """Инициализация SQLite для локального хранения"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица узлов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                address TEXT UNIQUE NOT NULL,
                capabilities TEXT NOT NULL,
                status TEXT NOT NULL,
                reputation REAL DEFAULT 0.0,
                stake_amount REAL DEFAULT 0.0,
                last_seen REAL NOT NULL,
                created_at REAL NOT NULL,
                public_key TEXT,
                encrypted_private_key TEXT
            )
        ''')
        
        # Таблица аукционов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auctions (
                auction_id TEXT PRIMARY KEY,
                subtask_id TEXT NOT NULL,
                min_bid REAL NOT NULL,
                max_bid REAL NOT NULL,
                started_at REAL NOT NULL,
                ends_at REAL NOT NULL,
                status TEXT NOT NULL,
                winner_bid_id TEXT,
                auction_data TEXT
            )
        ''')
        
        # Таблица ставок
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bids (
                bid_id TEXT PRIMARY KEY,
                auction_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                bid_amount REAL NOT NULL,
                estimated_time INTEGER NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                signature TEXT NOT NULL,
                FOREIGN KEY (auction_id) REFERENCES auctions (auction_id),
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            )
        ''')
        
        # Таблица назначений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assignments (
                assignment_id TEXT PRIMARY KEY,
                subtask_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                assigned_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                status TEXT NOT NULL,
                result_hash TEXT,
                verification_count INTEGER DEFAULT 0,
                reward_amount REAL NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            )
        ''')
        
        # Таблица репутации
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reputation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT NOT NULL,
                assignment_id TEXT,
                old_reputation REAL NOT NULL,
                new_reputation REAL NOT NULL,
                change_reason TEXT NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (node_id) REFERENCES nodes (node_id)
            )
        ''')
        
        # Индексы для производительности
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_auctions_status ON auctions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_assignments_status ON assignments(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bids_auction ON bids(auction_id)')
        
        conn.commit()
        conn.close()
        logger.info("SQLite база данных инициализирована")
    
    def _init_redis(self):
        """Инициализация Redis для кэширования и pub/sub"""
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                db=1,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis подключен")
        except Exception as e:
            logger.warning(f"Redis недоступен: {e}")
            self.redis_client = None
    
    async def _init_postgres(self):
        """Инициализация PostgreSQL для масштабируемого хранения"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='daur_ai',
                password='daur_ai_password',
                database='daur_ai_orchestrator',
                min_size=5,
                max_size=20
            )
            logger.info("PostgreSQL подключен")
        except Exception as e:
            logger.warning(f"PostgreSQL недоступен: {e}")
            self.postgres_pool = None
    
    def _init_mongodb(self):
        """Инициализация MongoDB для хранения больших объемов данных"""
        try:
            self.mongo_client = MongoClient('mongodb://localhost:27017/')
            self.mongo_db = self.mongo_client['daur_ai_orchestrator']
            # Проверка подключения
            self.mongo_client.admin.command('ping')
            logger.info("MongoDB подключен")
        except Exception as e:
            logger.warning(f"MongoDB недоступен: {e}")
            self.mongo_client = None
    
    def save_node(self, node: NodeCapabilities, private_key: str = None):
        """Сохранение узла в базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO nodes 
            (node_id, address, capabilities, status, reputation, stake_amount, 
             last_seen, created_at, public_key, encrypted_private_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            node.node_id,
            node.address,
            json.dumps(node.to_dict()),
            NodeStatus.ONLINE.value,
            node.reputation,
            node.stake_amount,
            node.last_seen,
            time.time(),
            None,  # public_key будет добавлен позже
            None   # encrypted_private_key будет добавлен позже
        ))
        
        conn.commit()
        conn.close()
        
        # Кэширование в Redis
        if self.redis_client:
            self.redis_client.setex(
                f"node:{node.node_id}", 
                3600, 
                json.dumps(node.to_dict())
            )
    
    def get_node(self, node_id: str) -> Optional[NodeCapabilities]:
        """Получение узла из базы данных"""
        # Проверка кэша
        if self.redis_client:
            cached = self.redis_client.get(f"node:{node_id}")
            if cached:
                data = json.loads(cached)
                return NodeCapabilities(**data)
        
        # Запрос к базе данных
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT capabilities FROM nodes WHERE node_id = ?', (node_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            data = json.loads(row[0])
            node = NodeCapabilities(**data)
            
            # Обновление кэша
            if self.redis_client:
                self.redis_client.setex(
                    f"node:{node_id}", 
                    3600, 
                    json.dumps(node.to_dict())
                )
            
            return node
        
        return None
    
    def save_auction(self, auction: Auction):
        """Сохранение аукциона в базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO auctions 
            (auction_id, subtask_id, min_bid, max_bid, started_at, ends_at, 
             status, winner_bid_id, auction_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            auction.auction_id,
            auction.subtask_id,
            auction.min_bid,
            auction.max_bid,
            auction.started_at,
            auction.ends_at,
            auction.status.value,
            auction.winner_bid_id,
            json.dumps(auction.to_dict())
        ))
        
        conn.commit()
        conn.close()
    
    def save_bid(self, bid: Bid, auction_id: str):
        """Сохранение ставки в базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bids 
            (bid_id, auction_id, node_id, bid_amount, estimated_time, 
             confidence, timestamp, signature)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bid.bid_id,
            auction_id,
            bid.node_id,
            bid.bid_amount,
            bid.estimated_time,
            bid.confidence,
            bid.timestamp,
            bid.signature
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_auctions(self) -> List[Auction]:
        """Получение активных аукционов"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT auction_data FROM auctions 
            WHERE status = ? AND ends_at > ?
        ''', (AuctionStatus.OPEN.value, time.time()))
        
        auctions = []
        for row in cursor.fetchall():
            data = json.loads(row[0])
            # Восстановление объектов Bid
            bids = [Bid(**bid_data) for bid_data in data['bids']]
            data['bids'] = bids
            data['status'] = AuctionStatus(data['status'])
            auctions.append(Auction(**data))
        
        conn.close()
        return auctions

class ReputationSystem:
    """Система репутации узлов"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.reputation_weights = {
            'task_completion': 0.4,
            'response_time': 0.2,
            'result_quality': 0.3,
            'uptime': 0.1
        }
    
    def calculate_reputation(self, node_id: str, task_result: Dict[str, Any]) -> float:
        """Расчет репутации на основе результата выполнения задачи"""
        current_node = self.db_manager.get_node(node_id)
        if not current_node:
            return 0.0
        
        # Базовая репутация
        base_reputation = current_node.reputation
        
        # Факторы для расчета изменения репутации
        completion_factor = 1.0 if task_result.get('success', False) else -0.5
        
        # Время выполнения (сравнение с оценкой)
        estimated_time = task_result.get('estimated_time', 60)
        actual_time = task_result.get('actual_time', 60)
        time_factor = max(0.1, min(2.0, estimated_time / max(actual_time, 1)))
        
        # Качество результата (на основе валидации)
        quality_score = task_result.get('quality_score', 0.5)
        quality_factor = (quality_score - 0.5) * 2  # Нормализация к [-1, 1]
        
        # Время отклика
        response_time = task_result.get('response_time', 5.0)
        response_factor = max(0.1, min(2.0, 5.0 / max(response_time, 0.1)))
        
        # Расчет изменения репутации
        reputation_change = (
            completion_factor * self.reputation_weights['task_completion'] +
            (time_factor - 1) * self.reputation_weights['response_time'] +
            quality_factor * self.reputation_weights['result_quality'] +
            (response_factor - 1) * self.reputation_weights['uptime']
        ) * 0.1  # Масштабирование изменения
        
        # Новая репутация (с ограничениями)
        new_reputation = max(0.0, min(1.0, base_reputation + reputation_change))
        
        # Сохранение истории изменений
        self._save_reputation_change(
            node_id, base_reputation, new_reputation, 
            f"Task completion: {task_result.get('assignment_id', 'unknown')}"
        )
        
        return new_reputation
    
    def _save_reputation_change(self, node_id: str, old_reputation: float, 
                              new_reputation: float, reason: str):
        """Сохранение изменения репутации в историю"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reputation_history 
            (node_id, old_reputation, new_reputation, change_reason, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (node_id, old_reputation, new_reputation, reason, time.time()))
        
        conn.commit()
        conn.close()
    
    def get_reputation_history(self, node_id: str, limit: int = 100) -> List[Dict]:
        """Получение истории изменений репутации"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT old_reputation, new_reputation, change_reason, timestamp
            FROM reputation_history 
            WHERE node_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (node_id, limit))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                'old_reputation': row[0],
                'new_reputation': row[1],
                'change_reason': row[2],
                'timestamp': row[3]
            })
        
        conn.close()
        return history
    
    def penalize_node(self, node_id: str, penalty_amount: float, reason: str):
        """Наказание узла за нарушения"""
        current_node = self.db_manager.get_node(node_id)
        if not current_node:
            return
        
        old_reputation = current_node.reputation
        new_reputation = max(0.0, old_reputation - penalty_amount)
        
        # Обновление репутации в базе данных
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE nodes SET reputation = ? WHERE node_id = ?
        ''', (new_reputation, node_id))
        
        conn.commit()
        conn.close()
        
        # Сохранение в историю
        self._save_reputation_change(node_id, old_reputation, new_reputation, f"Penalty: {reason}")
        
        logger.info(f"Узел {node_id} наказан: {reason}, репутация: {old_reputation:.3f} -> {new_reputation:.3f}")

class AuctionEngine:
    """Движок аукционов для распределения задач"""
    
    def __init__(self, db_manager: DatabaseManager, security_manager: SecurityManager):
        self.db_manager = db_manager
        self.security_manager = security_manager
        self.active_auctions: Dict[str, Auction] = {}
        self.auction_callbacks: Dict[str, Callable] = {}
    
    async def create_auction(self, subtask_id: str, min_bid: float, max_bid: float,
                           duration_seconds: int = 300) -> str:
        """Создание нового аукциона для подзадачи"""
        auction_id = f"auction_{uuid.uuid4().hex[:8]}"
        
        auction = Auction(
            auction_id=auction_id,
            subtask_id=subtask_id,
            min_bid=min_bid,
            max_bid=max_bid,
            duration_seconds=duration_seconds,
            started_at=time.time(),
            ends_at=time.time() + duration_seconds,
            status=AuctionStatus.OPEN,
            bids=[],
            winner_bid_id=None
        )
        
        self.active_auctions[auction_id] = auction
        self.db_manager.save_auction(auction)
        
        # Запуск таймера завершения аукциона
        asyncio.create_task(self._schedule_auction_end(auction_id, duration_seconds))
        
        logger.info(f"Создан аукцион {auction_id} для подзадачи {subtask_id}")
        return auction_id
    
    async def submit_bid(self, auction_id: str, node_id: str, bid_amount: float,
                        estimated_time: int, confidence: float, signature: str) -> bool:
        """Подача ставки на аукцион"""
        if auction_id not in self.active_auctions:
            logger.warning(f"Аукцион {auction_id} не найден")
            return False
        
        auction = self.active_auctions[auction_id]
        
        # Проверка статуса аукциона
        if auction.status != AuctionStatus.OPEN:
            logger.warning(f"Аукцион {auction_id} закрыт")
            return False
        
        # Проверка времени
        if time.time() > auction.ends_at:
            logger.warning(f"Аукцион {auction_id} истек")
            await self._end_auction(auction_id)
            return False
        
        # Проверка диапазона ставки
        if not (auction.min_bid <= bid_amount <= auction.max_bid):
            logger.warning(f"Ставка {bid_amount} вне диапазона [{auction.min_bid}, {auction.max_bid}]")
            return False
        
        # Получение информации об узле
        node = self.db_manager.get_node(node_id)
        if not node:
            logger.warning(f"Узел {node_id} не найден")
            return False
        
        # Проверка подписи ставки
        bid_data = {
            'auction_id': auction_id,
            'node_id': node_id,
            'bid_amount': bid_amount,
            'estimated_time': estimated_time,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
        # В реальной системе здесь была бы проверка подписи
        # if not self.security_manager.verify_signature(bid_data, signature, node.public_key):
        #     logger.warning(f"Неверная подпись ставки от узла {node_id}")
        #     return False
        
        # Создание ставки
        bid = Bid(
            bid_id=f"bid_{uuid.uuid4().hex[:8]}",
            node_id=node_id,
            subtask_id=auction.subtask_id,
            bid_amount=bid_amount,
            estimated_time=estimated_time,
            confidence=confidence,
            node_reputation=node.reputation,
            timestamp=bid_data['timestamp'],
            signature=signature
        )
        
        # Добавление ставки в аукцион
        auction.bids.append(bid)
        self.db_manager.save_bid(bid, auction_id)
        self.db_manager.save_auction(auction)
        
        logger.info(f"Ставка {bid.bid_id} от узла {node_id} добавлена в аукцион {auction_id}")
        return True
    
    async def _schedule_auction_end(self, auction_id: str, duration_seconds: int):
        """Планирование завершения аукциона"""
        await asyncio.sleep(duration_seconds)
        await self._end_auction(auction_id)
    
    async def _end_auction(self, auction_id: str):
        """Завершение аукциона и выбор победителя"""
        if auction_id not in self.active_auctions:
            return
        
        auction = self.active_auctions[auction_id]
        auction.status = AuctionStatus.CLOSED
        
        if not auction.bids:
            logger.warning(f"Аукцион {auction_id} завершен без ставок")
            auction.status = AuctionStatus.CANCELLED
        else:
            # Выбор победителя на основе комплексного скоринга
            winner_bid = self._select_winner(auction.bids)
            auction.winner_bid_id = winner_bid.bid_id
            
            logger.info(f"Аукцион {auction_id} выиграл узел {winner_bid.node_id} со ставкой {winner_bid.bid_amount}")
            
            # Вызов callback для обработки результата
            if auction_id in self.auction_callbacks:
                await self.auction_callbacks[auction_id](winner_bid)
        
        # Сохранение результата
        self.db_manager.save_auction(auction)
        
        # Удаление из активных аукционов
        del self.active_auctions[auction_id]
        if auction_id in self.auction_callbacks:
            del self.auction_callbacks[auction_id]
    
    def _select_winner(self, bids: List[Bid]) -> Bid:
        """Выбор победителя аукциона на основе комплексного скоринга"""
        if not bids:
            return None
        
        best_bid = None
        best_score = -1
        
        for bid in bids:
            # Комплексный скоринг учитывает:
            # 1. Цену (чем меньше, тем лучше)
            # 2. Репутацию узла (чем выше, тем лучше)
            # 3. Уверенность в выполнении (чем выше, тем лучше)
            # 4. Время выполнения (чем меньше, тем лучше)
            
            # Нормализация факторов
            price_score = 1.0 / max(bid.bid_amount, 0.01)  # Обратная зависимость от цены
            reputation_score = bid.node_reputation
            confidence_score = bid.confidence
            time_score = 1.0 / max(bid.estimated_time, 1)  # Обратная зависимость от времени
            
            # Взвешенная сумма
            total_score = (
                price_score * 0.3 +
                reputation_score * 0.4 +
                confidence_score * 0.2 +
                time_score * 0.1
            )
            
            if total_score > best_score:
                best_score = total_score
                best_bid = bid
        
        return best_bid
    
    def set_auction_callback(self, auction_id: str, callback: Callable):
        """Установка callback для обработки результата аукциона"""
        self.auction_callbacks[auction_id] = callback
    
    def get_auction_status(self, auction_id: str) -> Optional[Dict]:
        """Получение статуса аукциона"""
        if auction_id in self.active_auctions:
            return self.active_auctions[auction_id].to_dict()
        
        # Поиск в базе данных
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT auction_data FROM auctions WHERE auction_id = ?', (auction_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        
        return None

class NodeManager:
    """Менеджер узлов сети"""
    
    def __init__(self, db_manager: DatabaseManager, reputation_system: ReputationSystem):
        self.db_manager = db_manager
        self.reputation_system = reputation_system
        self.active_nodes: Dict[str, NodeCapabilities] = {}
        self.node_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.heartbeat_interval = 30  # секунд
        self.offline_threshold = 120  # секунд
        
        # Запуск мониторинга узлов
        asyncio.create_task(self._monitor_nodes())
    
    async def register_node(self, node_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Регистрация нового узла в сети"""
        try:
            # Валидация данных узла
            required_fields = ['address', 'cpu_cores', 'memory_mb', 'storage_mb', 'network_speed_mbps']
            for field in required_fields:
                if field not in node_data:
                    return False, f"Отсутствует обязательное поле: {field}"
            
            # Проверка минимальных требований
            if node_data['cpu_cores'] < 1:
                return False, "Минимум 1 ядро CPU"
            if node_data['memory_mb'] < 512:
                return False, "Минимум 512 МБ памяти"
            if node_data['storage_mb'] < 1000:
                return False, "Минимум 1 ГБ свободного места"
            
            # Генерация ID узла
            node_id = f"node_{hashlib.sha256(node_data['address'].encode()).hexdigest()[:16]}"
            
            # Создание объекта узла
            node = NodeCapabilities(
                node_id=node_id,
                address=node_data['address'],
                cpu_cores=node_data['cpu_cores'],
                memory_mb=node_data['memory_mb'],
                gpu_available=node_data.get('gpu_available', False),
                gpu_memory_mb=node_data.get('gpu_memory_mb', 0),
                storage_mb=node_data['storage_mb'],
                network_speed_mbps=node_data['network_speed_mbps'],
                battery_level=node_data.get('battery_level', 100),
                device_type=node_data.get('device_type', 'unknown'),
                os_type=node_data.get('os_type', 'unknown'),
                last_seen=time.time(),
                reputation=0.5,  # Начальная репутация
                total_tasks_completed=0,
                success_rate=0.0,
                average_response_time=0.0,
                stake_amount=node_data.get('stake_amount', 0.0)
            )
            
            # Сохранение в базу данных
            self.db_manager.save_node(node)
            self.active_nodes[node_id] = node
            
            logger.info(f"Узел {node_id} зарегистрирован: {node_data['address']}")
            return True, node_id
            
        except Exception as e:
            logger.error(f"Ошибка регистрации узла: {e}")
            return False, str(e)
    
    async def update_node_status(self, node_id: str, status_data: Dict[str, Any]):
        """Обновление статуса узла"""
        if node_id not in self.active_nodes:
            node = self.db_manager.get_node(node_id)
            if not node:
                logger.warning(f"Узел {node_id} не найден")
                return
            self.active_nodes[node_id] = node
        
        node = self.active_nodes[node_id]
        
        # Обновление данных
        node.last_seen = time.time()
        if 'battery_level' in status_data:
            node.battery_level = status_data['battery_level']
        if 'cpu_usage' in status_data:
            # Можно добавить поле cpu_usage в NodeCapabilities
            pass
        if 'memory_usage' in status_data:
            # Можно добавить поле memory_usage в NodeCapabilities
            pass
        
        # Сохранение обновлений
        self.db_manager.save_node(node)
    
    async def connect_node(self, node_id: str, websocket: websockets.WebSocketServerProtocol):
        """Подключение узла через WebSocket"""
        self.node_connections[node_id] = websocket
        
        # Обновление статуса на "онлайн"
        if node_id in self.active_nodes:
            await self.update_node_status(node_id, {'status': 'online'})
        
        logger.info(f"Узел {node_id} подключен через WebSocket")
    
    async def disconnect_node(self, node_id: str):
        """Отключение узла"""
        if node_id in self.node_connections:
            del self.node_connections[node_id]
        
        # Обновление статуса на "оффлайн"
        if node_id in self.active_nodes:
            await self.update_node_status(node_id, {'status': 'offline'})
        
        logger.info(f"Узел {node_id} отключен")
    
    async def send_message_to_node(self, node_id: str, message: Dict[str, Any]) -> bool:
        """Отправка сообщения узлу через WebSocket"""
        if node_id not in self.node_connections:
            logger.warning(f"Узел {node_id} не подключен")
            return False
        
        try:
            websocket = self.node_connections[node_id]
            await websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения узлу {node_id}: {e}")
            await self.disconnect_node(node_id)
            return False
    
    async def broadcast_message(self, message: Dict[str, Any], node_filter: Callable = None):
        """Рассылка сообщения всем подключенным узлам"""
        nodes_to_send = self.node_connections.keys()
        
        if node_filter:
            nodes_to_send = [
                node_id for node_id in nodes_to_send 
                if node_filter(self.active_nodes.get(node_id))
            ]
        
        for node_id in nodes_to_send:
            await self.send_message_to_node(node_id, message)
    
    def get_eligible_nodes(self, requirements: Dict[str, Any]) -> List[str]:
        """Получение списка узлов, подходящих для выполнения задачи"""
        eligible_nodes = []
        
        for node_id, node in self.active_nodes.items():
            # Проверка базовых требований
            if node.cpu_cores < requirements.get('cpu_cores', 1):
                continue
            if node.memory_mb < requirements.get('memory_mb', 512):
                continue
            if requirements.get('requires_gpu', False) and not node.gpu_available:
                continue
            if node.gpu_memory_mb < requirements.get('gpu_memory_mb', 0):
                continue
            if node.battery_level < requirements.get('min_battery_level', 20):
                continue
            if node.reputation < requirements.get('min_reputation', 0.1):
                continue
            
            # Проверка статуса узла
            if node_id not in self.node_connections:
                continue  # Узел не подключен
            
            # Проверка времени последней активности
            if time.time() - node.last_seen > self.offline_threshold:
                continue
            
            eligible_nodes.append(node_id)
        
        # Сортировка по репутации (лучшие первыми)
        eligible_nodes.sort(
            key=lambda node_id: self.active_nodes[node_id].reputation,
            reverse=True
        )
        
        return eligible_nodes
    
    async def _monitor_nodes(self):
        """Мониторинг состояния узлов"""
        while True:
            try:
                current_time = time.time()
                offline_nodes = []
                
                for node_id, node in self.active_nodes.items():
                    # Проверка времени последней активности
                    if current_time - node.last_seen > self.offline_threshold:
                        offline_nodes.append(node_id)
                
                # Отключение неактивных узлов
                for node_id in offline_nodes:
                    await self.disconnect_node(node_id)
                    logger.info(f"Узел {node_id} отключен по таймауту")
                
                # Отправка heartbeat активным узлам
                heartbeat_message = {
                    'type': 'heartbeat',
                    'timestamp': current_time
                }
                
                await self.broadcast_message(heartbeat_message)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Ошибка мониторинга узлов: {e}")
                await asyncio.sleep(5)

class TaskOrchestrator:
    """Основной оркестратор задач"""
    
    def __init__(self, db_path: str = "orchestrator.db"):
        self.db_manager = DatabaseManager(db_path)
        self.security_manager = SecurityManager()
        self.reputation_system = ReputationSystem(self.db_manager)
        self.auction_engine = AuctionEngine(self.db_manager, self.security_manager)
        self.node_manager = NodeManager(self.db_manager, self.reputation_system)
        
        self.assignments: Dict[str, TaskAssignment] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        
        # Запуск обработчиков
        asyncio.create_task(self._process_task_queue())
        asyncio.create_task(self._process_result_queue())
        
        logger.info("TaskOrchestrator инициализирован")
    
    async def submit_subtask(self, subtask_data: Dict[str, Any]) -> str:
        """Подача подзадачи на выполнение"""
        subtask_id = subtask_data.get('id', f"subtask_{uuid.uuid4().hex[:8]}")
        
        # Добавление в очередь обработки
        await self.task_queue.put({
            'subtask_id': subtask_id,
            'subtask_data': subtask_data,
            'submitted_at': time.time()
        })
        
        logger.info(f"Подзадача {subtask_id} добавлена в очередь")
        return subtask_id
    
    async def _process_task_queue(self):
        """Обработка очереди задач"""
        while True:
            try:
                # Получение задачи из очереди
                task_item = await self.task_queue.get()
                subtask_id = task_item['subtask_id']
                subtask_data = task_item['subtask_data']
                
                logger.info(f"Обработка подзадачи {subtask_id}")
                
                # Получение требований к ресурсам
                requirements = subtask_data.get('resources', {})
                
                # Поиск подходящих узлов
                eligible_nodes = self.node_manager.get_eligible_nodes(requirements)
                
                if not eligible_nodes:
                    logger.warning(f"Нет подходящих узлов для подзадачи {subtask_id}")
                    continue
                
                # Расчет диапазона ставок
                base_reward = subtask_data.get('reward', 10.0)
                min_bid = base_reward * 0.5
                max_bid = base_reward * 1.5
                
                # Создание аукциона
                auction_id = await self.auction_engine.create_auction(
                    subtask_id=subtask_id,
                    min_bid=min_bid,
                    max_bid=max_bid,
                    duration_seconds=300  # 5 минут
                )
                
                # Установка callback для обработки результата аукциона
                self.auction_engine.set_auction_callback(
                    auction_id,
                    lambda winner_bid: self._assign_task_to_node(subtask_id, winner_bid, subtask_data)
                )
                
                # Уведомление подходящих узлов об аукционе
                auction_notification = {
                    'type': 'auction_notification',
                    'auction_id': auction_id,
                    'subtask_id': subtask_id,
                    'min_bid': min_bid,
                    'max_bid': max_bid,
                    'requirements': requirements,
                    'deadline': time.time() + 300
                }
                
                # Отправка уведомлений только подходящим узлам
                for node_id in eligible_nodes[:20]:  # Ограничиваем количество уведомлений
                    await self.node_manager.send_message_to_node(node_id, auction_notification)
                
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"Ошибка обработки очереди задач: {e}")
                await asyncio.sleep(1)
    
    async def _assign_task_to_node(self, subtask_id: str, winner_bid: Bid, subtask_data: Dict):
        """Назначение задачи узлу-победителю аукциона"""
        assignment_id = f"assignment_{uuid.uuid4().hex[:8]}"
        
        assignment = TaskAssignment(
            assignment_id=assignment_id,
            subtask_id=subtask_id,
            node_id=winner_bid.node_id,
            assigned_at=time.time(),
            started_at=None,
            completed_at=None,
            status=TaskStatus.ASSIGNED,
            result_hash=None,
            verification_count=0,
            reward_amount=winner_bid.bid_amount
        )
        
        self.assignments[assignment_id] = assignment
        
        # Сохранение в базу данных
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assignments 
            (assignment_id, subtask_id, node_id, assigned_at, status, reward_amount)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            assignment_id, subtask_id, winner_bid.node_id,
            assignment.assigned_at, assignment.status.value, assignment.reward_amount
        ))
        
        conn.commit()
        conn.close()
        
        # Отправка задачи узлу
        task_message = {
            'type': 'task_assignment',
            'assignment_id': assignment_id,
            'subtask_id': subtask_id,
            'subtask_data': subtask_data,
            'reward_amount': winner_bid.bid_amount,
            'deadline': time.time() + winner_bid.estimated_time * 2
        }
        
        success = await self.node_manager.send_message_to_node(winner_bid.node_id, task_message)
        
        if success:
            logger.info(f"Задача {subtask_id} назначена узлу {winner_bid.node_id}")
        else:
            logger.error(f"Не удалось отправить задачу узлу {winner_bid.node_id}")
            assignment.status = TaskStatus.FAILED
    
    async def submit_task_result(self, assignment_id: str, result_data: Dict[str, Any]) -> bool:
        """Подача результата выполнения задачи"""
        if assignment_id not in self.assignments:
            logger.warning(f"Назначение {assignment_id} не найдено")
            return False
        
        assignment = self.assignments[assignment_id]
        
        # Обновление статуса
        assignment.completed_at = time.time()
        assignment.status = TaskStatus.COMPLETED
        assignment.result_hash = hashlib.sha256(
            json.dumps(result_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Добавление в очередь обработки результатов
        await self.result_queue.put({
            'assignment_id': assignment_id,
            'result_data': result_data,
            'submitted_at': time.time()
        })
        
        logger.info(f"Результат для назначения {assignment_id} получен")
        return True
    
    async def _process_result_queue(self):
        """Обработка очереди результатов"""
        while True:
            try:
                # Получение результата из очереди
                result_item = await self.result_queue.get()
                assignment_id = result_item['assignment_id']
                result_data = result_item['result_data']
                
                assignment = self.assignments.get(assignment_id)
                if not assignment:
                    continue
                
                logger.info(f"Обработка результата для назначения {assignment_id}")
                
                # Валидация результата
                is_valid = await self._validate_result(assignment, result_data)
                
                if is_valid:
                    # Обновление репутации узла
                    task_result = {
                        'success': True,
                        'assignment_id': assignment_id,
                        'estimated_time': 60,  # Получить из исходной задачи
                        'actual_time': assignment.completed_at - assignment.assigned_at,
                        'quality_score': result_data.get('quality_score', 0.8),
                        'response_time': 5.0
                    }
                    
                    new_reputation = self.reputation_system.calculate_reputation(
                        assignment.node_id, task_result
                    )
                    
                    # Обновление репутации в базе данных
                    conn = sqlite3.connect(self.db_manager.db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        'UPDATE nodes SET reputation = ? WHERE node_id = ?',
                        (new_reputation, assignment.node_id)
                    )
                    conn.commit()
                    conn.close()
                    
                    # Выплата вознаграждения (интеграция с блокчейном)
                    await self._process_reward(assignment)
                    
                    logger.info(f"Результат {assignment_id} принят, репутация узла обновлена")
                else:
                    # Наказание за некачественный результат
                    self.reputation_system.penalize_node(
                        assignment.node_id, 0.1, "Некачественный результат"
                    )
                    assignment.status = TaskStatus.FAILED
                    
                    logger.warning(f"Результат {assignment_id} отклонен")
                
                # Обновление назначения в базе данных
                conn = sqlite3.connect(self.db_manager.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE assignments 
                    SET completed_at = ?, status = ?, result_hash = ?
                    WHERE assignment_id = ?
                ''', (
                    assignment.completed_at,
                    assignment.status.value,
                    assignment.result_hash,
                    assignment_id
                ))
                conn.commit()
                conn.close()
                
                self.result_queue.task_done()
                
            except Exception as e:
                logger.error(f"Ошибка обработки очереди результатов: {e}")
                await asyncio.sleep(1)
    
    async def _validate_result(self, assignment: TaskAssignment, result_data: Dict) -> bool:
        """Валидация результата выполнения задачи"""
        # Базовые проверки
        if not result_data:
            return False
        
        # Проверка обязательных полей
        required_fields = ['result', 'execution_time', 'node_signature']
        for field in required_fields:
            if field not in result_data:
                logger.warning(f"Отсутствует поле {field} в результате")
                return False
        
        # Проверка подписи результата
        # В реальной системе здесь была бы криптографическая проверка
        
        # Проверка времени выполнения (не должно быть слишком быстрым)
        execution_time = result_data.get('execution_time', 0)
        if execution_time < 5:  # Минимум 5 секунд
            logger.warning(f"Подозрительно быстрое выполнение: {execution_time}с")
            return False
        
        # Проверка размера результата
        result_size = len(json.dumps(result_data['result']))
        if result_size < 10:  # Минимальный размер результата
            logger.warning(f"Слишком маленький результат: {result_size} байт")
            return False
        
        return True
    
    async def _process_reward(self, assignment: TaskAssignment):
        """Обработка выплаты вознаграждения"""
        # Интеграция с блокчейном TON для выплаты токенов DAUR
        # В реальной системе здесь был бы вызов смарт-контракта
        
        reward_data = {
            'node_address': assignment.node_id,  # В реальности - TON адрес
            'amount': assignment.reward_amount,
            'assignment_id': assignment.assignment_id,
            'timestamp': time.time()
        }
        
        # Логирование выплаты
        logger.info(f"Выплата {assignment.reward_amount} DAUR узлу {assignment.node_id}")
        
        # В реальной системе здесь был бы HTTP запрос к блокчейн API
        # await self._send_blockchain_transaction(reward_data)
    
    async def handle_bid_submission(self, bid_data: Dict[str, Any]) -> bool:
        """Обработка подачи ставки от узла"""
        required_fields = ['auction_id', 'node_id', 'bid_amount', 'estimated_time', 'confidence']
        for field in required_fields:
            if field not in bid_data:
                logger.warning(f"Отсутствует поле {field} в ставке")
                return False
        
        return await self.auction_engine.submit_bid(
            auction_id=bid_data['auction_id'],
            node_id=bid_data['node_id'],
            bid_amount=bid_data['bid_amount'],
            estimated_time=bid_data['estimated_time'],
            confidence=bid_data['confidence'],
            signature=bid_data.get('signature', '')
        )
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Получение статистики сети"""
        # Подсчет узлов по статусам
        online_nodes = len(self.node_manager.node_connections)
        total_nodes = len(self.node_manager.active_nodes)
        
        # Подсчет активных аукционов
        active_auctions = len(self.auction_engine.active_auctions)
        
        # Подсчет назначений по статусам
        assignments_by_status = {}
        for assignment in self.assignments.values():
            status = assignment.status.value
            assignments_by_status[status] = assignments_by_status.get(status, 0) + 1
        
        # Средняя репутация сети
        if self.node_manager.active_nodes:
            avg_reputation = sum(
                node.reputation for node in self.node_manager.active_nodes.values()
            ) / len(self.node_manager.active_nodes)
        else:
            avg_reputation = 0.0
        
        return {
            'nodes': {
                'total': total_nodes,
                'online': online_nodes,
                'offline': total_nodes - online_nodes
            },
            'auctions': {
                'active': active_auctions
            },
            'assignments': assignments_by_status,
            'network': {
                'average_reputation': avg_reputation,
                'total_capacity': {
                    'cpu_cores': sum(node.cpu_cores for node in self.node_manager.active_nodes.values()),
                    'memory_mb': sum(node.memory_mb for node in self.node_manager.active_nodes.values()),
                    'storage_mb': sum(node.storage_mb for node in self.node_manager.active_nodes.values())
                }
            },
            'timestamp': time.time()
        }
    
    def get_node_info(self, node_id: str) -> Optional[Dict]:
        """Получение информации об узле"""
        node = self.db_manager.get_node(node_id)
        if not node:
            return None
        
        # Дополнительная информация
        is_online = node_id in self.node_manager.node_connections
        reputation_history = self.reputation_system.get_reputation_history(node_id, 10)
        
        result = node.to_dict()
        result.update({
            'is_online': is_online,
            'reputation_history': reputation_history
        })
        
        return result
    
    def get_assignment_info(self, assignment_id: str) -> Optional[Dict]:
        """Получение информации о назначении"""
        if assignment_id in self.assignments:
            return self.assignments[assignment_id].to_dict()
        
        # Поиск в базе данных
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM assignments WHERE assignment_id = ?', (assignment_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'assignment_id': row[0],
                'subtask_id': row[1],
                'node_id': row[2],
                'assigned_at': row[3],
                'started_at': row[4],
                'completed_at': row[5],
                'status': row[6],
                'result_hash': row[7],
                'verification_count': row[8],
                'reward_amount': row[9]
            }
        
        return None
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """Очистка старых данных"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Очистка старых аукционов
        cursor.execute('DELETE FROM auctions WHERE started_at < ?', (cutoff_time,))
        
        # Очистка старых ставок
        cursor.execute('''
            DELETE FROM bids WHERE bid_id IN (
                SELECT b.bid_id FROM bids b
                JOIN auctions a ON b.auction_id = a.auction_id
                WHERE a.started_at < ?
            )
        ''', (cutoff_time,))
        
        # Очистка завершенных назначений
        cursor.execute('''
            DELETE FROM assignments 
            WHERE status IN ('completed', 'failed') AND assigned_at < ?
        ''', (cutoff_time,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Очищены данные старше {max_age_hours} часов")
    
    async def shutdown(self):
        """Корректное завершение работы оркестратора"""
        # Закрытие всех активных аукционов
        for auction_id in list(self.auction_engine.active_auctions.keys()):
            await self.auction_engine._end_auction(auction_id)
        
        # Отключение всех узлов
        for node_id in list(self.node_manager.node_connections.keys()):
            await self.node_manager.disconnect_node(node_id)
        
        # Закрытие соединений с базами данных
        if self.db_manager.redis_client:
            self.db_manager.redis_client.close()
        
        if self.db_manager.postgres_pool:
            await self.db_manager.postgres_pool.close()
        
        if self.db_manager.mongo_client:
            self.db_manager.mongo_client.close()
        
        logger.info("TaskOrchestrator завершил работу")

# Функция для тестирования
async def main():
    """Тестирование TaskOrchestrator"""
    orchestrator = TaskOrchestrator()
    
    # Регистрация тестового узла
    node_data = {
        'address': 'test_node_001',
        'cpu_cores': 4,
        'memory_mb': 8192,
        'gpu_available': True,
        'gpu_memory_mb': 4096,
        'storage_mb': 50000,
        'network_speed_mbps': 100.0,
        'battery_level': 85,
        'device_type': 'smartphone',
        'stake_amount': 100.0
    }
    
    success, node_id = await orchestrator.node_manager.register_node(node_data)
    print(f"Регистрация узла: {success}, ID: {node_id}")
    
    # Подача тестовой подзадачи
    subtask_data = {
        'id': 'test_subtask_001',
        'type': 'text_processing',
        'input_data': {'text': 'Тестовый текст для обработки'},
        'resources': {
            'cpu_cores': 2,
            'memory_mb': 1024,
            'requires_gpu': False
        },
        'reward': 50.0
    }
    
    subtask_id = await orchestrator.submit_subtask(subtask_data)
    print(f"Подзадача отправлена: {subtask_id}")
    
    # Получение статистики
    stats = orchestrator.get_network_statistics()
    print(f"Статистика сети: {json.dumps(stats, indent=2)}")
    
    # Ожидание обработки
    await asyncio.sleep(2)
    
    await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
