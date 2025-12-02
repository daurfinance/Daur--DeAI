"""
Daur-AI API Server - Полнофункциональная реализация
REST API и WebSocket сервер для взаимодействия с сетью Daur-AI
"""

import asyncio
import json
import logging
import time
import uuid
import hashlib
import hmac
import secrets
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import jwt
import bcrypt
import aiofiles
import aioredis
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, Form, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GzipMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import websockets
from contextlib import asynccontextmanager
import sqlite3
import redis
import requests
import numpy as np
from PIL import Image
import io
import base64
import mimetypes
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import ssl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ipfshttpclient
from web3 import Web3
import asyncio_mqtt
import aiohttp
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Настройка структурированного логирования
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Метрики Prometheus
REQUEST_COUNT = Counter('daur_ai_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('daur_ai_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('daur_ai_active_connections', 'Active WebSocket connections')
ACTIVE_NODES = Gauge('daur_ai_active_nodes', 'Active nodes in network')
PENDING_TASKS = Gauge('daur_ai_pending_tasks', 'Pending tasks in queue')

# Модели данных
class UserRegistration(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None
    organization: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class TaskSubmission(BaseModel):
    task_type: str = Field(..., regex=r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    input_data: Dict[str, Any]
    priority: int = Field(default=1, ge=1, le=10)
    max_reward: float = Field(..., gt=0)
    deadline: Optional[datetime] = None
    requirements: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class NodeRegistration(BaseModel):
    device_id: str
    device_type: str
    cpu_cores: int = Field(..., ge=1)
    memory_mb: int = Field(..., ge=512)
    storage_mb: int = Field(..., ge=1000)
    gpu_available: bool = False
    gpu_memory_mb: int = Field(default=0, ge=0)
    network_speed_mbps: float = Field(..., gt=0)
    battery_level: int = Field(default=100, ge=0, le=100)
    os_type: str
    stake_amount: float = Field(default=0.0, ge=0)

class BidSubmission(BaseModel):
    auction_id: str
    bid_amount: float = Field(..., gt=0)
    estimated_time: int = Field(..., gt=0)
    confidence: float = Field(..., ge=0, le=1)

class TaskResult(BaseModel):
    assignment_id: str
    result_data: Dict[str, Any]
    execution_time: int
    quality_metrics: Optional[Dict[str, float]] = None

class DatabaseManager:
    """Менеджер баз данных"""
    
    def __init__(self):
        self.sqlite_path = "api_server.db"
        self.redis_client = None
        self.postgres_pool = None
        self.mongo_client = None
        self.ipfs_client = None
        self._init_sqlite()
        
    async def initialize_async_connections(self):
        """Инициализация асинхронных соединений"""
        await self._init_redis()
        await self._init_postgres()
        await self._init_mongodb()
        await self._init_ipfs()
    
    def _init_sqlite(self):
        """Инициализация SQLite"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        # Таблица пользователей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                organization TEXT,
                api_key TEXT UNIQUE,
                created_at REAL NOT NULL,
                last_login REAL,
                is_active BOOLEAN DEFAULT 1,
                role TEXT DEFAULT 'user'
            )
        ''')
        
        # Таблица сессий
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at REAL NOT NULL,
                created_at REAL NOT NULL,
                last_activity REAL NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Таблица задач
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                input_data TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                max_reward REAL NOT NULL,
                deadline REAL,
                requirements TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                result_data TEXT,
                total_cost REAL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Таблица узлов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                user_id TEXT,
                device_id TEXT UNIQUE NOT NULL,
                device_type TEXT NOT NULL,
                capabilities TEXT NOT NULL,
                status TEXT DEFAULT 'offline',
                reputation REAL DEFAULT 0.5,
                total_earnings REAL DEFAULT 0,
                tasks_completed INTEGER DEFAULT 0,
                last_seen REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Таблица файлов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                mime_type TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                ipfs_hash TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Таблица уведомлений
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                type TEXT NOT NULL,
                is_read BOOLEAN DEFAULT 0,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_files_user_id ON files(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON notifications(user_id)')
        
        conn.commit()
        conn.close()
        logger.info("SQLite база данных инициализирована")
    
    async def _init_redis(self):
        """Инициализация Redis"""
        try:
            self.redis_client = await aioredis.from_url(
                "redis://localhost:6379/0",
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            await self.redis_client.ping()
            logger.info("Redis подключен")
        except Exception as e:
            logger.warning("Redis недоступен", error=str(e))
            self.redis_client = None
    
    async def _init_postgres(self):
        """Инициализация PostgreSQL"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host='localhost',
                port=5432,
                user='daur_ai',
                password='daur_ai_password',
                database='daur_ai_api',
                min_size=5,
                max_size=20
            )
            logger.info("PostgreSQL подключен")
        except Exception as e:
            logger.warning("PostgreSQL недоступен", error=str(e))
            self.postgres_pool = None
    
    async def _init_mongodb(self):
        """Инициализация MongoDB"""
        try:
            self.mongo_client = AsyncIOMotorClient('mongodb://localhost:27017/')
            await self.mongo_client.admin.command('ping')
            self.mongo_db = self.mongo_client['daur_ai_api']
            logger.info("MongoDB подключен")
        except Exception as e:
            logger.warning("MongoDB недоступен", error=str(e))
            self.mongo_client = None
    
    async def _init_ipfs(self):
        """Инициализация IPFS"""
        try:
            self.ipfs_client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001')
            # Проверка подключения
            self.ipfs_client.version()
            logger.info("IPFS подключен")
        except Exception as e:
            logger.warning("IPFS недоступен", error=str(e))
            self.ipfs_client = None

class AuthenticationManager:
    """Менеджер аутентификации и авторизации"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.jwt_secret = os.getenv('JWT_SECRET', secrets.token_hex(32))
        self.jwt_algorithm = 'HS256'
        self.token_expire_hours = 24
        
        # Инициализация шифрования
        self.master_key = os.getenv('MASTER_KEY', secrets.token_bytes(32))
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'daur_ai_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        self.cipher = Fernet(key)
    
    def hash_password(self, password: str) -> str:
        """Хеширование пароля"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Проверка пароля"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_api_key(self) -> str:
        """Генерация API ключа"""
        return f"daur_{secrets.token_hex(32)}"
    
    def create_jwt_token(self, user_id: str, username: str) -> str:
        """Создание JWT токена"""
        payload = {
            'user_id': user_id,
            'username': username,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expire_hours),
            'iat': datetime.utcnow(),
            'iss': 'daur-ai-api'
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Проверка JWT токена"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT токен истек")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Неверный JWT токен")
            return None
    
    async def register_user(self, user_data: UserRegistration) -> Tuple[bool, str]:
        """Регистрация пользователя"""
        try:
            # Проверка существования пользователя
            conn = sqlite3.connect(self.db_manager.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT username FROM users WHERE username = ? OR email = ?', 
                         (user_data.username, user_data.email))
            if cursor.fetchone():
                conn.close()
                return False, "Пользователь с таким именем или email уже существует"
            
            # Создание пользователя
            user_id = f"user_{uuid.uuid4().hex[:16]}"
            password_hash = self.hash_password(user_data.password)
            api_key = self.generate_api_key()
            
            cursor.execute('''
                INSERT INTO users 
                (user_id, username, email, password_hash, full_name, organization, api_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, user_data.username, user_data.email, password_hash,
                user_data.full_name, user_data.organization, api_key, time.time()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("Пользователь зарегистрирован", user_id=user_id, username=user_data.username)
            return True, user_id
            
        except Exception as e:
            logger.error("Ошибка регистрации пользователя", error=str(e))
            return False, str(e)
    
    async def authenticate_user(self, login_data: UserLogin) -> Tuple[bool, Optional[str], Optional[str]]:
        """Аутентификация пользователя"""
        try:
            conn = sqlite3.connect(self.db_manager.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, password_hash, is_active 
                FROM users WHERE username = ?
            ''', (login_data.username,))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False, None, "Пользователь не найден"
            
            user_id, password_hash, is_active = row
            
            if not is_active:
                conn.close()
                return False, None, "Аккаунт заблокирован"
            
            if not self.verify_password(login_data.password, password_hash):
                conn.close()
                return False, None, "Неверный пароль"
            
            # Обновление времени последнего входа
            cursor.execute('UPDATE users SET last_login = ? WHERE user_id = ?', 
                         (time.time(), user_id))
            conn.commit()
            conn.close()
            
            # Создание JWT токена
            token = self.create_jwt_token(user_id, login_data.username)
            
            # Сохранение сессии
            await self._create_session(user_id, token)
            
            logger.info("Пользователь аутентифицирован", user_id=user_id, username=login_data.username)
            return True, token, None
            
        except Exception as e:
            logger.error("Ошибка аутентификации", error=str(e))
            return False, None, str(e)
    
    async def _create_session(self, user_id: str, token: str, ip_address: str = None, user_agent: str = None):
        """Создание сессии"""
        session_id = f"session_{uuid.uuid4().hex[:16]}"
        expires_at = time.time() + (self.token_expire_hours * 3600)
        
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sessions 
            (session_id, user_id, token, expires_at, created_at, last_activity, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, user_id, token, expires_at, time.time(), time.time(),
            ip_address, user_agent
        ))
        
        conn.commit()
        conn.close()
        
        # Кэширование в Redis
        if self.db_manager.redis_client:
            await self.db_manager.redis_client.setex(
                f"session:{token}", 
                self.token_expire_hours * 3600,
                json.dumps({'user_id': user_id, 'session_id': session_id})
            )
    
    async def get_current_user(self, token: str) -> Optional[Dict[str, Any]]:
        """Получение текущего пользователя по токену"""
        # Проверка в кэше Redis
        if self.db_manager.redis_client:
            cached = await self.db_manager.redis_client.get(f"session:{token}")
            if cached:
                session_data = json.loads(cached)
                user_id = session_data['user_id']
                
                # Получение данных пользователя
                conn = sqlite3.connect(self.db_manager.sqlite_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, username, email, full_name, organization, role, api_key
                    FROM users WHERE user_id = ? AND is_active = 1
                ''', (user_id,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return {
                        'user_id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'organization': row[4],
                        'role': row[5],
                        'api_key': row[6]
                    }
        
        # Проверка JWT токена
        payload = self.verify_jwt_token(token)
        if not payload:
            return None
        
        # Получение пользователя из базы данных
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_id, username, email, full_name, organization, role, api_key
            FROM users WHERE user_id = ? AND is_active = 1
        ''', (payload['user_id'],))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'user_id': row[0],
                'username': row[1],
                'email': row[2],
                'full_name': row[3],
                'organization': row[4],
                'role': row[5],
                'api_key': row[6]
            }
        
        return None
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Шифрование чувствительных данных"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Расшифровка чувствительных данных"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()

class FileManager:
    """Менеджер файлов"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.upload_dir = "uploads"
        self.max_file_size = 100 * 1024 * 1024  # 100 MB
        self.allowed_extensions = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
            'document': ['.pdf', '.doc', '.docx', '.txt', '.md'],
            'data': ['.json', '.csv', '.xml', '.yaml'],
            'archive': ['.zip', '.tar', '.gz']
        }
        
        # Создание директории для загрузок
        os.makedirs(self.upload_dir, exist_ok=True)
    
    async def upload_file(self, file: UploadFile, user_id: str) -> Tuple[bool, str, Optional[str]]:
        """Загрузка файла"""
        try:
            # Проверка размера файла
            if file.size > self.max_file_size:
                return False, "Файл слишком большой", None
            
            # Проверка расширения
            file_ext = os.path.splitext(file.filename)[1].lower()
            allowed = any(file_ext in exts for exts in self.allowed_extensions.values())
            if not allowed:
                return False, "Неподдерживаемый тип файла", None
            
            # Генерация уникального имени файла
            file_id = f"file_{uuid.uuid4().hex[:16]}"
            file_hash = hashlib.sha256()
            
            # Определение MIME типа
            mime_type, _ = mimetypes.guess_type(file.filename)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Сохранение файла
            file_path = os.path.join(self.upload_dir, f"{file_id}_{file.filename}")
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
                file_hash.update(content)
            
            file_hash_hex = file_hash.hexdigest()
            
            # Загрузка в IPFS (если доступен)
            ipfs_hash = None
            if self.db_manager.ipfs_client:
                try:
                    result = self.db_manager.ipfs_client.add(file_path)
                    ipfs_hash = result['Hash']
                    logger.info("Файл загружен в IPFS", file_id=file_id, ipfs_hash=ipfs_hash)
                except Exception as e:
                    logger.warning("Ошибка загрузки в IPFS", error=str(e))
            
            # Сохранение метаданных в базу данных
            conn = sqlite3.connect(self.db_manager.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO files 
                (file_id, user_id, filename, file_size, mime_type, file_hash, storage_path, ipfs_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_id, user_id, file.filename, file.size, mime_type,
                file_hash_hex, file_path, ipfs_hash, time.time()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info("Файл загружен", file_id=file_id, filename=file.filename, user_id=user_id)
            return True, "Файл успешно загружен", file_id
            
        except Exception as e:
            logger.error("Ошибка загрузки файла", error=str(e))
            return False, str(e), None
    
    async def get_file(self, file_id: str, user_id: str) -> Optional[Tuple[str, str, str]]:
        """Получение файла"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, mime_type, storage_path 
            FROM files WHERE file_id = ? AND user_id = ?
        ''', (file_id, user_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if row and os.path.exists(row[2]):
            return row[0], row[1], row[2]  # filename, mime_type, file_path
        
        return None
    
    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """Удаление файла"""
        try:
            conn = sqlite3.connect(self.db_manager.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT storage_path FROM files WHERE file_id = ? AND user_id = ?
            ''', (file_id, user_id))
            
            row = cursor.fetchone()
            if not row:
                conn.close()
                return False
            
            file_path = row[0]
            
            # Удаление из базы данных
            cursor.execute('DELETE FROM files WHERE file_id = ? AND user_id = ?', (file_id, user_id))
            conn.commit()
            conn.close()
            
            # Удаление физического файла
            if os.path.exists(file_path):
                os.remove(file_path)
            
            logger.info("Файл удален", file_id=file_id, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Ошибка удаления файла", error=str(e))
            return False
    
    def get_file_info(self, file_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Получение информации о файле"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, file_size, mime_type, file_hash, ipfs_hash, created_at
            FROM files WHERE file_id = ? AND user_id = ?
        ''', (file_id, user_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'file_id': file_id,
                'filename': row[0],
                'file_size': row[1],
                'mime_type': row[2],
                'file_hash': row[3],
                'ipfs_hash': row[4],
                'created_at': row[5]
            }
        
        return None

class NotificationManager:
    """Менеджер уведомлений"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.websocket_connections: Dict[str, WebSocket] = {}
        
        # Настройки email
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
    
    async def create_notification(self, user_id: str, title: str, message: str, 
                                notification_type: str = 'info') -> str:
        """Создание уведомления"""
        notification_id = f"notif_{uuid.uuid4().hex[:16]}"
        
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO notifications 
            (notification_id, user_id, title, message, type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (notification_id, user_id, title, message, notification_type, time.time()))
        
        conn.commit()
        conn.close()
        
        # Отправка через WebSocket
        await self._send_websocket_notification(user_id, {
            'notification_id': notification_id,
            'title': title,
            'message': message,
            'type': notification_type,
            'timestamp': time.time()
        })
        
        logger.info("Уведомление создано", notification_id=notification_id, user_id=user_id)
        return notification_id
    
    async def _send_websocket_notification(self, user_id: str, notification_data: Dict):
        """Отправка уведомления через WebSocket"""
        if user_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[user_id]
                await websocket.send_text(json.dumps({
                    'type': 'notification',
                    'data': notification_data
                }))
            except Exception as e:
                logger.warning("Ошибка отправки WebSocket уведомления", error=str(e))
                # Удаление неактивного соединения
                if user_id in self.websocket_connections:
                    del self.websocket_connections[user_id]
    
    async def send_email_notification(self, user_email: str, subject: str, body: str) -> bool:
        """Отправка email уведомления"""
        if not self.smtp_username or not self.smtp_password:
            logger.warning("SMTP не настроен")
            return False
        
        try:
            message = MIMEMultipart()
            message["From"] = self.smtp_username
            message["To"] = user_email
            message["Subject"] = subject
            
            message.attach(MIMEText(body, "html"))
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.smtp_username, user_email, message.as_string())
            
            logger.info("Email отправлен", email=user_email)
            return True
            
        except Exception as e:
            logger.error("Ошибка отправки email", error=str(e))
            return False
    
    def register_websocket(self, user_id: str, websocket: WebSocket):
        """Регистрация WebSocket соединения"""
        self.websocket_connections[user_id] = websocket
        ACTIVE_CONNECTIONS.inc()
        logger.info("WebSocket зарегистрирован", user_id=user_id)
    
    def unregister_websocket(self, user_id: str):
        """Отмена регистрации WebSocket соединения"""
        if user_id in self.websocket_connections:
            del self.websocket_connections[user_id]
            ACTIVE_CONNECTIONS.dec()
            logger.info("WebSocket отменен", user_id=user_id)
    
    def get_notifications(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Получение уведомлений пользователя"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT notification_id, title, message, type, is_read, created_at
            FROM notifications 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ? OFFSET ?
        ''', (user_id, limit, offset))
        
        notifications = []
        for row in cursor.fetchall():
            notifications.append({
                'notification_id': row[0],
                'title': row[1],
                'message': row[2],
                'type': row[3],
                'is_read': bool(row[4]),
                'created_at': row[5]
            })
        
        conn.close()
        return notifications
    
    def mark_as_read(self, notification_id: str, user_id: str) -> bool:
        """Отметка уведомления как прочитанного"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE notifications SET is_read = 1 
            WHERE notification_id = ? AND user_id = ?
        ''', (notification_id, user_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success

class TaskManager:
    """Менеджер задач"""
    
    def __init__(self, db_manager: DatabaseManager, notification_manager: NotificationManager):
        self.db_manager = db_manager
        self.notification_manager = notification_manager
        self.orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8001')
    
    async def submit_task(self, user_id: str, task_data: TaskSubmission) -> Tuple[bool, str]:
        """Подача задачи на выполнение"""
        try:
            task_id = f"task_{uuid.uuid4().hex[:16]}"
            
            # Сохранение в базу данных
            conn = sqlite3.connect(self.db_manager.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO tasks 
                (task_id, user_id, task_type, input_data, priority, max_reward, 
                 deadline, requirements, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id, user_id, task_data.task_type, json.dumps(task_data.input_data),
                task_data.priority, task_data.max_reward,
                task_data.deadline.timestamp() if task_data.deadline else None,
                json.dumps(task_data.requirements) if task_data.requirements else None,
                json.dumps(task_data.metadata) if task_data.metadata else None,
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
            # Отправка в оркестратор
            orchestrator_data = {
                'task_id': task_id,
                'user_id': user_id,
                'task_type': task_data.task_type,
                'input_data': task_data.input_data,
                'priority': task_data.priority,
                'max_reward': task_data.max_reward,
                'requirements': task_data.requirements or {}
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.orchestrator_url}/api/v1/tasks/submit",
                    json=orchestrator_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        # Уведомление пользователя
                        await self.notification_manager.create_notification(
                            user_id,
                            "Задача принята",
                            f"Ваша задача {task_id} принята к выполнению",
                            "success"
                        )
                        
                        PENDING_TASKS.inc()
                        logger.info("Задача отправлена в оркестратор", task_id=task_id, user_id=user_id)
                        return True, task_id
                    else:
                        error_text = await response.text()
                        logger.error("Ошибка отправки в оркестратор", 
                                   status=response.status, error=error_text)
                        return False, f"Ошибка оркестратора: {error_text}"
            
        except Exception as e:
            logger.error("Ошибка подачи задачи", error=str(e))
            return False, str(e)
    
    async def update_task_status(self, task_id: str, status: str, result_data: Dict = None):
        """Обновление статуса задачи"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        update_fields = ['status = ?']
        update_values = [status]
        
        if status == 'running' and not result_data:
            update_fields.append('started_at = ?')
            update_values.append(time.time())
        elif status in ['completed', 'failed']:
            update_fields.append('completed_at = ?')
            update_values.append(time.time())
            if result_data:
                update_fields.append('result_data = ?')
                update_values.append(json.dumps(result_data))
            PENDING_TASKS.dec()
        
        update_values.append(task_id)
        
        cursor.execute(f'''
            UPDATE tasks SET {', '.join(update_fields)}
            WHERE task_id = ?
        ''', update_values)
        
        # Получение user_id для уведомления
        cursor.execute('SELECT user_id FROM tasks WHERE task_id = ?', (task_id,))
        row = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        if row:
            user_id = row[0]
            
            # Уведомление пользователя об изменении статуса
            status_messages = {
                'running': 'Ваша задача начала выполняться',
                'completed': 'Ваша задача успешно выполнена',
                'failed': 'Выполнение вашей задачи завершилось с ошибкой'
            }
            
            if status in status_messages:
                await self.notification_manager.create_notification(
                    user_id,
                    f"Статус задачи: {status}",
                    status_messages[status],
                    'success' if status == 'completed' else 'warning' if status == 'failed' else 'info'
                )
    
    def get_task(self, task_id: str, user_id: str) -> Optional[Dict]:
        """Получение задачи"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT task_id, task_type, input_data, priority, max_reward, deadline,
                   requirements, metadata, status, created_at, started_at, 
                   completed_at, result_data, total_cost
            FROM tasks WHERE task_id = ? AND user_id = ?
        ''', (task_id, user_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'task_id': row[0],
                'task_type': row[1],
                'input_data': json.loads(row[2]) if row[2] else None,
                'priority': row[3],
                'max_reward': row[4],
                'deadline': row[5],
                'requirements': json.loads(row[6]) if row[6] else None,
                'metadata': json.loads(row[7]) if row[7] else None,
                'status': row[8],
                'created_at': row[9],
                'started_at': row[10],
                'completed_at': row[11],
                'result_data': json.loads(row[12]) if row[12] else None,
                'total_cost': row[13]
            }
        
        return None
    
    def get_user_tasks(self, user_id: str, status: str = None, limit: int = 50, offset: int = 0) -> List[Dict]:
        """Получение задач пользователя"""
        conn = sqlite3.connect(self.db_manager.sqlite_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT task_id, task_type, priority, max_reward, status, created_at, completed_at
            FROM tasks WHERE user_id = ?
        '''
        params = [user_id]
        
        if status:
            query += ' AND status = ?'
            params.append(status)
        
        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        tasks = []
        for row in cursor.fetchall():
            tasks.append({
                'task_id': row[0],
                'task_type': row[1],
                'priority': row[2],
                'max_reward': row[3],
                'status': row[4],
                'created_at': row[5],
                'completed_at': row[6]
            })
        
        conn.close()
        return tasks

# Инициализация компонентов
db_manager = DatabaseManager()
auth_manager = AuthenticationManager(db_manager)
file_manager = FileManager(db_manager)
notification_manager = NotificationManager(db_manager)
task_manager = TaskManager(db_manager, notification_manager)

# Создание FastAPI приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация при запуске
    await db_manager.initialize_async_connections()
    logger.info("API Server запущен")
    yield
    # Очистка при завершении
    if db_manager.redis_client:
        await db_manager.redis_client.close()
    if db_manager.postgres_pool:
        await db_manager.postgres_pool.close()
    if db_manager.mongo_client:
        db_manager.mongo_client.close()
    logger.info("API Server остановлен")

app = FastAPI(
    title="Daur-AI API",
    description="API для децентрализованной сети искусственного интеллекта",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GzipMiddleware, minimum_size=1000)

# Схема безопасности
security = HTTPBearer()

# Зависимость для аутентификации
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user = await auth_manager.get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Неверный токен аутентификации")
    return user

# Middleware для метрик
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

# API Endpoints

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "service": "Daur-AI API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "database": "healthy",
            "redis": "healthy" if db_manager.redis_client else "unavailable",
            "postgres": "healthy" if db_manager.postgres_pool else "unavailable",
            "mongodb": "healthy" if db_manager.mongo_client else "unavailable",
            "ipfs": "healthy" if db_manager.ipfs_client else "unavailable"
        }
    }
    
    return health_status

@app.get("/metrics")
async def metrics():
    """Метрики Prometheus"""
    return Response(generate_latest(), media_type="text/plain")

# Аутентификация
@app.post("/api/v1/auth/register")
async def register(user_data: UserRegistration):
    """Регистрация пользователя"""
    success, result = await auth_manager.register_user(user_data)
    
    if success:
        return {"success": True, "user_id": result, "message": "Пользователь успешно зарегистрирован"}
    else:
        raise HTTPException(status_code=400, detail=result)

@app.post("/api/v1/auth/login")
async def login(login_data: UserLogin):
    """Вход пользователя"""
    success, token, error = await auth_manager.authenticate_user(login_data)
    
    if success:
        return {"success": True, "token": token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail=error)

@app.get("/api/v1/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Получение информации о текущем пользователе"""
    return current_user

@app.post("/api/v1/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Выход пользователя"""
    # В реальной системе здесь была бы инвалидация токена
    return {"success": True, "message": "Успешный выход"}

# Управление задачами
@app.post("/api/v1/tasks/submit")
async def submit_task(task_data: TaskSubmission, current_user: dict = Depends(get_current_user)):
    """Подача задачи на выполнение"""
    success, result = await task_manager.submit_task(current_user['user_id'], task_data)
    
    if success:
        return {"success": True, "task_id": result, "message": "Задача принята к выполнению"}
    else:
        raise HTTPException(status_code=400, detail=result)

@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str, current_user: dict = Depends(get_current_user)):
    """Получение информации о задаче"""
    task = task_manager.get_task(task_id, current_user['user_id'])
    
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    return task

@app.get("/api/v1/tasks")
async def get_tasks(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Получение списка задач пользователя"""
    tasks = task_manager.get_user_tasks(current_user['user_id'], status, limit, offset)
    return {"tasks": tasks, "total": len(tasks)}

@app.post("/api/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str, current_user: dict = Depends(get_current_user)):
    """Отмена задачи"""
    # Проверка существования задачи
    task = task_manager.get_task(task_id, current_user['user_id'])
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    
    if task['status'] in ['completed', 'failed', 'cancelled']:
        raise HTTPException(status_code=400, detail="Задача не может быть отменена")
    
    # Обновление статуса
    await task_manager.update_task_status(task_id, 'cancelled')
    
    return {"success": True, "message": "Задача отменена"}

# Управление файлами
@app.post("/api/v1/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Загрузка файла"""
    success, message, file_id = await file_manager.upload_file(file, current_user['user_id'])
    
    if success:
        return {"success": True, "file_id": file_id, "message": message}
    else:
        raise HTTPException(status_code=400, detail=message)

@app.get("/api/v1/files/{file_id}")
async def get_file(file_id: str, current_user: dict = Depends(get_current_user)):
    """Получение файла"""
    file_data = await file_manager.get_file(file_id, current_user['user_id'])
    
    if not file_data:
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    filename, mime_type, file_path = file_data
    return FileResponse(file_path, filename=filename, media_type=mime_type)

@app.get("/api/v1/files/{file_id}/info")
async def get_file_info(file_id: str, current_user: dict = Depends(get_current_user)):
    """Получение информации о файле"""
    file_info = file_manager.get_file_info(file_id, current_user['user_id'])
    
    if not file_info:
        raise HTTPException(status_code=404, detail="Файл не найден")
    
    return file_info

@app.delete("/api/v1/files/{file_id}")
async def delete_file(file_id: str, current_user: dict = Depends(get_current_user)):
    """Удаление файла"""
    success = await file_manager.delete_file(file_id, current_user['user_id'])
    
    if success:
        return {"success": True, "message": "Файл удален"}
    else:
        raise HTTPException(status_code=404, detail="Файл не найден")

# Уведомления
@app.get("/api/v1/notifications")
async def get_notifications(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Получение уведомлений"""
    notifications = notification_manager.get_notifications(current_user['user_id'], limit, offset)
    return {"notifications": notifications}

@app.post("/api/v1/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Отметка уведомления как прочитанного"""
    success = notification_manager.mark_as_read(notification_id, current_user['user_id'])
    
    if success:
        return {"success": True, "message": "Уведомление отмечено как прочитанное"}
    else:
        raise HTTPException(status_code=404, detail="Уведомление не найдено")

# Регистрация узлов
@app.post("/api/v1/nodes/register")
async def register_node(
    node_data: NodeRegistration,
    current_user: dict = Depends(get_current_user)
):
    """Регистрация узла в сети"""
    try:
        # Отправка данных в оркестратор
        orchestrator_data = {
            'user_id': current_user['user_id'],
            'device_id': node_data.device_id,
            'address': f"{current_user['username']}_{node_data.device_id}",
            'cpu_cores': node_data.cpu_cores,
            'memory_mb': node_data.memory_mb,
            'gpu_available': node_data.gpu_available,
            'gpu_memory_mb': node_data.gpu_memory_mb,
            'storage_mb': node_data.storage_mb,
            'network_speed_mbps': node_data.network_speed_mbps,
            'battery_level': node_data.battery_level,
            'device_type': node_data.device_type,
            'os_type': node_data.os_type,
            'stake_amount': node_data.stake_amount
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{task_manager.orchestrator_url}/api/v1/nodes/register",
                json=orchestrator_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    node_id = result.get('node_id')
                    
                    # Сохранение в локальной базе данных
                    conn = sqlite3.connect(db_manager.sqlite_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO nodes 
                        (node_id, user_id, device_id, device_type, capabilities, last_seen, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        node_id, current_user['user_id'], node_data.device_id,
                        node_data.device_type, json.dumps(node_data.dict()),
                        time.time(), time.time()
                    ))
                    
                    conn.commit()
                    conn.close()
                    
                    ACTIVE_NODES.inc()
                    
                    return {"success": True, "node_id": node_id, "message": "Узел зарегистрирован"}
                else:
                    error_text = await response.text()
                    raise HTTPException(status_code=400, detail=f"Ошибка регистрации узла: {error_text}")
    
    except Exception as e:
        logger.error("Ошибка регистрации узла", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket для уведомлений
@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket, token: str):
    """WebSocket для получения уведомлений в реальном времени"""
    # Аутентификация через токен
    user = await auth_manager.get_current_user(token)
    if not user:
        await websocket.close(code=4001, reason="Неверный токен")
        return
    
    await websocket.accept()
    user_id = user['user_id']
    
    # Регистрация соединения
    notification_manager.register_websocket(user_id, websocket)
    
    try:
        # Отправка приветственного сообщения
        await websocket.send_text(json.dumps({
            'type': 'connected',
            'message': 'Подключение к уведомлениям установлено'
        }))
        
        # Ожидание сообщений от клиента
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обработка ping/pong для поддержания соединения
            if message.get('type') == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Ошибка WebSocket", error=str(e))
    finally:
        # Отмена регистрации соединения
        notification_manager.unregister_websocket(user_id)

# WebSocket для узлов
@app.websocket("/ws/nodes/{node_id}")
async def websocket_node(websocket: WebSocket, node_id: str, token: str):
    """WebSocket для взаимодействия с узлами"""
    # Аутентификация
    user = await auth_manager.get_current_user(token)
    if not user:
        await websocket.close(code=4001, reason="Неверный токен")
        return
    
    await websocket.accept()
    
    try:
        # Уведомление оркестратора о подключении узла
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{task_manager.orchestrator_url}/api/v1/nodes/{node_id}/connect",
                json={'user_id': user['user_id']},
                timeout=aiohttp.ClientTimeout(total=10)
            )
        
        # Основной цикл обработки сообщений
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обработка различных типов сообщений
            if message.get('type') == 'heartbeat':
                await websocket.send_text(json.dumps({
                    'type': 'heartbeat_ack',
                    'timestamp': time.time()
                }))
            
            elif message.get('type') == 'bid_submission':
                # Пересылка ставки в оркестратор
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{task_manager.orchestrator_url}/api/v1/auctions/bid",
                        json=message['data'],
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
            
            elif message.get('type') == 'task_result':
                # Обработка результата выполнения задачи
                result_data = message['data']
                assignment_id = result_data.get('assignment_id')
                
                if assignment_id:
                    # Пересылка результата в оркестратор
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            f"{task_manager.orchestrator_url}/api/v1/assignments/{assignment_id}/result",
                            json=result_data,
                            timeout=aiohttp.ClientTimeout(total=30)
                        )
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Ошибка WebSocket узла", node_id=node_id, error=str(e))
    finally:
        # Уведомление оркестратора об отключении узла
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{task_manager.orchestrator_url}/api/v1/nodes/{node_id}/disconnect",
                    timeout=aiohttp.ClientTimeout(total=5)
                )
        except:
            pass

# Статистика и мониторинг
@app.get("/api/v1/stats/network")
async def get_network_stats():
    """Получение статистики сети"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{task_manager.orchestrator_url}/api/v1/stats/network",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(status_code=503, detail="Статистика недоступна")
    except Exception as e:
        logger.error("Ошибка получения статистики", error=str(e))
        raise HTTPException(status_code=503, detail="Сервис статистики недоступен")

@app.get("/api/v1/stats/user")
async def get_user_stats(current_user: dict = Depends(get_current_user)):
    """Получение статистики пользователя"""
    conn = sqlite3.connect(db_manager.sqlite_path)
    cursor = conn.cursor()
    
    # Статистика задач
    cursor.execute('''
        SELECT status, COUNT(*) FROM tasks 
        WHERE user_id = ? GROUP BY status
    ''', (current_user['user_id'],))
    
    task_stats = {}
    for row in cursor.fetchall():
        task_stats[row[0]] = row[1]
    
    # Статистика узлов
    cursor.execute('''
        SELECT COUNT(*), SUM(total_earnings), SUM(tasks_completed)
        FROM nodes WHERE user_id = ?
    ''', (current_user['user_id'],))
    
    node_row = cursor.fetchone()
    node_stats = {
        'total_nodes': node_row[0] or 0,
        'total_earnings': node_row[1] or 0.0,
        'tasks_completed': node_row[2] or 0
    }
    
    conn.close()
    
    return {
        'tasks': task_stats,
        'nodes': node_stats,
        'user_id': current_user['user_id']
    }

# Обработчики ошибок
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error("Необработанная ошибка", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Внутренняя ошибка сервера", "status_code": 500}
    )

# Запуск сервера
if __name__ == "__main__":
    uvicorn.run(
        "main_fixed:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )
