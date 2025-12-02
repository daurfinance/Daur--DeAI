"""
Daur-AI Python SDK - Полнофункциональная реализация
Библиотека для взаимодействия с сетью Daur-AI
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
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import websockets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import jwt
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import asyncio_mqtt
import ssl
from pathlib import Path
import mimetypes
from PIL import Image
import numpy as np
import pandas as pd
import io
import zipfile
import tarfile
import pickle
import msgpack
import cbor2
from urllib.parse import urljoin, urlparse
import certifi
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger(__name__)

class TaskStatus(Enum):
    """Статусы задач"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class NodeStatus(Enum):
    """Статусы узлов"""
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"
    MAINTENANCE = "maintenance"

class TaskType(Enum):
    """Типы задач"""
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    DATA_ANALYSIS = "data_analysis"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"

@dataclass
class TaskConfig:
    """Конфигурация задачи"""
    task_type: str
    input_data: Dict[str, Any]
    priority: int = 1
    max_reward: float = 10.0
    deadline: Optional[datetime] = None
    requirements: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.deadline:
            result['deadline'] = self.deadline.isoformat()
        return result

@dataclass
class NodeConfig:
    """Конфигурация узла"""
    device_id: str
    device_type: str
    cpu_cores: int
    memory_mb: int
    storage_mb: int
    gpu_available: bool = False
    gpu_memory_mb: int = 0
    network_speed_mbps: float = 10.0
    battery_level: int = 100
    os_type: str = "unknown"
    stake_amount: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TaskResult:
    """Результат выполнения задачи"""
    task_id: str
    status: TaskStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    cost: Optional[float] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class DaurAIException(Exception):
    """Базовое исключение для Daur-AI SDK"""
    pass

class AuthenticationError(DaurAIException):
    """Ошибка аутентификации"""
    pass

class NetworkError(DaurAIException):
    """Ошибка сети"""
    pass

class TaskError(DaurAIException):
    """Ошибка выполнения задачи"""
    pass

class ValidationError(DaurAIException):
    """Ошибка валидации данных"""
    pass

class HTTPClient:
    """HTTP клиент с retry логикой и обработкой ошибок"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Настройка retry стратегии
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS", "POST", "PUT", "DELETE"],
            backoff_factor=1
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Настройка SSL
        self.session.verify = certifi.where()
        
        # Заголовки по умолчанию
        self.session.headers.update({
            'User-Agent': 'Daur-AI-SDK/1.0.0',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def set_auth_token(self, token: str):
        """Установка токена аутентификации"""
        self.session.headers['Authorization'] = f'Bearer {token}'
    
    def remove_auth_token(self):
        """Удаление токена аутентификации"""
        if 'Authorization' in self.session.headers:
            del self.session.headers['Authorization']
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Выполнение HTTP запроса с retry логикой"""
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            
            # Логирование запроса
            logger.info(
                "HTTP запрос",
                method=method,
                url=url,
                status_code=response.status_code,
                response_time=response.elapsed.total_seconds()
            )
            
            # Проверка статуса ответа
            if response.status_code == 401:
                raise AuthenticationError("Неверный токен аутентификации")
            elif response.status_code == 403:
                raise AuthenticationError("Недостаточно прав доступа")
            elif response.status_code >= 500:
                raise NetworkError(f"Ошибка сервера: {response.status_code}")
            elif response.status_code >= 400:
                try:
                    error_data = response.json()
                    error_message = error_data.get('error', f'HTTP {response.status_code}')
                except:
                    error_message = f'HTTP {response.status_code}: {response.text}'
                raise DaurAIException(error_message)
            
            return response
            
        except requests.exceptions.Timeout:
            raise NetworkError("Превышено время ожидания запроса")
        except requests.exceptions.ConnectionError:
            raise NetworkError("Ошибка подключения к серверу")
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Ошибка HTTP запроса: {str(e)}")
    
    def get(self, endpoint: str, **kwargs) -> requests.Response:
        """GET запрос"""
        return self.request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> requests.Response:
        """POST запрос"""
        return self.request('POST', endpoint, **kwargs)
    
    def put(self, endpoint: str, **kwargs) -> requests.Response:
        """PUT запрос"""
        return self.request('PUT', endpoint, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """DELETE запрос"""
        return self.request('DELETE', endpoint, **kwargs)
    
    def close(self):
        """Закрытие сессии"""
        self.session.close()

class AsyncHTTPClient:
    """Асинхронный HTTP клиент"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ssl=ssl.create_default_context(cafile=certifi.where())
        )
        self.session = None
        self.auth_token = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'Daur-AI-SDK/1.0.0',
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        )
        
        if self.auth_token:
            self.session.headers['Authorization'] = f'Bearer {self.auth_token}'
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def set_auth_token(self, token: str):
        """Установка токена аутентификации"""
        self.auth_token = token
        if self.session:
            self.session.headers['Authorization'] = f'Bearer {token}'
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def request(self, method: str, endpoint: str, **kwargs) -> aiohttp.ClientResponse:
        """Выполнение асинхронного HTTP запроса"""
        if not self.session:
            raise RuntimeError("Клиент не инициализирован. Используйте async with.")
        
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                # Логирование запроса
                logger.info(
                    "Async HTTP запрос",
                    method=method,
                    url=url,
                    status_code=response.status
                )
                
                # Проверка статуса ответа
                if response.status == 401:
                    raise AuthenticationError("Неверный токен аутентификации")
                elif response.status == 403:
                    raise AuthenticationError("Недостаточно прав доступа")
                elif response.status >= 500:
                    raise NetworkError(f"Ошибка сервера: {response.status}")
                elif response.status >= 400:
                    try:
                        error_data = await response.json()
                        error_message = error_data.get('error', f'HTTP {response.status}')
                    except:
                        error_message = f'HTTP {response.status}: {await response.text()}'
                    raise DaurAIException(error_message)
                
                return response
                
        except asyncio.TimeoutError:
            raise NetworkError("Превышено время ожидания запроса")
        except aiohttp.ClientConnectionError:
            raise NetworkError("Ошибка подключения к серверу")
        except aiohttp.ClientError as e:
            raise NetworkError(f"Ошибка HTTP запроса: {str(e)}")

class WebSocketManager:
    """Менеджер WebSocket соединений"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        self.connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.auth_token = None
        self.running = False
    
    def set_auth_token(self, token: str):
        """Установка токена аутентификации"""
        self.auth_token = token
    
    def on(self, event_type: str, handler: Callable):
        """Регистрация обработчика событий"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: Callable):
        """Отмена регистрации обработчика событий"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    async def connect(self, endpoint: str, connection_id: str = None) -> str:
        """Подключение к WebSocket"""
        if not connection_id:
            connection_id = f"conn_{uuid.uuid4().hex[:8]}"
        
        url = urljoin(self.base_url, endpoint.lstrip('/'))
        
        # Добавление токена аутентификации в URL
        if self.auth_token:
            separator = '&' if '?' in url else '?'
            url += f"{separator}token={self.auth_token}"
        
        try:
            websocket = await websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connections[connection_id] = websocket
            
            # Запуск обработчика сообщений
            asyncio.create_task(self._message_handler(connection_id, websocket))
            
            logger.info("WebSocket подключен", connection_id=connection_id, url=url)
            return connection_id
            
        except Exception as e:
            logger.error("Ошибка подключения WebSocket", error=str(e))
            raise NetworkError(f"Не удалось подключиться к WebSocket: {str(e)}")
    
    async def disconnect(self, connection_id: str):
        """Отключение от WebSocket"""
        if connection_id in self.connections:
            websocket = self.connections[connection_id]
            await websocket.close()
            del self.connections[connection_id]
            logger.info("WebSocket отключен", connection_id=connection_id)
    
    async def send(self, connection_id: str, message: Dict[str, Any]):
        """Отправка сообщения через WebSocket"""
        if connection_id not in self.connections:
            raise DaurAIException(f"WebSocket соединение {connection_id} не найдено")
        
        websocket = self.connections[connection_id]
        
        try:
            await websocket.send(json.dumps(message))
            logger.debug("WebSocket сообщение отправлено", connection_id=connection_id)
        except Exception as e:
            logger.error("Ошибка отправки WebSocket сообщения", error=str(e))
            raise NetworkError(f"Не удалось отправить сообщение: {str(e)}")
    
    async def _message_handler(self, connection_id: str, websocket: websockets.WebSocketClientProtocol):
        """Обработчик входящих сообщений"""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get('type', 'message')
                    
                    # Вызов обработчиков событий
                    if event_type in self.event_handlers:
                        for handler in self.event_handlers[event_type]:
                            try:
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(data)
                                else:
                                    handler(data)
                            except Exception as e:
                                logger.error("Ошибка в обработчике события", 
                                           event_type=event_type, error=str(e))
                    
                    logger.debug("WebSocket сообщение получено", 
                               connection_id=connection_id, event_type=event_type)
                    
                except json.JSONDecodeError:
                    logger.warning("Получено некорректное JSON сообщение", message=message)
                except Exception as e:
                    logger.error("Ошибка обработки сообщения", error=str(e))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket соединение закрыто", connection_id=connection_id)
        except Exception as e:
            logger.error("Ошибка в обработчике сообщений", error=str(e))
        finally:
            if connection_id in self.connections:
                del self.connections[connection_id]
    
    async def disconnect_all(self):
        """Отключение всех WebSocket соединений"""
        for connection_id in list(self.connections.keys()):
            await self.disconnect(connection_id)

class FileManager:
    """Менеджер файлов"""
    
    def __init__(self, http_client: HTTPClient):
        self.http_client = http_client
        self.supported_formats = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
            'document': ['.pdf', '.doc', '.docx', '.txt', '.md'],
            'data': ['.json', '.csv', '.xml', '.yaml', '.pkl', '.npy'],
            'archive': ['.zip', '.tar', '.gz', '.bz2']
        }
    
    def upload_file(self, file_path: str, chunk_size: int = 8192) -> str:
        """Загрузка файла на сервер"""
        if not os.path.exists(file_path):
            raise ValidationError(f"Файл не найден: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100 MB
            raise ValidationError("Файл слишком большой (максимум 100 МБ)")
        
        # Проверка типа файла
        file_ext = Path(file_path).suffix.lower()
        if not any(file_ext in formats for formats in self.supported_formats.values()):
            raise ValidationError(f"Неподдерживаемый тип файла: {file_ext}")
        
        # Определение MIME типа
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = 'application/octet-stream'
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, mime_type)}
                
                # Удаление Content-Type заголовка для multipart/form-data
                headers = dict(self.http_client.session.headers)
                if 'Content-Type' in headers:
                    del headers['Content-Type']
                
                response = self.http_client.post(
                    '/api/v1/files/upload',
                    files=files,
                    headers=headers
                )
                
                result = response.json()
                if result.get('success'):
                    file_id = result.get('file_id')
                    logger.info("Файл загружен", file_path=file_path, file_id=file_id)
                    return file_id
                else:
                    raise DaurAIException(f"Ошибка загрузки файла: {result.get('message', 'Неизвестная ошибка')}")
        
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            logger.error("Ошибка загрузки файла", error=str(e))
            raise DaurAIException(f"Не удалось загрузить файл: {str(e)}")
    
    def download_file(self, file_id: str, save_path: str = None) -> str:
        """Скачивание файла с сервера"""
        try:
            response = self.http_client.get(f'/api/v1/files/{file_id}')
            
            # Определение имени файла
            if save_path:
                filename = save_path
            else:
                content_disposition = response.headers.get('content-disposition', '')
                if 'filename=' in content_disposition:
                    filename = content_disposition.split('filename=')[1].strip('"')
                else:
                    filename = f"file_{file_id}"
            
            # Сохранение файла
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("Файл скачан", file_id=file_id, filename=filename)
            return filename
        
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            logger.error("Ошибка скачивания файла", error=str(e))
            raise DaurAIException(f"Не удалось скачать файл: {str(e)}")
    
    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        """Получение информации о файле"""
        try:
            response = self.http_client.get(f'/api/v1/files/{file_id}/info')
            return response.json()
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            raise DaurAIException(f"Не удалось получить информацию о файле: {str(e)}")
    
    def delete_file(self, file_id: str) -> bool:
        """Удаление файла"""
        try:
            response = self.http_client.delete(f'/api/v1/files/{file_id}')
            result = response.json()
            return result.get('success', False)
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            raise DaurAIException(f"Не удалось удалить файл: {str(e)}")
    
    def process_image(self, image_path: str, operations: List[Dict[str, Any]]) -> np.ndarray:
        """Обработка изображения"""
        try:
            with Image.open(image_path) as img:
                img_array = np.array(img)
                
                for operation in operations:
                    op_type = operation.get('type')
                    
                    if op_type == 'resize':
                        size = operation.get('size', (256, 256))
                        img = img.resize(size, Image.Resampling.LANCZOS)
                        img_array = np.array(img)
                    
                    elif op_type == 'rotate':
                        angle = operation.get('angle', 0)
                        img_array = np.rot90(img_array, k=angle//90)
                    
                    elif op_type == 'normalize':
                        img_array = img_array.astype(np.float32) / 255.0
                    
                    elif op_type == 'grayscale':
                        if len(img_array.shape) == 3:
                            img_array = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
                
                return img_array
        
        except Exception as e:
            raise ValidationError(f"Ошибка обработки изображения: {str(e)}")
    
    def create_archive(self, files: List[str], archive_path: str, format: str = 'zip') -> str:
        """Создание архива файлов"""
        try:
            if format == 'zip':
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in files:
                        if os.path.exists(file_path):
                            zipf.write(file_path, os.path.basename(file_path))
            
            elif format in ['tar', 'tar.gz']:
                mode = 'w:gz' if format == 'tar.gz' else 'w'
                with tarfile.open(archive_path, mode) as tarf:
                    for file_path in files:
                        if os.path.exists(file_path):
                            tarf.add(file_path, arcname=os.path.basename(file_path))
            
            else:
                raise ValidationError(f"Неподдерживаемый формат архива: {format}")
            
            logger.info("Архив создан", archive_path=archive_path, files_count=len(files))
            return archive_path
        
        except Exception as e:
            raise DaurAIException(f"Не удалось создать архив: {str(e)}")

class DataProcessor:
    """Процессор данных"""
    
    def __init__(self):
        self.supported_formats = ['json', 'csv', 'xml', 'yaml', 'pickle', 'msgpack', 'cbor']
    
    def serialize_data(self, data: Any, format: str = 'json') -> bytes:
        """Сериализация данных"""
        try:
            if format == 'json':
                return json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
            
            elif format == 'pickle':
                return pickle.dumps(data)
            
            elif format == 'msgpack':
                return msgpack.packb(data, use_bin_type=True)
            
            elif format == 'cbor':
                return cbor2.dumps(data)
            
            else:
                raise ValidationError(f"Неподдерживаемый формат сериализации: {format}")
        
        except Exception as e:
            raise ValidationError(f"Ошибка сериализации данных: {str(e)}")
    
    def deserialize_data(self, data: bytes, format: str = 'json') -> Any:
        """Десериализация данных"""
        try:
            if format == 'json':
                return json.loads(data.decode('utf-8'))
            
            elif format == 'pickle':
                return pickle.loads(data)
            
            elif format == 'msgpack':
                return msgpack.unpackb(data, raw=False)
            
            elif format == 'cbor':
                return cbor2.loads(data)
            
            else:
                raise ValidationError(f"Неподдерживаемый формат десериализации: {format}")
        
        except Exception as e:
            raise ValidationError(f"Ошибка десериализации данных: {str(e)}")
    
    def process_csv(self, file_path: str, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Обработка CSV файла"""
        try:
            df = pd.read_csv(file_path)
            
            for operation in operations:
                op_type = operation.get('type')
                
                if op_type == 'filter':
                    condition = operation.get('condition')
                    if condition:
                        df = df.query(condition)
                
                elif op_type == 'sort':
                    columns = operation.get('columns', [])
                    ascending = operation.get('ascending', True)
                    if columns:
                        df = df.sort_values(by=columns, ascending=ascending)
                
                elif op_type == 'group':
                    group_by = operation.get('group_by', [])
                    agg_func = operation.get('agg_func', 'mean')
                    if group_by:
                        df = df.groupby(group_by).agg(agg_func).reset_index()
                
                elif op_type == 'select':
                    columns = operation.get('columns', [])
                    if columns:
                        df = df[columns]
            
            return df
        
        except Exception as e:
            raise ValidationError(f"Ошибка обработки CSV: {str(e)}")
    
    def validate_data_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Валидация схемы данных"""
        try:
            def validate_field(value, field_schema):
                field_type = field_schema.get('type')
                required = field_schema.get('required', False)
                
                if value is None:
                    return not required
                
                if field_type == 'string':
                    if not isinstance(value, str):
                        return False
                    min_length = field_schema.get('min_length')
                    max_length = field_schema.get('max_length')
                    if min_length and len(value) < min_length:
                        return False
                    if max_length and len(value) > max_length:
                        return False
                
                elif field_type == 'number':
                    if not isinstance(value, (int, float)):
                        return False
                    minimum = field_schema.get('minimum')
                    maximum = field_schema.get('maximum')
                    if minimum is not None and value < minimum:
                        return False
                    if maximum is not None and value > maximum:
                        return False
                
                elif field_type == 'boolean':
                    if not isinstance(value, bool):
                        return False
                
                elif field_type == 'array':
                    if not isinstance(value, list):
                        return False
                    item_schema = field_schema.get('items')
                    if item_schema:
                        for item in value:
                            if not validate_field(item, item_schema):
                                return False
                
                elif field_type == 'object':
                    if not isinstance(value, dict):
                        return False
                    properties = field_schema.get('properties', {})
                    for prop_name, prop_schema in properties.items():
                        if not validate_field(value.get(prop_name), prop_schema):
                            return False
                
                return True
            
            properties = schema.get('properties', {})
            for field_name, field_schema in properties.items():
                if not validate_field(data.get(field_name), field_schema):
                    return False
            
            return True
        
        except Exception as e:
            logger.error("Ошибка валидации схемы", error=str(e))
            return False

class DaurAIClient:
    """Основной клиент для взаимодействия с Daur-AI"""
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: str = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.auth_token = None
        self.user_info = None
        
        # Инициализация компонентов
        self.http_client = HTTPClient(self.api_url)
        self.websocket_manager = WebSocketManager(self.api_url)
        self.file_manager = FileManager(self.http_client)
        self.data_processor = DataProcessor()
        
        # Кэш для задач и узлов
        self.tasks_cache: Dict[str, TaskResult] = {}
        self.nodes_cache: Dict[str, Dict[str, Any]] = {}
        
        # Обработчики событий
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Настройка обработчиков WebSocket событий
        self._setup_websocket_handlers()
        
        logger.info("DaurAI клиент инициализирован", api_url=self.api_url)
    
    def _setup_websocket_handlers(self):
        """Настройка обработчиков WebSocket событий"""
        self.websocket_manager.on('notification', self._handle_notification)
        self.websocket_manager.on('task_update', self._handle_task_update)
        self.websocket_manager.on('node_update', self._handle_node_update)
    
    async def _handle_notification(self, data: Dict[str, Any]):
        """Обработчик уведомлений"""
        notification = data.get('data', {})
        logger.info("Получено уведомление", 
                   title=notification.get('title'),
                   type=notification.get('type'))
        
        # Вызов пользовательских обработчиков
        await self._emit_event('notification', notification)
    
    async def _handle_task_update(self, data: Dict[str, Any]):
        """Обработчик обновлений задач"""
        task_data = data.get('data', {})
        task_id = task_data.get('task_id')
        
        if task_id:
            # Обновление кэша
            if task_id in self.tasks_cache:
                self.tasks_cache[task_id].status = TaskStatus(task_data.get('status', 'pending'))
                if task_data.get('result_data'):
                    self.tasks_cache[task_id].result_data = task_data['result_data']
            
            logger.info("Обновление задачи", task_id=task_id, status=task_data.get('status'))
            
            # Вызов пользовательских обработчиков
            await self._emit_event('task_update', task_data)
    
    async def _handle_node_update(self, data: Dict[str, Any]):
        """Обработчик обновлений узлов"""
        node_data = data.get('data', {})
        node_id = node_data.get('node_id')
        
        if node_id:
            # Обновление кэша
            self.nodes_cache[node_id] = node_data
            
            logger.info("Обновление узла", node_id=node_id, status=node_data.get('status'))
            
            # Вызов пользовательских обработчиков
            await self._emit_event('node_update', node_data)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Вызов обработчиков событий"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error("Ошибка в обработчике события", 
                               event_type=event_type, error=str(e))
    
    def on(self, event_type: str, handler: Callable):
        """Регистрация обработчика событий"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def off(self, event_type: str, handler: Callable):
        """Отмена регистрации обработчика событий"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def login(self, username: str, password: str) -> bool:
        """Вход в систему"""
        try:
            response = self.http_client.post('/api/v1/auth/login', json={
                'username': username,
                'password': password
            })
            
            result = response.json()
            if result.get('success'):
                self.auth_token = result.get('token')
                self.http_client.set_auth_token(self.auth_token)
                self.websocket_manager.set_auth_token(self.auth_token)
                
                # Получение информации о пользователе
                self.user_info = self.get_user_info()
                
                logger.info("Успешный вход", username=username)
                return True
            else:
                logger.warning("Неудачный вход", username=username)
                return False
        
        except Exception as e:
            logger.error("Ошибка входа", error=str(e))
            return False
    
    def register(self, username: str, email: str, password: str, 
                full_name: str = None, organization: str = None) -> bool:
        """Регистрация пользователя"""
        try:
            user_data = {
                'username': username,
                'email': email,
                'password': password
            }
            
            if full_name:
                user_data['full_name'] = full_name
            if organization:
                user_data['organization'] = organization
            
            response = self.http_client.post('/api/v1/auth/register', json=user_data)
            result = response.json()
            
            if result.get('success'):
                logger.info("Успешная регистрация", username=username)
                return True
            else:
                logger.warning("Неудачная регистрация", username=username)
                return False
        
        except Exception as e:
            logger.error("Ошибка регистрации", error=str(e))
            return False
    
    def logout(self) -> bool:
        """Выход из системы"""
        try:
            if self.auth_token:
                self.http_client.post('/api/v1/auth/logout')
                self.http_client.remove_auth_token()
                self.websocket_manager.auth_token = None
                self.auth_token = None
                self.user_info = None
                
                logger.info("Успешный выход")
                return True
            return False
        
        except Exception as e:
            logger.error("Ошибка выхода", error=str(e))
            return False
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """Получение информации о пользователе"""
        try:
            response = self.http_client.get('/api/v1/auth/me')
            return response.json()
        except Exception as e:
            logger.error("Ошибка получения информации о пользователе", error=str(e))
            return None
    
    def submit_task(self, task_config: TaskConfig) -> str:
        """Подача задачи на выполнение"""
        try:
            response = self.http_client.post('/api/v1/tasks/submit', json=task_config.to_dict())
            result = response.json()
            
            if result.get('success'):
                task_id = result.get('task_id')
                
                # Добавление в кэш
                self.tasks_cache[task_id] = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.PENDING,
                    created_at=datetime.now()
                )
                
                logger.info("Задача отправлена", task_id=task_id, task_type=task_config.task_type)
                return task_id
            else:
                raise TaskError(f"Не удалось отправить задачу: {result.get('message', 'Неизвестная ошибка')}")
        
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            logger.error("Ошибка отправки задачи", error=str(e))
            raise TaskError(f"Не удалось отправить задачу: {str(e)}")
    
    def get_task(self, task_id: str, use_cache: bool = True) -> Optional[TaskResult]:
        """Получение информации о задаче"""
        # Проверка кэша
        if use_cache and task_id in self.tasks_cache:
            return self.tasks_cache[task_id]
        
        try:
            response = self.http_client.get(f'/api/v1/tasks/{task_id}')
            task_data = response.json()
            
            task_result = TaskResult(
                task_id=task_data['task_id'],
                status=TaskStatus(task_data['status']),
                result_data=task_data.get('result_data'),
                execution_time=task_data.get('execution_time'),
                cost=task_data.get('total_cost'),
                created_at=datetime.fromtimestamp(task_data['created_at']) if task_data.get('created_at') else None,
                completed_at=datetime.fromtimestamp(task_data['completed_at']) if task_data.get('completed_at') else None
            )
            
            # Обновление кэша
            self.tasks_cache[task_id] = task_result
            
            return task_result
        
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            logger.error("Ошибка получения задачи", task_id=task_id, error=str(e))
            return None
    
    def get_tasks(self, status: str = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Получение списка задач"""
        try:
            params = {'limit': limit, 'offset': offset}
            if status:
                params['status'] = status
            
            response = self.http_client.get('/api/v1/tasks', params=params)
            result = response.json()
            
            return result.get('tasks', [])
        
        except Exception as e:
            logger.error("Ошибка получения списка задач", error=str(e))
            return []
    
    def cancel_task(self, task_id: str) -> bool:
        """Отмена задачи"""
        try:
            response = self.http_client.post(f'/api/v1/tasks/{task_id}/cancel')
            result = response.json()
            
            if result.get('success'):
                # Обновление кэша
                if task_id in self.tasks_cache:
                    self.tasks_cache[task_id].status = TaskStatus.CANCELLED
                
                logger.info("Задача отменена", task_id=task_id)
                return True
            
            return False
        
        except Exception as e:
            logger.error("Ошибка отмены задачи", task_id=task_id, error=str(e))
            return False
    
    def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskResult:
        """Ожидание завершения задачи"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.get_task(task_id, use_cache=False)
            
            if not task:
                raise TaskError(f"Задача {task_id} не найдена")
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task
            
            time.sleep(poll_interval)
        
        raise TaskError(f"Превышено время ожидания задачи {task_id}")
    
    def register_node(self, node_config: NodeConfig) -> str:
        """Регистрация узла в сети"""
        try:
            response = self.http_client.post('/api/v1/nodes/register', json=node_config.to_dict())
            result = response.json()
            
            if result.get('success'):
                node_id = result.get('node_id')
                
                # Добавление в кэш
                self.nodes_cache[node_id] = {
                    'node_id': node_id,
                    'status': NodeStatus.ONLINE.value,
                    'config': node_config.to_dict()
                }
                
                logger.info("Узел зарегистрирован", node_id=node_id, device_id=node_config.device_id)
                return node_id
            else:
                raise DaurAIException(f"Не удалось зарегистрировать узел: {result.get('message', 'Неизвестная ошибка')}")
        
        except Exception as e:
            if isinstance(e, DaurAIException):
                raise
            logger.error("Ошибка регистрации узла", error=str(e))
            raise DaurAIException(f"Не удалось зарегистрировать узел: {str(e)}")
    
    async def connect_notifications(self) -> str:
        """Подключение к уведомлениям через WebSocket"""
        try:
            connection_id = await self.websocket_manager.connect('/ws/notifications')
            logger.info("Подключен к уведомлениям", connection_id=connection_id)
            return connection_id
        except Exception as e:
            logger.error("Ошибка подключения к уведомлениям", error=str(e))
            raise NetworkError(f"Не удалось подключиться к уведомлениям: {str(e)}")
    
    async def connect_node(self, node_id: str) -> str:
        """Подключение узла через WebSocket"""
        try:
            connection_id = await self.websocket_manager.connect(f'/ws/nodes/{node_id}')
            logger.info("Узел подключен", node_id=node_id, connection_id=connection_id)
            return connection_id
        except Exception as e:
            logger.error("Ошибка подключения узла", node_id=node_id, error=str(e))
            raise NetworkError(f"Не удалось подключить узел: {str(e)}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Получение статистики сети"""
        try:
            response = self.http_client.get('/api/v1/stats/network')
            return response.json()
        except Exception as e:
            logger.error("Ошибка получения статистики сети", error=str(e))
            return {}
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Получение статистики пользователя"""
        try:
            response = self.http_client.get('/api/v1/stats/user')
            return response.json()
        except Exception as e:
            logger.error("Ошибка получения статистики пользователя", error=str(e))
            return {}
    
    def get_notifications(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Получение уведомлений"""
        try:
            params = {'limit': limit, 'offset': offset}
            response = self.http_client.get('/api/v1/notifications', params=params)
            result = response.json()
            return result.get('notifications', [])
        except Exception as e:
            logger.error("Ошибка получения уведомлений", error=str(e))
            return []
    
    def mark_notification_read(self, notification_id: str) -> bool:
        """Отметка уведомления как прочитанного"""
        try:
            response = self.http_client.post(f'/api/v1/notifications/{notification_id}/read')
            result = response.json()
            return result.get('success', False)
        except Exception as e:
            logger.error("Ошибка отметки уведомления", notification_id=notification_id, error=str(e))
            return False
    
    async def disconnect_all(self):
        """Отключение всех соединений"""
        await self.websocket_manager.disconnect_all()
        self.http_client.close()
        logger.info("Все соединения закрыты")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self.disconnect_all())

# Вспомогательные функции
def create_text_processing_task(text: str, operation: str = "analyze", 
                              priority: int = 1, max_reward: float = 10.0) -> TaskConfig:
    """Создание задачи обработки текста"""
    return TaskConfig(
        task_type=TaskType.TEXT_PROCESSING.value,
        input_data={
            'text': text,
            'operation': operation
        },
        priority=priority,
        max_reward=max_reward,
        requirements={
            'cpu_cores': 1,
            'memory_mb': 512,
            'requires_gpu': False
        }
    )

def create_image_processing_task(image_path: str, operations: List[str],
                               priority: int = 1, max_reward: float = 15.0) -> TaskConfig:
    """Создание задачи обработки изображений"""
    return TaskConfig(
        task_type=TaskType.IMAGE_PROCESSING.value,
        input_data={
            'image_path': image_path,
            'operations': operations
        },
        priority=priority,
        max_reward=max_reward,
        requirements={
            'cpu_cores': 2,
            'memory_mb': 1024,
            'requires_gpu': True,
            'gpu_memory_mb': 512
        }
    )

def create_data_analysis_task(data_path: str, analysis_type: str = "statistics",
                            priority: int = 1, max_reward: float = 20.0) -> TaskConfig:
    """Создание задачи анализа данных"""
    return TaskConfig(
        task_type=TaskType.DATA_ANALYSIS.value,
        input_data={
            'data_path': data_path,
            'analysis_type': analysis_type
        },
        priority=priority,
        max_reward=max_reward,
        requirements={
            'cpu_cores': 4,
            'memory_mb': 2048,
            'requires_gpu': False
        }
    )

def create_node_config(device_id: str = None, device_type: str = "smartphone") -> NodeConfig:
    """Создание конфигурации узла с автоопределением параметров"""
    if not device_id:
        device_id = f"device_{uuid.uuid4().hex[:8]}"
    
    # Автоопределение характеристик системы
    import psutil
    
    cpu_cores = psutil.cpu_count(logical=False) or 1
    memory_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    storage_mb = int(psutil.disk_usage('/').free / (1024 * 1024))
    
    # Определение ОС
    import platform
    os_type = platform.system().lower()
    
    return NodeConfig(
        device_id=device_id,
        device_type=device_type,
        cpu_cores=cpu_cores,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        gpu_available=False,  # Требует дополнительной проверки
        network_speed_mbps=100.0,  # Значение по умолчанию
        os_type=os_type
    )

# Пример использования
async def example_usage():
    """Пример использования SDK"""
    
    # Создание клиента
    client = DaurAIClient("http://localhost:8000")
    
    try:
        # Регистрация пользователя
        success = client.register("test_user", "test@example.com", "password123")
        if not success:
            print("Ошибка регистрации")
            return
        
        # Вход в систему
        success = client.login("test_user", "password123")
        if not success:
            print("Ошибка входа")
            return
        
        print(f"Вошел как: {client.user_info['username']}")
        
        # Подключение к уведомлениям
        await client.connect_notifications()
        
        # Регистрация обработчика уведомлений
        def on_notification(notification):
            print(f"Уведомление: {notification['title']}")
        
        client.on('notification', on_notification)
        
        # Создание и отправка задачи
        task_config = create_text_processing_task(
            "Проанализируйте этот текст на тональность",
            "sentiment_analysis"
        )
        
        task_id = client.submit_task(task_config)
        print(f"Задача отправлена: {task_id}")
        
        # Ожидание результата
        result = client.wait_for_task(task_id, timeout=60)
        print(f"Результат: {result.result_data}")
        
        # Регистрация узла
        node_config = create_node_config()
        node_id = client.register_node(node_config)
        print(f"Узел зарегистрирован: {node_id}")
        
        # Подключение узла
        await client.connect_node(node_id)
        
        # Получение статистики
        stats = client.get_network_stats()
        print(f"Статистика сети: {stats}")
        
        # Загрузка файла
        # file_id = client.file_manager.upload_file("example.txt")
        # print(f"Файл загружен: {file_id}")
        
        # Ожидание событий
        await asyncio.sleep(10)
        
    finally:
        # Отключение
        await client.disconnect_all()

if __name__ == "__main__":
    # Запуск примера
    asyncio.run(example_usage())
