"""
Daur-AI API Server - REST API для взаимодействия с сетью Daur-AI
Автор: Дауиржан Нуридинулы
"""

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import json
import uuid
import hashlib
import time
from datetime import datetime, timedelta
import logging
import os
import sys

# Добавляем путь к модулям
sys.path.append('/home/ubuntu/daur-ai-complete/code/backend')

from main_ai.task_analyzer import TaskAnalyzer, TaskGraph
from orchestrator.task_orchestrator import TaskOrchestrator


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание приложения FastAPI
app = FastAPI(
    title="Daur-AI API",
    description="API для децентрализованной сети искусственного интеллекта Daur-AI",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация компонентов
task_analyzer = TaskAnalyzer()
orchestrator = TaskOrchestrator()

# Безопасность
security = HTTPBearer()

# Хранилище API ключей (в продакшене должно быть в базе данных)
API_KEYS = {
    "test_key_123": {
        "user_id": "user_001",
        "permissions": ["read", "write"],
        "created_at": datetime.now()
    }
}

# WebSocket соединения
websocket_connections: Dict[str, WebSocket] = {}


# Модели данных
class TaskCreateRequest(BaseModel):
    name: str = Field(..., description="Название задачи")
    description: str = Field(..., description="Описание задачи")
    input_cid: str = Field(..., description="CID входных данных в IPFS")
    model_cid: Optional[str] = Field(None, description="CID модели ИИ в IPFS")
    reward: float = Field(..., gt=0, description="Вознаграждение в токенах DAUR")
    priority: str = Field("normal", description="Приоритет задачи (low, normal, high)")


class TaskResponse(BaseModel):
    task_id: str
    name: str
    status: str
    progress: float
    result_cid: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    estimated_completion_time: Optional[int]


class NodeRegistrationRequest(BaseModel):
    address: str = Field(..., description="TON адрес узла")
    cpu_cores: int = Field(..., gt=0, description="Количество ядер CPU")
    memory_mb: int = Field(..., gt=0, description="Объем памяти в МБ")
    gpu_available: bool = Field(False, description="Наличие GPU")
    gpu_memory_mb: int = Field(0, description="Объем памяти GPU в МБ")
    storage_mb: int = Field(..., gt=0, description="Доступное место на диске в МБ")
    network_speed_mbps: float = Field(..., gt=0, description="Скорость сети в Мбит/с")
    battery_level: int = Field(100, ge=0, le=100, description="Уровень батареи (0-100)")
    device_type: str = Field("smartphone", description="Тип устройства")
    stake_amount: float = Field(0.0, ge=0, description="Количество застейканных токенов")


class NodeResponse(BaseModel):
    node_id: str
    address: str
    status: str
    reputation: float
    completed_tasks: int
    failed_tasks: int
    total_earnings: float


class BidRequest(BaseModel):
    subtask_id: str = Field(..., description="ID подзадачи")
    bid_amount: float = Field(..., gt=0, description="Предлагаемая цена")
    estimated_time: int = Field(..., gt=0, description="Оценочное время выполнения в секундах")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность в выполнении (0-1)")


class NetworkStatsResponse(BaseModel):
    total_nodes: int
    online_nodes: int
    pending_tasks: int
    active_assignments: int
    total_assignments: int
    active_auctions: int


# Хранилище задач (в продакшене должно быть в базе данных)
tasks_storage: Dict[str, Dict[str, Any]] = {}


# Функции аутентификации
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Проверяет API ключ"""
    api_key = credentials.credentials
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return API_KEYS[api_key]


# Эндпоинты для управления задачами
@app.post("/v1/tasks", response_model=Dict[str, str])
async def create_task(
    task_request: TaskCreateRequest,
    user_info: Dict = Depends(verify_api_key)
):
    """Создает новую задачу"""
    
    try:
        # Генерируем уникальный ID задачи
        task_id = str(uuid.uuid4())
        
        # Анализируем и декомпозируем задачу
        task_graph = task_analyzer.decompose_task(
            task_id=task_id,
            task_description=task_request.description,
            input_data_cid=task_request.input_cid,
            model_cid=task_request.model_cid,
            reward=task_request.reward,
            priority=task_request.priority
        )
        
        # Сохраняем задачу
        tasks_storage[task_id] = {
            "id": task_id,
            "name": task_request.name,
            "description": task_request.description,
            "input_cid": task_request.input_cid,
            "model_cid": task_request.model_cid,
            "reward": task_request.reward,
            "priority": task_request.priority,
            "status": "created",
            "progress": 0.0,
            "result_cid": None,
            "created_at": datetime.now(),
            "completed_at": None,
            "user_id": user_info["user_id"],
            "task_graph": task_graph.to_dict(),
            "estimated_completion_time": task_graph.estimated_completion_time
        }
        
        # Отправляем задачу в оркестратор
        await orchestrator.submit_task_graph(task_graph.to_dict())
        
        # Уведомляем WebSocket клиентов
        await broadcast_task_update(task_id, "created")
        
        logger.info(f"Task {task_id} created successfully")
        
        return {
            "task_id": task_id,
            "status": "created",
            "message": "Task created and submitted for processing"
        }
        
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )


@app.get("/v1/tasks/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    user_info: Dict = Depends(verify_api_key)
):
    """Получает информацию о задаче"""
    
    if task_id not in tasks_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task = tasks_storage[task_id]
    
    # Проверяем права доступа
    if task["user_id"] != user_info["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return TaskResponse(
        task_id=task["id"],
        name=task["name"],
        status=task["status"],
        progress=task["progress"],
        result_cid=task["result_cid"],
        created_at=task["created_at"],
        completed_at=task["completed_at"],
        estimated_completion_time=task["estimated_completion_time"]
    )


@app.get("/v1/tasks", response_model=List[TaskResponse])
async def list_tasks(
    limit: int = 10,
    offset: int = 0,
    status_filter: Optional[str] = None,
    user_info: Dict = Depends(verify_api_key)
):
    """Получает список задач пользователя"""
    
    user_tasks = [
        task for task in tasks_storage.values()
        if task["user_id"] == user_info["user_id"]
    ]
    
    # Фильтрация по статусу
    if status_filter:
        user_tasks = [task for task in user_tasks if task["status"] == status_filter]
    
    # Сортировка по дате создания (новые первыми)
    user_tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Пагинация
    paginated_tasks = user_tasks[offset:offset + limit]
    
    return [
        TaskResponse(
            task_id=task["id"],
            name=task["name"],
            status=task["status"],
            progress=task["progress"],
            result_cid=task["result_cid"],
            created_at=task["created_at"],
            completed_at=task["completed_at"],
            estimated_completion_time=task["estimated_completion_time"]
        )
        for task in paginated_tasks
    ]


# Эндпоинты для управления узлами
@app.post("/v1/nodes/register", response_model=Dict[str, str])
async def register_node(node_request: NodeRegistrationRequest):
    """Регистрирует новый узел в сети"""
    
    try:
        node_id = await orchestrator.register_node(node_request.dict())
        
        logger.info(f"Node {node_id} registered successfully")
        
        return {
            "node_id": node_id,
            "status": "registered",
            "message": "Node registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Error registering node: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register node: {str(e)}"
        )


@app.get("/v1/nodes", response_model=List[NodeResponse])
async def list_nodes(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    min_reputation: float = 0.0
):
    """Получает список узлов в сети"""
    
    # В реальной реализации это будет запрос к базе данных
    # Здесь используем данные из оркестратора
    all_nodes = []
    
    for node_id, node in orchestrator.nodes.items():
        if (not status_filter or node.status.value == status_filter) and \
           node.reputation >= min_reputation:
            all_nodes.append(NodeResponse(
                node_id=node_id,
                address=node.address,
                status=node.status.value,
                reputation=node.reputation,
                completed_tasks=node.completed_tasks,
                failed_tasks=node.failed_tasks,
                total_earnings=node.total_earnings
            ))
    
    # Сортировка по репутации
    all_nodes.sort(key=lambda x: x.reputation, reverse=True)
    
    return all_nodes[offset:offset + limit]


@app.get("/v1/nodes/{node_id}", response_model=NodeResponse)
async def get_node(node_id: str):
    """Получает информацию о конкретном узле"""
    
    node_info = await orchestrator.get_node_info(node_id)
    
    if not node_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found"
        )
    
    return NodeResponse(
        node_id=node_id,
        address=node_info["address"],
        status=node_info["status"],
        reputation=node_info["reputation"],
        completed_tasks=node_info["completed_tasks"],
        failed_tasks=node_info["failed_tasks"],
        total_earnings=node_info["total_earnings"]
    )


@app.put("/v1/nodes/{node_id}/status")
async def update_node_status(
    node_id: str,
    status_update: Dict[str, Any]
):
    """Обновляет статус узла"""
    
    success = await orchestrator.update_node_status(
        node_id=node_id,
        status=status_update.get("status"),
        capabilities=status_update.get("capabilities")
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found"
        )
    
    return {"message": "Node status updated successfully"}


# Эндпоинты для аукционов и ставок
@app.post("/v1/bids")
async def submit_bid(bid_request: BidRequest):
    """Отправляет ставку на выполнение подзадачи"""
    
    # В реальной реализации здесь должна быть аутентификация узла
    # Для демонстрации используем простую проверку
    
    from orchestrator.task_orchestrator import Bid
    
    bid = Bid(
        node_id="demo_node",  # В реальности получаем из аутентификации
        subtask_id=bid_request.subtask_id,
        bid_amount=bid_request.bid_amount,
        estimated_time=bid_request.estimated_time,
        confidence=bid_request.confidence,
        submitted_at=datetime.now()
    )
    
    success = await orchestrator.submit_bid(bid)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to submit bid"
        )
    
    return {"message": "Bid submitted successfully"}


# Эндпоинты для статистики
@app.get("/v1/stats/network", response_model=NetworkStatsResponse)
async def get_network_stats():
    """Получает статистику сети"""
    
    stats = await orchestrator.get_network_stats()
    
    return NetworkStatsResponse(**stats)


@app.get("/v1/stats/tasks")
async def get_task_stats():
    """Получает статистику задач"""
    
    total_tasks = len(tasks_storage)
    status_counts = {}
    
    for task in tasks_storage.values():
        status = task["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "total_tasks": total_tasks,
        "status_distribution": status_counts
    }


# WebSocket для реального времени
@app.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket соединение для получения обновлений в реальном времени"""
    
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    websocket_connections[connection_id] = websocket
    
    try:
        while True:
            # Ожидаем сообщения от клиента
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Обрабатываем подписки на события
            if message.get("action") == "subscribe":
                # В реальной реализации здесь будет логика подписки
                await websocket.send_text(json.dumps({
                    "status": "subscribed",
                    "event": message.get("event")
                }))
            
    except WebSocketDisconnect:
        del websocket_connections[connection_id]
        logger.info(f"WebSocket connection {connection_id} disconnected")


async def broadcast_task_update(task_id: str, status: str):
    """Отправляет обновление задачи всем подключенным WebSocket клиентам"""
    
    message = {
        "event": "task_update",
        "data": {
            "task_id": task_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Отправляем сообщение всем подключенным клиентам
    disconnected = []
    for connection_id, websocket in websocket_connections.items():
        try:
            await websocket.send_text(json.dumps(message))
        except:
            disconnected.append(connection_id)
    
    # Удаляем отключенные соединения
    for connection_id in disconnected:
        del websocket_connections[connection_id]


# Эндпоинты для моделей
@app.get("/v1/models")
async def list_models():
    """Получает список доступных моделей ИИ"""
    
    # В реальной реализации это будет запрос к реестру моделей
    models = [
        {
            "model_cid": "QmTextTranslationModel123",
            "name": "Neural Machine Translation",
            "version": "1.2.0",
            "description": "Модель для перевода текста между языками",
            "task_types": ["text_processing", "natural_language"],
            "size_mb": 150,
            "accuracy": 0.92
        },
        {
            "model_cid": "QmImageClassifierModel456",
            "name": "ImageNet Classifier",
            "version": "2.1.0",
            "description": "Классификатор изображений на основе ResNet",
            "task_types": ["image_classification", "computer_vision"],
            "size_mb": 250,
            "accuracy": 0.89
        }
    ]
    
    return models


@app.get("/v1/models/{model_cid}")
async def get_model(model_cid: str):
    """Получает информацию о конкретной модели"""
    
    # Заглушка для демонстрации
    if model_cid == "QmTextTranslationModel123":
        return {
            "model_cid": model_cid,
            "name": "Neural Machine Translation",
            "version": "1.2.0",
            "description": "Модель для перевода текста между языками",
            "task_types": ["text_processing", "natural_language"],
            "size_mb": 150,
            "accuracy": 0.92,
            "created_at": "2025-01-01T00:00:00Z",
            "downloads": 1250
        }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model not found"
    )


# Эндпоинт для здоровья сервиса
@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Запуск фоновых задач
@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    
    logger.info("Starting Daur-AI API Server")
    
    # Запускаем фоновую задачу для очистки неактивных узлов
    asyncio.create_task(cleanup_inactive_nodes_task())


async def cleanup_inactive_nodes_task():
    """Фоновая задача для очистки неактивных узлов"""
    
    while True:
        try:
            await orchestrator.cleanup_inactive_nodes()
            await asyncio.sleep(300)  # Каждые 5 минут
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
