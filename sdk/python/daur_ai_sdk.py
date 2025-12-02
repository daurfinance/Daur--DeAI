"""
Daur-AI Python SDK - Библиотека для взаимодействия с API Daur-AI
Автор: Дауиржан Нуридинулы
"""

import requests
import websocket
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class Task:
    """Класс для представления задачи"""
    task_id: str
    name: str
    status: str
    progress: float
    result_cid: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    estimated_completion_time: Optional[int]


@dataclass
class Node:
    """Класс для представления узла"""
    node_id: str
    address: str
    status: str
    reputation: float
    completed_tasks: int
    failed_tasks: int
    total_earnings: float


@dataclass
class NetworkStats:
    """Класс для статистики сети"""
    total_nodes: int
    online_nodes: int
    pending_tasks: int
    active_assignments: int
    total_assignments: int
    active_auctions: int


class DaurAIException(Exception):
    """Базовое исключение для SDK"""
    pass


class DaurAIClient:
    """Основной клиент для взаимодействия с API Daur-AI"""
    
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        """
        Инициализация клиента
        
        Args:
            api_key: API ключ для аутентификации
            base_url: Базовый URL API сервера
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # WebSocket соединение
        self._ws = None
        self._ws_thread = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Логирование
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Выполняет HTTP запрос к API"""
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            if response.content:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise DaurAIException(f"API request failed: {str(e)}")
    
    # Методы для работы с задачами
    def create_task(self, name: str, description: str, input_cid: str,
                   model_cid: Optional[str] = None, reward: float = 10.0,
                   priority: str = "normal") -> str:
        """
        Создает новую задачу
        
        Args:
            name: Название задачи
            description: Описание задачи
            input_cid: CID входных данных в IPFS
            model_cid: CID модели ИИ в IPFS (опционально)
            reward: Вознаграждение в токенах DAUR
            priority: Приоритет задачи (low, normal, high)
            
        Returns:
            str: ID созданной задачи
        """
        
        data = {
            "name": name,
            "description": description,
            "input_cid": input_cid,
            "model_cid": model_cid,
            "reward": reward,
            "priority": priority
        }
        
        response = self._make_request("POST", "/v1/tasks", json=data)
        return response["task_id"]
    
    def get_task(self, task_id: str) -> Task:
        """
        Получает информацию о задаче
        
        Args:
            task_id: ID задачи
            
        Returns:
            Task: Объект задачи
        """
        
        response = self._make_request("GET", f"/v1/tasks/{task_id}")
        
        return Task(
            task_id=response["task_id"],
            name=response["name"],
            status=response["status"],
            progress=response["progress"],
            result_cid=response["result_cid"],
            created_at=datetime.fromisoformat(response["created_at"].replace('Z', '+00:00')),
            completed_at=datetime.fromisoformat(response["completed_at"].replace('Z', '+00:00')) if response["completed_at"] else None,
            estimated_completion_time=response["estimated_completion_time"]
        )
    
    def list_tasks(self, limit: int = 10, offset: int = 0,
                  status_filter: Optional[str] = None) -> List[Task]:
        """
        Получает список задач пользователя
        
        Args:
            limit: Максимальное количество задач
            offset: Смещение для пагинации
            status_filter: Фильтр по статусу
            
        Returns:
            List[Task]: Список задач
        """
        
        params = {"limit": limit, "offset": offset}
        if status_filter:
            params["status_filter"] = status_filter
        
        response = self._make_request("GET", "/v1/tasks", params=params)
        
        tasks = []
        for task_data in response:
            tasks.append(Task(
                task_id=task_data["task_id"],
                name=task_data["name"],
                status=task_data["status"],
                progress=task_data["progress"],
                result_cid=task_data["result_cid"],
                created_at=datetime.fromisoformat(task_data["created_at"].replace('Z', '+00:00')),
                completed_at=datetime.fromisoformat(task_data["completed_at"].replace('Z', '+00:00')) if task_data["completed_at"] else None,
                estimated_completion_time=task_data["estimated_completion_time"]
            ))
        
        return tasks
    
    def wait_for_task_completion(self, task_id: str, timeout: int = 3600,
                               poll_interval: int = 5) -> Task:
        """
        Ожидает завершения задачи
        
        Args:
            task_id: ID задачи
            timeout: Максимальное время ожидания в секундах
            poll_interval: Интервал опроса в секундах
            
        Returns:
            Task: Завершенная задача
        """
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task = self.get_task(task_id)
            
            if task.status in ["completed", "failed", "cancelled"]:
                return task
            
            time.sleep(poll_interval)
        
        raise DaurAIException(f"Task {task_id} did not complete within {timeout} seconds")
    
    # Методы для работы с узлами
    def register_node(self, address: str, cpu_cores: int, memory_mb: int,
                     gpu_available: bool = False, gpu_memory_mb: int = 0,
                     storage_mb: int = 1000, network_speed_mbps: float = 1.0,
                     battery_level: int = 100, device_type: str = "smartphone",
                     stake_amount: float = 0.0) -> str:
        """
        Регистрирует новый узел в сети
        
        Args:
            address: TON адрес узла
            cpu_cores: Количество ядер CPU
            memory_mb: Объем памяти в МБ
            gpu_available: Наличие GPU
            gpu_memory_mb: Объем памяти GPU в МБ
            storage_mb: Доступное место на диске в МБ
            network_speed_mbps: Скорость сети в Мбит/с
            battery_level: Уровень батареи (0-100)
            device_type: Тип устройства
            stake_amount: Количество застейканных токенов
            
        Returns:
            str: ID зарегистрированного узла
        """
        
        data = {
            "address": address,
            "cpu_cores": cpu_cores,
            "memory_mb": memory_mb,
            "gpu_available": gpu_available,
            "gpu_memory_mb": gpu_memory_mb,
            "storage_mb": storage_mb,
            "network_speed_mbps": network_speed_mbps,
            "battery_level": battery_level,
            "device_type": device_type,
            "stake_amount": stake_amount
        }
        
        response = self._make_request("POST", "/v1/nodes/register", json=data)
        return response["node_id"]
    
    def get_node(self, node_id: str) -> Node:
        """
        Получает информацию об узле
        
        Args:
            node_id: ID узла
            
        Returns:
            Node: Объект узла
        """
        
        response = self._make_request("GET", f"/v1/nodes/{node_id}")
        
        return Node(
            node_id=response["node_id"],
            address=response["address"],
            status=response["status"],
            reputation=response["reputation"],
            completed_tasks=response["completed_tasks"],
            failed_tasks=response["failed_tasks"],
            total_earnings=response["total_earnings"]
        )
    
    def list_nodes(self, limit: int = 50, offset: int = 0,
                  status_filter: Optional[str] = None,
                  min_reputation: float = 0.0) -> List[Node]:
        """
        Получает список узлов в сети
        
        Args:
            limit: Максимальное количество узлов
            offset: Смещение для пагинации
            status_filter: Фильтр по статусу
            min_reputation: Минимальная репутация
            
        Returns:
            List[Node]: Список узлов
        """
        
        params = {
            "limit": limit,
            "offset": offset,
            "min_reputation": min_reputation
        }
        if status_filter:
            params["status_filter"] = status_filter
        
        response = self._make_request("GET", "/v1/nodes", params=params)
        
        nodes = []
        for node_data in response:
            nodes.append(Node(
                node_id=node_data["node_id"],
                address=node_data["address"],
                status=node_data["status"],
                reputation=node_data["reputation"],
                completed_tasks=node_data["completed_tasks"],
                failed_tasks=node_data["failed_tasks"],
                total_earnings=node_data["total_earnings"]
            ))
        
        return nodes
    
    def update_node_status(self, node_id: str, status: str,
                          capabilities: Optional[Dict] = None) -> bool:
        """
        Обновляет статус узла
        
        Args:
            node_id: ID узла
            status: Новый статус
            capabilities: Обновленные возможности узла
            
        Returns:
            bool: Успешность операции
        """
        
        data = {"status": status}
        if capabilities:
            data["capabilities"] = capabilities
        
        try:
            self._make_request("PUT", f"/v1/nodes/{node_id}/status", json=data)
            return True
        except DaurAIException:
            return False
    
    # Методы для работы со ставками
    def submit_bid(self, subtask_id: str, bid_amount: float,
                  estimated_time: int, confidence: float) -> bool:
        """
        Отправляет ставку на выполнение подзадачи
        
        Args:
            subtask_id: ID подзадачи
            bid_amount: Предлагаемая цена
            estimated_time: Оценочное время выполнения в секундах
            confidence: Уверенность в выполнении (0-1)
            
        Returns:
            bool: Успешность отправки ставки
        """
        
        data = {
            "subtask_id": subtask_id,
            "bid_amount": bid_amount,
            "estimated_time": estimated_time,
            "confidence": confidence
        }
        
        try:
            self._make_request("POST", "/v1/bids", json=data)
            return True
        except DaurAIException:
            return False
    
    # Методы для получения статистики
    def get_network_stats(self) -> NetworkStats:
        """
        Получает статистику сети
        
        Returns:
            NetworkStats: Статистика сети
        """
        
        response = self._make_request("GET", "/v1/stats/network")
        
        return NetworkStats(
            total_nodes=response["total_nodes"],
            online_nodes=response["online_nodes"],
            pending_tasks=response["pending_tasks"],
            active_assignments=response["active_assignments"],
            total_assignments=response["total_assignments"],
            active_auctions=response["active_auctions"]
        )
    
    def get_task_stats(self) -> Dict[str, Any]:
        """
        Получает статистику задач
        
        Returns:
            Dict[str, Any]: Статистика задач
        """
        
        return self._make_request("GET", "/v1/stats/tasks")
    
    # Методы для работы с моделями
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Получает список доступных моделей ИИ
        
        Returns:
            List[Dict[str, Any]]: Список моделей
        """
        
        return self._make_request("GET", "/v1/models")
    
    def get_model(self, model_cid: str) -> Dict[str, Any]:
        """
        Получает информацию о конкретной модели
        
        Args:
            model_cid: CID модели
            
        Returns:
            Dict[str, Any]: Информация о модели
        """
        
        return self._make_request("GET", f"/v1/models/{model_cid}")
    
    # WebSocket методы для реального времени
    def connect_websocket(self):
        """Устанавливает WebSocket соединение"""
        
        if self._ws is not None:
            return
        
        ws_url = self.base_url.replace('http', 'ws') + '/v1/ws'
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                event_type = data.get('event')
                
                if event_type in self._event_handlers:
                    for handler in self._event_handlers[event_type]:
                        handler(data.get('data', {}))
                        
            except Exception as e:
                self.logger.error(f"Error processing WebSocket message: {str(e)}")
        
        def on_error(ws, error):
            self.logger.error(f"WebSocket error: {str(error)}")
        
        def on_close(ws, close_status_code, close_msg):
            self.logger.info("WebSocket connection closed")
        
        def on_open(ws):
            self.logger.info("WebSocket connection established")
        
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        # Запускаем WebSocket в отдельном потоке
        self._ws_thread = threading.Thread(target=self._ws.run_forever)
        self._ws_thread.daemon = True
        self._ws_thread.start()
    
    def disconnect_websocket(self):
        """Закрывает WebSocket соединение"""
        
        if self._ws is not None:
            self._ws.close()
            self._ws = None
            self._ws_thread = None
    
    def subscribe_to_task_updates(self, task_id: str, callback: Callable):
        """
        Подписывается на обновления задачи
        
        Args:
            task_id: ID задачи
            callback: Функция обратного вызова
        """
        
        if 'task_update' not in self._event_handlers:
            self._event_handlers['task_update'] = []
        
        def filtered_callback(data):
            if data.get('task_id') == task_id:
                callback(data)
        
        self._event_handlers['task_update'].append(filtered_callback)
        
        # Отправляем команду подписки
        if self._ws is not None:
            subscribe_message = {
                "action": "subscribe",
                "event": "task_update",
                "task_id": task_id
            }
            self._ws.send(json.dumps(subscribe_message))
    
    def add_event_handler(self, event_type: str, callback: Callable):
        """
        Добавляет обработчик события
        
        Args:
            event_type: Тип события
            callback: Функция обратного вызова
        """
        
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        
        self._event_handlers[event_type].append(callback)
    
    # Утилитарные методы
    def health_check(self) -> bool:
        """
        Проверяет доступность API
        
        Returns:
            bool: True если API доступен
        """
        
        try:
            response = self._make_request("GET", "/health")
            return response.get("status") == "healthy"
        except DaurAIException:
            return False


# Вспомогательные функции
def create_client(api_key: str, base_url: str = "http://localhost:8000") -> DaurAIClient:
    """
    Создает клиент Daur-AI
    
    Args:
        api_key: API ключ
        base_url: Базовый URL API
        
    Returns:
        DaurAIClient: Клиент для работы с API
    """
    
    return DaurAIClient(api_key, base_url)


# Пример использования
def main():
    """Пример использования SDK"""
    
    # Создаем клиент
    client = create_client("test_key_123")
    
    # Проверяем доступность API
    if not client.health_check():
        print("API недоступен")
        return
    
    print("API доступен")
    
    # Получаем статистику сети
    stats = client.get_network_stats()
    print(f"Узлов в сети: {stats.total_nodes}")
    print(f"Онлайн узлов: {stats.online_nodes}")
    
    # Создаем задачу
    task_id = client.create_task(
        name="Тестовая задача",
        description="Обработка тестовых данных с помощью ИИ",
        input_cid="QmTestInputData123",
        model_cid="QmTestModel456",
        reward=50.0,
        priority="normal"
    )
    
    print(f"Создана задача: {task_id}")
    
    # Получаем информацию о задаче
    task = client.get_task(task_id)
    print(f"Статус задачи: {task.status}")
    print(f"Прогресс: {task.progress}%")
    
    # Подключаемся к WebSocket для получения обновлений
    client.connect_websocket()
    
    def on_task_update(data):
        print(f"Обновление задачи {data['task_id']}: {data['status']}")
    
    client.subscribe_to_task_updates(task_id, on_task_update)
    
    # Ожидаем завершения задачи (с таймаутом)
    try:
        completed_task = client.wait_for_task_completion(task_id, timeout=300)
        print(f"Задача завершена со статусом: {completed_task.status}")
        
        if completed_task.result_cid:
            print(f"Результат доступен по CID: {completed_task.result_cid}")
            
    except DaurAIException as e:
        print(f"Ошибка при ожидании завершения задачи: {str(e)}")
    
    # Отключаемся от WebSocket
    client.disconnect_websocket()


if __name__ == "__main__":
    main()
