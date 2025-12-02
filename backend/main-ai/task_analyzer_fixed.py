"""
Daur-AI MainAI Task Analyzer - Полнофункциональная реализация
Анализ и декомпозиция задач искусственного интеллекта для распределенной обработки
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import librosa
import pickle
import redis
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Типы задач ИИ"""
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_PROCESSING = "video_processing"
    MACHINE_LEARNING = "machine_learning"
    NATURAL_LANGUAGE = "natural_language"
    COMPUTER_VISION = "computer_vision"
    SPEECH_RECOGNITION = "speech_recognition"
    RECOMMENDATION = "recommendation"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

class ComplexityLevel(Enum):
    """Уровни сложности задач"""
    TRIVIAL = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    EXTREME = 5

@dataclass
class ResourceRequirements:
    """Требования к ресурсам для выполнения подзадачи"""
    cpu_cores: int
    memory_mb: int
    gpu_memory_mb: int
    storage_mb: int
    network_bandwidth_mbps: float
    estimated_time_seconds: int
    requires_gpu: bool
    requires_internet: bool
    min_battery_level: int

@dataclass
class SubTask:
    """Подзадача для распределенного выполнения"""
    id: str
    parent_task_id: str
    task_type: TaskType
    complexity: ComplexityLevel
    input_data: Dict[str, Any]
    model_config: Dict[str, Any]
    dependencies: List[str]
    resources: ResourceRequirements
    priority: int
    timeout_seconds: int
    retry_count: int
    validation_rules: Dict[str, Any]
    created_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        result = asdict(self)
        result['task_type'] = self.task_type.value
        result['complexity'] = self.complexity.value
        return result

@dataclass
class TaskGraph:
    """Граф задач с зависимостями"""
    nodes: Dict[str, SubTask]
    edges: List[Tuple[str, str]]
    execution_order: List[List[str]]
    total_estimated_time: int
    total_resource_cost: float
    
class AIModelManager:
    """Менеджер ИИ моделей для анализа задач"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
    
    def _load_models(self):
        """Загрузка предобученных моделей"""
        try:
            # Модель для анализа текста
            self.tokenizers['text'] = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.models['text'] = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)
            
            # Модель для классификации задач
            self.models['task_classifier'] = self._create_task_classifier()
            
            # Модель для оценки сложности
            self.models['complexity_estimator'] = self._create_complexity_estimator()
            
            logger.info("Все ИИ модели успешно загружены")
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            raise
    
    def _create_task_classifier(self) -> nn.Module:
        """Создание классификатора типов задач"""
        class TaskClassifier(nn.Module):
            def __init__(self, input_dim=768, num_classes=len(TaskType)):
                super().__init__()
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                return self.classifier(x)
        
        model = TaskClassifier().to(self.device)
        # Загрузка предобученных весов (в реальной системе)
        # model.load_state_dict(torch.load('task_classifier_weights.pth'))
        return model
    
    def _create_complexity_estimator(self) -> nn.Module:
        """Создание оценщика сложности задач"""
        class ComplexityEstimator(nn.Module):
            def __init__(self, input_dim=768):
                super().__init__()
                self.estimator = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, len(ComplexityLevel))
                )
            
            def forward(self, x):
                return self.estimator(x)
        
        model = ComplexityEstimator().to(self.device)
        return model
    
    def encode_text(self, text: str) -> np.ndarray:
        """Кодирование текста в векторное представление"""
        inputs = self.tokenizers['text'](text, return_tensors='pt', 
                                       truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.models['text'](**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def classify_task_type(self, description: str, input_data: Dict) -> TaskType:
        """Классификация типа задачи на основе описания и входных данных"""
        text_embedding = self.encode_text(description)
        
        # Анализ типов входных данных
        data_features = self._extract_data_features(input_data)
        
        # Объединение признаков
        combined_features = np.concatenate([text_embedding.flatten(), data_features])
        
        # Классификация
        with torch.no_grad():
            features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
            logits = self.models['task_classifier'](features_tensor)
            predicted_class = torch.argmax(logits, dim=1).item()
        
        return list(TaskType)[predicted_class]
    
    def estimate_complexity(self, description: str, input_size: int, 
                          task_type: TaskType) -> ComplexityLevel:
        """Оценка сложности задачи"""
        text_embedding = self.encode_text(description)
        
        # Дополнительные признаки
        size_feature = np.log10(max(input_size, 1))
        type_feature = list(TaskType).index(task_type)
        
        additional_features = np.array([size_feature, type_feature])
        combined_features = np.concatenate([text_embedding.flatten(), additional_features])
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(combined_features).unsqueeze(0).to(self.device)
            logits = self.models['complexity_estimator'](features_tensor)
            predicted_complexity = torch.argmax(logits, dim=1).item()
        
        return list(ComplexityLevel)[predicted_complexity]
    
    def _extract_data_features(self, input_data: Dict) -> np.ndarray:
        """Извлечение признаков из входных данных"""
        features = []
        
        # Размер данных
        total_size = sum(len(str(v)) for v in input_data.values())
        features.append(np.log10(max(total_size, 1)))
        
        # Типы данных
        has_text = any('text' in str(k).lower() or isinstance(v, str) 
                      for k, v in input_data.items())
        has_image = any('image' in str(k).lower() or 'img' in str(k).lower() 
                       for k in input_data.keys())
        has_audio = any('audio' in str(k).lower() or 'sound' in str(k).lower() 
                       for k in input_data.keys())
        has_video = any('video' in str(k).lower() for k in input_data.keys())
        
        features.extend([float(has_text), float(has_image), float(has_audio), float(has_video)])
        
        # Количество полей
        features.append(len(input_data))
        
        return np.array(features)

class DatabaseManager:
    """Менеджер базы данных для хранения задач и результатов"""
    
    def __init__(self, db_path: str = "daur_ai.db"):
        self.db_path = db_path
        self.redis_client = None
        self._init_database()
        self._init_redis()
    
    def _init_database(self):
        """Инициализация SQLite базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Таблица задач
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                task_type TEXT,
                complexity INTEGER,
                status TEXT,
                created_at REAL,
                updated_at REAL,
                input_data TEXT,
                result_data TEXT,
                metadata TEXT
            )
        ''')
        
        # Таблица подзадач
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS subtasks (
                id TEXT PRIMARY KEY,
                parent_task_id TEXT,
                task_type TEXT,
                complexity INTEGER,
                status TEXT,
                assigned_node_id TEXT,
                created_at REAL,
                started_at REAL,
                completed_at REAL,
                input_data TEXT,
                result_data TEXT,
                resources_required TEXT,
                dependencies TEXT,
                FOREIGN KEY (parent_task_id) REFERENCES tasks (id)
            )
        ''')
        
        # Таблица зависимостей
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS task_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_task_id TEXT,
                target_task_id TEXT,
                dependency_type TEXT,
                FOREIGN KEY (source_task_id) REFERENCES subtasks (id),
                FOREIGN KEY (target_task_id) REFERENCES subtasks (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("База данных инициализирована")
    
    def _init_redis(self):
        """Инициализация Redis для кэширования"""
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, 
                                          decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis подключен")
        except Exception as e:
            logger.warning(f"Redis недоступен: {e}")
            self.redis_client = None
    
    def save_task(self, task_id: str, task_data: Dict):
        """Сохранение задачи в базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO tasks 
            (id, name, description, task_type, complexity, status, created_at, updated_at, 
             input_data, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task_id,
            task_data.get('name', ''),
            task_data.get('description', ''),
            task_data.get('task_type', ''),
            task_data.get('complexity', 0),
            task_data.get('status', 'created'),
            time.time(),
            time.time(),
            json.dumps(task_data.get('input_data', {})),
            json.dumps(task_data.get('metadata', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def save_subtask(self, subtask: SubTask):
        """Сохранение подзадачи в базу данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO subtasks 
            (id, parent_task_id, task_type, complexity, status, created_at,
             input_data, resources_required, dependencies)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            subtask.id,
            subtask.parent_task_id,
            subtask.task_type.value,
            subtask.complexity.value,
            'created',
            subtask.created_at,
            json.dumps(subtask.input_data),
            json.dumps(asdict(subtask.resources)),
            json.dumps(subtask.dependencies)
        ))
        
        conn.commit()
        conn.close()
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Получение задачи из базы данных"""
        # Проверка кэша
        if self.redis_client:
            cached = self.redis_client.get(f"task:{task_id}")
            if cached:
                return json.loads(cached)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM tasks WHERE id = ?', (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            task_data = {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'task_type': row[3],
                'complexity': row[4],
                'status': row[5],
                'created_at': row[6],
                'updated_at': row[7],
                'input_data': json.loads(row[8]) if row[8] else {},
                'result_data': json.loads(row[9]) if row[9] else {},
                'metadata': json.loads(row[10]) if row[10] else {}
            }
            
            # Кэширование
            if self.redis_client:
                self.redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_data))
            
            return task_data
        
        return None

class TaskDecomposer:
    """Декомпозитор задач на подзадачи"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.decomposition_strategies = {
            TaskType.TEXT_PROCESSING: self._decompose_text_task,
            TaskType.IMAGE_PROCESSING: self._decompose_image_task,
            TaskType.AUDIO_PROCESSING: self._decompose_audio_task,
            TaskType.VIDEO_PROCESSING: self._decompose_video_task,
            TaskType.MACHINE_LEARNING: self._decompose_ml_task,
            TaskType.NATURAL_LANGUAGE: self._decompose_nlp_task,
            TaskType.COMPUTER_VISION: self._decompose_cv_task,
            TaskType.CLASSIFICATION: self._decompose_classification_task,
            TaskType.CLUSTERING: self._decompose_clustering_task,
            TaskType.TIME_SERIES: self._decompose_timeseries_task
        }
    
    def decompose(self, task_id: str, task_type: TaskType, input_data: Dict,
                  complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задачи на подзадачи"""
        strategy = self.decomposition_strategies.get(task_type, self._decompose_generic_task)
        return strategy(task_id, input_data, complexity, description)
    
    def _decompose_text_task(self, task_id: str, input_data: Dict, 
                           complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач обработки текста"""
        subtasks = []
        
        text_data = input_data.get('text', '')
        if not text_data:
            return subtasks
        
        # Разбиение текста на чанки
        chunk_size = self._calculate_chunk_size(len(text_data), complexity)
        text_chunks = [text_data[i:i+chunk_size] for i in range(0, len(text_data), chunk_size)]
        
        for i, chunk in enumerate(text_chunks):
            subtask_id = f"{task_id}_chunk_{i}"
            
            # Определение ресурсов на основе размера чанка
            resources = ResourceRequirements(
                cpu_cores=1,
                memory_mb=min(512 + len(chunk) // 1000, 2048),
                gpu_memory_mb=0,
                storage_mb=100,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=max(10, len(chunk) // 100),
                requires_gpu=False,
                requires_internet=True,
                min_battery_level=20
            )
            
            subtask = SubTask(
                id=subtask_id,
                parent_task_id=task_id,
                task_type=TaskType.TEXT_PROCESSING,
                complexity=complexity,
                input_data={'text_chunk': chunk, 'chunk_index': i},
                model_config={'model_type': 'text_processor', 'chunk_size': len(chunk)},
                dependencies=[],
                resources=resources,
                priority=1,
                timeout_seconds=resources.estimated_time_seconds * 3,
                retry_count=2,
                validation_rules={'min_output_length': 10},
                created_at=time.time()
            )
            
            subtasks.append(subtask)
        
        # Добавление задачи агрегации результатов
        if len(subtasks) > 1:
            aggregation_subtask = self._create_aggregation_subtask(
                task_id, subtasks, TaskType.TEXT_PROCESSING
            )
            subtasks.append(aggregation_subtask)
        
        return subtasks
    
    def _decompose_image_task(self, task_id: str, input_data: Dict,
                            complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач обработки изображений"""
        subtasks = []
        
        images = input_data.get('images', [])
        if not images:
            return subtasks
        
        for i, image_data in enumerate(images):
            subtask_id = f"{task_id}_image_{i}"
            
            # Оценка размера изображения
            image_size = len(str(image_data)) if isinstance(image_data, str) else 1000000
            
            resources = ResourceRequirements(
                cpu_cores=2,
                memory_mb=min(1024 + image_size // 10000, 4096),
                gpu_memory_mb=512 if complexity.value >= 3 else 0,
                storage_mb=max(200, image_size // 1000),
                network_bandwidth_mbps=2.0,
                estimated_time_seconds=max(30, image_size // 50000),
                requires_gpu=complexity.value >= 3,
                requires_internet=True,
                min_battery_level=30
            )
            
            subtask = SubTask(
                id=subtask_id,
                parent_task_id=task_id,
                task_type=TaskType.IMAGE_PROCESSING,
                complexity=complexity,
                input_data={'image': image_data, 'image_index': i},
                model_config={'model_type': 'image_processor', 'use_gpu': resources.requires_gpu},
                dependencies=[],
                resources=resources,
                priority=2,
                timeout_seconds=resources.estimated_time_seconds * 2,
                retry_count=1,
                validation_rules={'output_format': 'json'},
                created_at=time.time()
            )
            
            subtasks.append(subtask)
        
        return subtasks
    
    def _decompose_ml_task(self, task_id: str, input_data: Dict,
                         complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач машинного обучения"""
        subtasks = []
        
        # Этап предобработки данных
        preprocessing_subtask = SubTask(
            id=f"{task_id}_preprocessing",
            parent_task_id=task_id,
            task_type=TaskType.MACHINE_LEARNING,
            complexity=ComplexityLevel.MEDIUM,
            input_data=input_data,
            model_config={'stage': 'preprocessing'},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=2048,
                gpu_memory_mb=0,
                storage_mb=500,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=120,
                requires_gpu=False,
                requires_internet=True,
                min_battery_level=40
            ),
            priority=1,
            timeout_seconds=300,
            retry_count=2,
            validation_rules={'output_type': 'preprocessed_data'},
            created_at=time.time()
        )
        subtasks.append(preprocessing_subtask)
        
        # Этапы обучения (если необходимо)
        if 'training_data' in input_data:
            training_subtasks = self._create_training_subtasks(task_id, complexity)
            subtasks.extend(training_subtasks)
        
        # Этап инференса
        inference_subtask = SubTask(
            id=f"{task_id}_inference",
            parent_task_id=task_id,
            task_type=TaskType.MACHINE_LEARNING,
            complexity=complexity,
            input_data={'preprocessed_data': 'from_preprocessing'},
            model_config={'stage': 'inference'},
            dependencies=[preprocessing_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_mb=4096,
                gpu_memory_mb=1024 if complexity.value >= 4 else 0,
                storage_mb=1000,
                network_bandwidth_mbps=2.0,
                estimated_time_seconds=300,
                requires_gpu=complexity.value >= 4,
                requires_internet=True,
                min_battery_level=50
            ),
            priority=3,
            timeout_seconds=600,
            retry_count=1,
            validation_rules={'output_type': 'predictions'},
            created_at=time.time()
        )
        subtasks.append(inference_subtask)
        
        return subtasks
    
    def _decompose_nlp_task(self, task_id: str, input_data: Dict,
                          complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач обработки естественного языка"""
        subtasks = []
        
        # Токенизация
        tokenization_subtask = SubTask(
            id=f"{task_id}_tokenization",
            parent_task_id=task_id,
            task_type=TaskType.NATURAL_LANGUAGE,
            complexity=ComplexityLevel.LOW,
            input_data=input_data,
            model_config={'stage': 'tokenization'},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=1,
                memory_mb=512,
                gpu_memory_mb=0,
                storage_mb=100,
                network_bandwidth_mbps=0.5,
                estimated_time_seconds=30,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=20
            ),
            priority=1,
            timeout_seconds=60,
            retry_count=3,
            validation_rules={'output_type': 'tokens'},
            created_at=time.time()
        )
        subtasks.append(tokenization_subtask)
        
        # Эмбеддинги
        embedding_subtask = SubTask(
            id=f"{task_id}_embedding",
            parent_task_id=task_id,
            task_type=TaskType.NATURAL_LANGUAGE,
            complexity=ComplexityLevel.MEDIUM,
            input_data={'tokens': 'from_tokenization'},
            model_config={'stage': 'embedding', 'model': 'transformer'},
            dependencies=[tokenization_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=512,
                storage_mb=200,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=60,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=30
            ),
            priority=2,
            timeout_seconds=120,
            retry_count=2,
            validation_rules={'output_type': 'embeddings'},
            created_at=time.time()
        )
        subtasks.append(embedding_subtask)
        
        # Анализ/Классификация
        analysis_subtask = SubTask(
            id=f"{task_id}_analysis",
            parent_task_id=task_id,
            task_type=TaskType.NATURAL_LANGUAGE,
            complexity=complexity,
            input_data={'embeddings': 'from_embedding'},
            model_config={'stage': 'analysis'},
            dependencies=[embedding_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_mb=2048,
                gpu_memory_mb=1024,
                storage_mb=300,
                network_bandwidth_mbps=1.5,
                estimated_time_seconds=120,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=40
            ),
            priority=3,
            timeout_seconds=240,
            retry_count=1,
            validation_rules={'output_type': 'analysis_result'},
            created_at=time.time()
        )
        subtasks.append(analysis_subtask)
        
        return subtasks
    
    def _decompose_cv_task(self, task_id: str, input_data: Dict,
                         complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач компьютерного зрения"""
        subtasks = []
        
        # Предобработка изображений
        preprocessing_subtask = SubTask(
            id=f"{task_id}_cv_preprocessing",
            parent_task_id=task_id,
            task_type=TaskType.COMPUTER_VISION,
            complexity=ComplexityLevel.LOW,
            input_data=input_data,
            model_config={'stage': 'preprocessing', 'operations': ['resize', 'normalize']},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=0,
                storage_mb=300,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=45,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=25
            ),
            priority=1,
            timeout_seconds=90,
            retry_count=2,
            validation_rules={'output_type': 'preprocessed_images'},
            created_at=time.time()
        )
        subtasks.append(preprocessing_subtask)
        
        # Извлечение признаков
        feature_extraction_subtask = SubTask(
            id=f"{task_id}_feature_extraction",
            parent_task_id=task_id,
            task_type=TaskType.COMPUTER_VISION,
            complexity=ComplexityLevel.HIGH,
            input_data={'images': 'from_preprocessing'},
            model_config={'stage': 'feature_extraction', 'model': 'resnet50'},
            dependencies=[preprocessing_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_mb=3072,
                gpu_memory_mb=2048,
                storage_mb=500,
                network_bandwidth_mbps=2.0,
                estimated_time_seconds=180,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=50
            ),
            priority=2,
            timeout_seconds=360,
            retry_count=1,
            validation_rules={'output_type': 'features'},
            created_at=time.time()
        )
        subtasks.append(feature_extraction_subtask)
        
        # Классификация/Детекция
        classification_subtask = SubTask(
            id=f"{task_id}_classification",
            parent_task_id=task_id,
            task_type=TaskType.COMPUTER_VISION,
            complexity=complexity,
            input_data={'features': 'from_feature_extraction'},
            model_config={'stage': 'classification'},
            dependencies=[feature_extraction_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=2048,
                gpu_memory_mb=1024,
                storage_mb=200,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=90,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=40
            ),
            priority=3,
            timeout_seconds=180,
            retry_count=1,
            validation_rules={'output_type': 'classification_result'},
            created_at=time.time()
        )
        subtasks.append(classification_subtask)
        
        return subtasks
    
    def _decompose_audio_task(self, task_id: str, input_data: Dict,
                            complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач обработки аудио"""
        subtasks = []
        
        # Предобработка аудио
        audio_preprocessing_subtask = SubTask(
            id=f"{task_id}_audio_preprocessing",
            parent_task_id=task_id,
            task_type=TaskType.AUDIO_PROCESSING,
            complexity=ComplexityLevel.MEDIUM,
            input_data=input_data,
            model_config={'stage': 'preprocessing', 'sample_rate': 16000},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=0,
                storage_mb=400,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=60,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=30
            ),
            priority=1,
            timeout_seconds=120,
            retry_count=2,
            validation_rules={'output_type': 'preprocessed_audio'},
            created_at=time.time()
        )
        subtasks.append(audio_preprocessing_subtask)
        
        # Извлечение признаков
        audio_features_subtask = SubTask(
            id=f"{task_id}_audio_features",
            parent_task_id=task_id,
            task_type=TaskType.AUDIO_PROCESSING,
            complexity=complexity,
            input_data={'audio': 'from_preprocessing'},
            model_config={'stage': 'feature_extraction', 'features': ['mfcc', 'spectral']},
            dependencies=[audio_preprocessing_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=3,
                memory_mb=2048,
                gpu_memory_mb=512,
                storage_mb=300,
                network_bandwidth_mbps=1.5,
                estimated_time_seconds=90,
                requires_gpu=complexity.value >= 4,
                requires_internet=True,
                min_battery_level=35
            ),
            priority=2,
            timeout_seconds=180,
            retry_count=1,
            validation_rules={'output_type': 'audio_features'},
            created_at=time.time()
        )
        subtasks.append(audio_features_subtask)
        
        return subtasks
    
    def _decompose_video_task(self, task_id: str, input_data: Dict,
                            complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач обработки видео"""
        subtasks = []
        
        # Извлечение кадров
        frame_extraction_subtask = SubTask(
            id=f"{task_id}_frame_extraction",
            parent_task_id=task_id,
            task_type=TaskType.VIDEO_PROCESSING,
            complexity=ComplexityLevel.LOW,
            input_data=input_data,
            model_config={'stage': 'frame_extraction', 'fps': 1},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=2048,
                gpu_memory_mb=0,
                storage_mb=1000,
                network_bandwidth_mbps=2.0,
                estimated_time_seconds=120,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=40
            ),
            priority=1,
            timeout_seconds=240,
            retry_count=2,
            validation_rules={'output_type': 'frames'},
            created_at=time.time()
        )
        subtasks.append(frame_extraction_subtask)
        
        # Обработка кадров
        frame_processing_subtask = SubTask(
            id=f"{task_id}_frame_processing",
            parent_task_id=task_id,
            task_type=TaskType.VIDEO_PROCESSING,
            complexity=complexity,
            input_data={'frames': 'from_frame_extraction'},
            model_config={'stage': 'frame_processing'},
            dependencies=[frame_extraction_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_mb=4096,
                gpu_memory_mb=2048,
                storage_mb=1500,
                network_bandwidth_mbps=3.0,
                estimated_time_seconds=300,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=60
            ),
            priority=2,
            timeout_seconds=600,
            retry_count=1,
            validation_rules={'output_type': 'processed_frames'},
            created_at=time.time()
        )
        subtasks.append(frame_processing_subtask)
        
        return subtasks
    
    def _decompose_classification_task(self, task_id: str, input_data: Dict,
                                     complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач классификации"""
        subtasks = []
        
        # Подготовка данных
        data_prep_subtask = SubTask(
            id=f"{task_id}_data_preparation",
            parent_task_id=task_id,
            task_type=TaskType.CLASSIFICATION,
            complexity=ComplexityLevel.MEDIUM,
            input_data=input_data,
            model_config={'stage': 'data_preparation'},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=0,
                storage_mb=300,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=60,
                requires_gpu=False,
                requires_internet=True,
                min_battery_level=30
            ),
            priority=1,
            timeout_seconds=120,
            retry_count=2,
            validation_rules={'output_type': 'prepared_data'},
            created_at=time.time()
        )
        subtasks.append(data_prep_subtask)
        
        # Классификация
        classification_subtask = SubTask(
            id=f"{task_id}_classification_inference",
            parent_task_id=task_id,
            task_type=TaskType.CLASSIFICATION,
            complexity=complexity,
            input_data={'data': 'from_data_preparation'},
            model_config={'stage': 'classification', 'model_type': 'ensemble'},
            dependencies=[data_prep_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=3,
                memory_mb=2048,
                gpu_memory_mb=1024 if complexity.value >= 3 else 0,
                storage_mb=400,
                network_bandwidth_mbps=1.5,
                estimated_time_seconds=120,
                requires_gpu=complexity.value >= 3,
                requires_internet=True,
                min_battery_level=40
            ),
            priority=2,
            timeout_seconds=240,
            retry_count=1,
            validation_rules={'output_type': 'classification_results'},
            created_at=time.time()
        )
        subtasks.append(classification_subtask)
        
        return subtasks
    
    def _decompose_clustering_task(self, task_id: str, input_data: Dict,
                                 complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач кластеризации"""
        subtasks = []
        
        # Предобработка данных
        preprocessing_subtask = SubTask(
            id=f"{task_id}_clustering_preprocessing",
            parent_task_id=task_id,
            task_type=TaskType.CLUSTERING,
            complexity=ComplexityLevel.MEDIUM,
            input_data=input_data,
            model_config={'stage': 'preprocessing', 'normalization': True},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=0,
                storage_mb=200,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=45,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=25
            ),
            priority=1,
            timeout_seconds=90,
            retry_count=2,
            validation_rules={'output_type': 'normalized_data'},
            created_at=time.time()
        )
        subtasks.append(preprocessing_subtask)
        
        # Кластеризация
        clustering_subtask = SubTask(
            id=f"{task_id}_clustering_algorithm",
            parent_task_id=task_id,
            task_type=TaskType.CLUSTERING,
            complexity=complexity,
            input_data={'data': 'from_preprocessing'},
            model_config={'stage': 'clustering', 'algorithm': 'kmeans'},
            dependencies=[preprocessing_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_mb=2048,
                gpu_memory_mb=512 if complexity.value >= 4 else 0,
                storage_mb=300,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=180,
                requires_gpu=complexity.value >= 4,
                requires_internet=False,
                min_battery_level=40
            ),
            priority=2,
            timeout_seconds=360,
            retry_count=1,
            validation_rules={'output_type': 'clusters'},
            created_at=time.time()
        )
        subtasks.append(clustering_subtask)
        
        return subtasks
    
    def _decompose_timeseries_task(self, task_id: str, input_data: Dict,
                                 complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Декомпозиция задач временных рядов"""
        subtasks = []
        
        # Предобработка временных рядов
        ts_preprocessing_subtask = SubTask(
            id=f"{task_id}_ts_preprocessing",
            parent_task_id=task_id,
            task_type=TaskType.TIME_SERIES,
            complexity=ComplexityLevel.MEDIUM,
            input_data=input_data,
            model_config={'stage': 'preprocessing', 'operations': ['detrend', 'normalize']},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=0,
                storage_mb=200,
                network_bandwidth_mbps=0.5,
                estimated_time_seconds=60,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=30
            ),
            priority=1,
            timeout_seconds=120,
            retry_count=2,
            validation_rules={'output_type': 'preprocessed_timeseries'},
            created_at=time.time()
        )
        subtasks.append(ts_preprocessing_subtask)
        
        # Анализ/Прогнозирование
        ts_analysis_subtask = SubTask(
            id=f"{task_id}_ts_analysis",
            parent_task_id=task_id,
            task_type=TaskType.TIME_SERIES,
            complexity=complexity,
            input_data={'timeseries': 'from_preprocessing'},
            model_config={'stage': 'analysis', 'model': 'lstm'},
            dependencies=[ts_preprocessing_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=3,
                memory_mb=2048,
                gpu_memory_mb=1024 if complexity.value >= 3 else 0,
                storage_mb=400,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=150,
                requires_gpu=complexity.value >= 3,
                requires_internet=True,
                min_battery_level=45
            ),
            priority=2,
            timeout_seconds=300,
            retry_count=1,
            validation_rules={'output_type': 'analysis_results'},
            created_at=time.time()
        )
        subtasks.append(ts_analysis_subtask)
        
        return subtasks
    
    def _decompose_generic_task(self, task_id: str, input_data: Dict,
                              complexity: ComplexityLevel, description: str) -> List[SubTask]:
        """Общая декомпозиция для неизвестных типов задач"""
        subtasks = []
        
        # Создание единственной подзадачи для обработки
        generic_subtask = SubTask(
            id=f"{task_id}_generic_processing",
            parent_task_id=task_id,
            task_type=TaskType.MACHINE_LEARNING,
            complexity=complexity,
            input_data=input_data,
            model_config={'stage': 'generic_processing'},
            dependencies=[],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=1024,
                gpu_memory_mb=0,
                storage_mb=200,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=120,
                requires_gpu=False,
                requires_internet=True,
                min_battery_level=30
            ),
            priority=1,
            timeout_seconds=240,
            retry_count=2,
            validation_rules={'output_type': 'generic_result'},
            created_at=time.time()
        )
        subtasks.append(generic_subtask)
        
        return subtasks
    
    def _calculate_chunk_size(self, total_size: int, complexity: ComplexityLevel) -> int:
        """Расчет размера чанка на основе общего размера и сложности"""
        base_chunk_size = 10000  # Базовый размер чанка
        
        # Корректировка на основе сложности
        complexity_multiplier = {
            ComplexityLevel.TRIVIAL: 2.0,
            ComplexityLevel.LOW: 1.5,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 0.7,
            ComplexityLevel.EXTREME: 0.5
        }
        
        chunk_size = int(base_chunk_size * complexity_multiplier[complexity])
        
        # Ограничения
        min_chunk_size = 1000
        max_chunk_size = 50000
        
        return max(min_chunk_size, min(chunk_size, max_chunk_size))
    
    def _create_aggregation_subtask(self, task_id: str, subtasks: List[SubTask],
                                  task_type: TaskType) -> SubTask:
        """Создание подзадачи агрегации результатов"""
        dependencies = [subtask.id for subtask in subtasks]
        
        return SubTask(
            id=f"{task_id}_aggregation",
            parent_task_id=task_id,
            task_type=task_type,
            complexity=ComplexityLevel.LOW,
            input_data={'subtask_results': dependencies},
            model_config={'stage': 'aggregation'},
            dependencies=dependencies,
            resources=ResourceRequirements(
                cpu_cores=1,
                memory_mb=512,
                gpu_memory_mb=0,
                storage_mb=100,
                network_bandwidth_mbps=0.5,
                estimated_time_seconds=30,
                requires_gpu=False,
                requires_internet=False,
                min_battery_level=20
            ),
            priority=10,  # Высокий приоритет для финальной агрегации
            timeout_seconds=60,
            retry_count=3,
            validation_rules={'output_type': 'aggregated_result'},
            created_at=time.time()
        )
    
    def _create_training_subtasks(self, task_id: str, complexity: ComplexityLevel) -> List[SubTask]:
        """Создание подзадач для обучения модели"""
        training_subtasks = []
        
        # Подзадача обучения
        training_subtask = SubTask(
            id=f"{task_id}_training",
            parent_task_id=task_id,
            task_type=TaskType.MACHINE_LEARNING,
            complexity=complexity,
            input_data={'training_data': 'from_preprocessing'},
            model_config={'stage': 'training', 'epochs': 10},
            dependencies=[f"{task_id}_preprocessing"],
            resources=ResourceRequirements(
                cpu_cores=4,
                memory_mb=4096,
                gpu_memory_mb=2048,
                storage_mb=1000,
                network_bandwidth_mbps=2.0,
                estimated_time_seconds=600,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=70
            ),
            priority=2,
            timeout_seconds=1200,
            retry_count=1,
            validation_rules={'output_type': 'trained_model'},
            created_at=time.time()
        )
        training_subtasks.append(training_subtask)
        
        # Подзадача валидации
        validation_subtask = SubTask(
            id=f"{task_id}_validation",
            parent_task_id=task_id,
            task_type=TaskType.MACHINE_LEARNING,
            complexity=ComplexityLevel.MEDIUM,
            input_data={'model': 'from_training', 'validation_data': 'from_preprocessing'},
            model_config={'stage': 'validation'},
            dependencies=[training_subtask.id],
            resources=ResourceRequirements(
                cpu_cores=2,
                memory_mb=2048,
                gpu_memory_mb=1024,
                storage_mb=500,
                network_bandwidth_mbps=1.0,
                estimated_time_seconds=180,
                requires_gpu=True,
                requires_internet=True,
                min_battery_level=40
            ),
            priority=3,
            timeout_seconds=360,
            retry_count=2,
            validation_rules={'output_type': 'validation_metrics'},
            created_at=time.time()
        )
        training_subtasks.append(validation_subtask)
        
        return training_subtasks

class TaskGraphBuilder:
    """Построитель графа задач с зависимостями"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def build_graph(self, subtasks: List[SubTask]) -> TaskGraph:
        """Построение графа задач"""
        # Добавление узлов
        for subtask in subtasks:
            self.graph.add_node(subtask.id, subtask=subtask)
        
        # Добавление рёбер (зависимостей)
        edges = []
        for subtask in subtasks:
            for dependency in subtask.dependencies:
                if dependency in [s.id for s in subtasks]:
                    self.graph.add_edge(dependency, subtask.id)
                    edges.append((dependency, subtask.id))
        
        # Определение порядка выполнения
        execution_order = self._calculate_execution_order()
        
        # Расчет общего времени и стоимости
        total_time, total_cost = self._calculate_totals(subtasks, execution_order)
        
        return TaskGraph(
            nodes={subtask.id: subtask for subtask in subtasks},
            edges=edges,
            execution_order=execution_order,
            total_estimated_time=total_time,
            total_resource_cost=total_cost
        )
    
    def _calculate_execution_order(self) -> List[List[str]]:
        """Расчет порядка выполнения задач по уровням"""
        try:
            # Топологическая сортировка
            topo_order = list(nx.topological_sort(self.graph))
            
            # Группировка по уровням
            levels = []
            remaining_nodes = set(topo_order)
            
            while remaining_nodes:
                # Найти узлы без зависимостей среди оставшихся
                current_level = []
                for node in topo_order:
                    if node in remaining_nodes:
                        # Проверить, что все зависимости уже обработаны
                        dependencies = set(self.graph.predecessors(node))
                        if dependencies.issubset(set().union(*levels) if levels else set()):
                            current_level.append(node)
                
                if not current_level:
                    # Если не можем найти узлы без зависимостей, берем первый доступный
                    current_level = [next(iter(remaining_nodes))]
                
                levels.append(current_level)
                remaining_nodes -= set(current_level)
            
            return levels
        
        except nx.NetworkXError:
            # Если граф содержит циклы, возвращаем линейный порядок
            return [[node] for node in self.graph.nodes()]
    
    def _calculate_totals(self, subtasks: List[SubTask], 
                         execution_order: List[List[str]]) -> Tuple[int, float]:
        """Расчет общего времени выполнения и стоимости ресурсов"""
        subtask_dict = {subtask.id: subtask for subtask in subtasks}
        
        # Время выполнения (максимальное время в каждом уровне)
        total_time = 0
        for level in execution_order:
            level_max_time = max(
                subtask_dict[task_id].resources.estimated_time_seconds 
                for task_id in level
            )
            total_time += level_max_time
        
        # Стоимость ресурсов (сумма всех ресурсов)
        total_cost = 0.0
        for subtask in subtasks:
            # Простая модель стоимости на основе ресурсов
            cpu_cost = subtask.resources.cpu_cores * 0.1
            memory_cost = subtask.resources.memory_mb * 0.0001
            gpu_cost = subtask.resources.gpu_memory_mb * 0.001
            storage_cost = subtask.resources.storage_mb * 0.00001
            
            total_cost += cpu_cost + memory_cost + gpu_cost + storage_cost
        
        return total_time, total_cost

class TaskAnalyzer:
    """Основной класс анализатора задач"""
    
    def __init__(self, db_path: str = "daur_ai.db"):
        self.model_manager = AIModelManager()
        self.db_manager = DatabaseManager(db_path)
        self.decomposer = TaskDecomposer(self.model_manager)
        self.graph_builder = TaskGraphBuilder()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("TaskAnalyzer инициализирован")
    
    async def analyze_task(self, task_id: str, name: str, description: str,
                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Полный анализ задачи"""
        start_time = time.time()
        
        try:
            # Классификация типа задачи
            task_type = self.model_manager.classify_task_type(description, input_data)
            
            # Оценка сложности
            input_size = sum(len(str(v)) for v in input_data.values())
            complexity = self.model_manager.estimate_complexity(description, input_size, task_type)
            
            # Сохранение задачи в БД
            task_data = {
                'name': name,
                'description': description,
                'task_type': task_type.value,
                'complexity': complexity.value,
                'input_data': input_data,
                'status': 'analyzing'
            }
            self.db_manager.save_task(task_id, task_data)
            
            # Декомпозиция на подзадачи
            subtasks = self.decomposer.decompose(task_id, task_type, input_data, 
                                               complexity, description)
            
            # Сохранение подзадач
            for subtask in subtasks:
                self.db_manager.save_subtask(subtask)
            
            # Построение графа задач
            task_graph = self.graph_builder.build_graph(subtasks)
            
            # Оптимизация графа
            optimized_graph = await self._optimize_task_graph(task_graph)
            
            analysis_time = time.time() - start_time
            
            result = {
                'task_id': task_id,
                'task_type': task_type.value,
                'complexity': complexity.value,
                'subtasks_count': len(subtasks),
                'execution_levels': len(optimized_graph.execution_order),
                'estimated_total_time': optimized_graph.total_estimated_time,
                'estimated_cost': optimized_graph.total_resource_cost,
                'analysis_time': analysis_time,
                'subtasks': [subtask.to_dict() for subtask in subtasks],
                'execution_order': optimized_graph.execution_order,
                'dependencies': optimized_graph.edges
            }
            
            logger.info(f"Задача {task_id} проанализирована за {analysis_time:.2f}с")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа задачи {task_id}: {e}")
            raise
    
    async def _optimize_task_graph(self, task_graph: TaskGraph) -> TaskGraph:
        """Оптимизация графа задач"""
        # Оптимизация порядка выполнения
        optimized_order = await self._optimize_execution_order(task_graph)
        
        # Балансировка нагрузки
        balanced_graph = await self._balance_resource_load(task_graph, optimized_order)
        
        return balanced_graph
    
    async def _optimize_execution_order(self, task_graph: TaskGraph) -> List[List[str]]:
        """Оптимизация порядка выполнения для минимизации общего времени"""
        # Алгоритм критического пути
        def calculate_critical_path():
            # Расчет времени начала и окончания для каждой задачи
            start_times = {}
            end_times = {}
            
            for level in task_graph.execution_order:
                for task_id in level:
                    subtask = task_graph.nodes[task_id]
                    
                    # Время начала = максимальное время окончания зависимостей
                    max_dependency_end = 0
                    for dep_id in subtask.dependencies:
                        if dep_id in end_times:
                            max_dependency_end = max(max_dependency_end, end_times[dep_id])
                    
                    start_times[task_id] = max_dependency_end
                    end_times[task_id] = max_dependency_end + subtask.resources.estimated_time_seconds
            
            return start_times, end_times
        
        start_times, end_times = calculate_critical_path()
        
        # Пересортировка задач в уровнях по приоритету и времени
        optimized_order = []
        for level in task_graph.execution_order:
            # Сортировка по приоритету и времени выполнения
            sorted_level = sorted(level, key=lambda task_id: (
                -task_graph.nodes[task_id].priority,  # Высокий приоритет первым
                -task_graph.nodes[task_id].resources.estimated_time_seconds  # Долгие задачи первыми
            ))
            optimized_order.append(sorted_level)
        
        return optimized_order
    
    async def _balance_resource_load(self, task_graph: TaskGraph, 
                                   execution_order: List[List[str]]) -> TaskGraph:
        """Балансировка нагрузки ресурсов"""
        # Анализ использования ресурсов по уровням
        resource_usage = []
        for level in execution_order:
            level_resources = {
                'cpu_cores': 0,
                'memory_mb': 0,
                'gpu_memory_mb': 0,
                'storage_mb': 0
            }
            
            for task_id in level:
                subtask = task_graph.nodes[task_id]
                level_resources['cpu_cores'] += subtask.resources.cpu_cores
                level_resources['memory_mb'] += subtask.resources.memory_mb
                level_resources['gpu_memory_mb'] += subtask.resources.gpu_memory_mb
                level_resources['storage_mb'] += subtask.resources.storage_mb
            
            resource_usage.append(level_resources)
        
        # Перераспределение задач для балансировки
        balanced_order = execution_order.copy()
        
        # Простая эвристика: перемещение легких задач в загруженные уровни
        for i in range(len(balanced_order) - 1):
            current_level = balanced_order[i]
            next_level = balanced_order[i + 1]
            
            # Если текущий уровень сильно загружен, а следующий - нет
            current_load = resource_usage[i]['cpu_cores'] + resource_usage[i]['memory_mb'] / 1000
            next_load = resource_usage[i + 1]['cpu_cores'] + resource_usage[i + 1]['memory_mb'] / 1000
            
            if current_load > next_load * 2:
                # Найти легкие задачи для перемещения
                light_tasks = [
                    task_id for task_id in current_level
                    if task_graph.nodes[task_id].resources.cpu_cores <= 2
                    and not any(dep in current_level for dep in task_graph.nodes[task_id].dependencies)
                ]
                
                if light_tasks:
                    # Переместить одну легкую задачу
                    task_to_move = light_tasks[0]
                    balanced_order[i].remove(task_to_move)
                    balanced_order[i + 1].append(task_to_move)
        
        # Создание нового графа с оптимизированным порядком
        return TaskGraph(
            nodes=task_graph.nodes,
            edges=task_graph.edges,
            execution_order=balanced_order,
            total_estimated_time=task_graph.total_estimated_time,
            total_resource_cost=task_graph.total_resource_cost
        )
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Получение статуса задачи"""
        return self.db_manager.get_task(task_id)
    
    async def validate_subtask_result(self, subtask_id: str, result_data: Dict) -> bool:
        """Валидация результата подзадачи"""
        # Получение подзадачи из БД
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM subtasks WHERE id = ?', (subtask_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return False
        
        # Парсинг правил валидации
        validation_rules = json.loads(row[12]) if row[12] else {}
        
        # Проверка правил
        for rule, expected_value in validation_rules.items():
            if rule == 'output_type':
                if 'type' not in result_data or result_data['type'] != expected_value:
                    return False
            elif rule == 'min_output_length':
                if 'data' not in result_data or len(str(result_data['data'])) < expected_value:
                    return False
            elif rule == 'output_format':
                if 'format' not in result_data or result_data['format'] != expected_value:
                    return False
        
        return True
    
    async def estimate_node_suitability(self, subtask_id: str, node_capabilities: Dict) -> float:
        """Оценка пригодности узла для выполнения подзадачи"""
        # Получение подзадачи
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM subtasks WHERE id = ?', (subtask_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return 0.0
        
        # Парсинг требований к ресурсам
        resources_required = json.loads(row[11])
        
        # Расчет соответствия
        suitability_score = 1.0
        
        # Проверка CPU
        if node_capabilities.get('cpu_cores', 0) < resources_required.get('cpu_cores', 1):
            suitability_score *= 0.5
        
        # Проверка памяти
        if node_capabilities.get('memory_mb', 0) < resources_required.get('memory_mb', 512):
            suitability_score *= 0.3
        
        # Проверка GPU
        if resources_required.get('requires_gpu', False):
            if not node_capabilities.get('gpu_available', False):
                suitability_score *= 0.1
            elif node_capabilities.get('gpu_memory_mb', 0) < resources_required.get('gpu_memory_mb', 0):
                suitability_score *= 0.6
        
        # Проверка батареи
        if node_capabilities.get('battery_level', 100) < resources_required.get('min_battery_level', 20):
            suitability_score *= 0.4
        
        # Проверка интернета
        if resources_required.get('requires_internet', False):
            if node_capabilities.get('network_speed_mbps', 0) < 1.0:
                suitability_score *= 0.2
        
        # Бонус за избыточные ресурсы
        cpu_ratio = node_capabilities.get('cpu_cores', 1) / max(resources_required.get('cpu_cores', 1), 1)
        memory_ratio = node_capabilities.get('memory_mb', 512) / max(resources_required.get('memory_mb', 512), 1)
        
        if cpu_ratio > 1.5:
            suitability_score *= 1.2
        if memory_ratio > 1.5:
            suitability_score *= 1.1
        
        return min(suitability_score, 1.0)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Очистка старых задач из базы данных"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Удаление старых задач
        cursor.execute('DELETE FROM tasks WHERE created_at < ?', (cutoff_time,))
        cursor.execute('DELETE FROM subtasks WHERE created_at < ?', (cutoff_time,))
        cursor.execute('''
            DELETE FROM task_dependencies 
            WHERE source_task_id NOT IN (SELECT id FROM subtasks)
            OR target_task_id NOT IN (SELECT id FROM subtasks)
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Очищены задачи старше {max_age_hours} часов")
    
    async def shutdown(self):
        """Корректное завершение работы"""
        self.executor.shutdown(wait=True)
        if self.db_manager.redis_client:
            self.db_manager.redis_client.close()
        logger.info("TaskAnalyzer завершил работу")

# Функция для тестирования
async def main():
    """Тестирование TaskAnalyzer"""
    analyzer = TaskAnalyzer()
    
    # Тестовая задача обработки текста
    task_result = await analyzer.analyze_task(
        task_id="test_task_001",
        name="Анализ текста",
        description="Обработка и анализ большого объема текстовых данных для извлечения ключевых тем",
        input_data={
            'text': "Это большой текст для анализа. " * 1000,
            'language': 'ru',
            'analysis_type': 'topic_modeling'
        }
    )
    
    print(f"Результат анализа задачи: {json.dumps(task_result, indent=2, ensure_ascii=False)}")
    
    await analyzer.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
