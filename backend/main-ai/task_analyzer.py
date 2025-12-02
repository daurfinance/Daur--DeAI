"""
MainAI Task Analyzer - Анализ и декомпозиция задач для Daur-AI
Автор: Дауиржан Нуридинулы
"""

import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
import numpy as np
from datetime import datetime, timedelta


class TaskType(Enum):
    """Типы задач, поддерживаемые системой"""
    TEXT_PROCESSING = "text_processing"
    IMAGE_CLASSIFICATION = "image_classification"
    VIDEO_ANALYSIS = "video_analysis"
    MACHINE_LEARNING = "machine_learning"
    DATA_ANALYSIS = "data_analysis"
    NATURAL_LANGUAGE = "natural_language"
    COMPUTER_VISION = "computer_vision"
    AUDIO_PROCESSING = "audio_processing"


class Priority(Enum):
    """Приоритеты задач"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceRequirements:
    """Требования к ресурсам для выполнения подзадачи"""
    cpu_cores: int
    memory_mb: int
    gpu_required: bool
    storage_mb: int
    network_bandwidth_mbps: float
    estimated_time_seconds: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubTask:
    """Подзадача в графе выполнения"""
    id: str
    name: str
    description: str
    task_type: TaskType
    input_data_cid: Optional[str]
    model_cid: Optional[str]
    dependencies: List[str]
    resources: ResourceRequirements
    priority: Priority
    reward_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['task_type'] = self.task_type.value
        data['priority'] = self.priority.value
        return data


@dataclass
class TaskGraph:
    """Граф задач (DAG) для выполнения"""
    task_id: str
    subtasks: List[SubTask]
    execution_order: List[List[str]]  # Уровни выполнения
    total_reward: float
    estimated_completion_time: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'subtasks': [st.to_dict() for st in self.subtasks],
            'execution_order': self.execution_order,
            'total_reward': self.total_reward,
            'estimated_completion_time': self.estimated_completion_time
        }


class TaskAnalyzer:
    """Основной класс для анализа и декомпозиции задач"""
    
    def __init__(self):
        self.supported_types = {
            'text': TaskType.TEXT_PROCESSING,
            'image': TaskType.IMAGE_CLASSIFICATION,
            'video': TaskType.VIDEO_ANALYSIS,
            'ml': TaskType.MACHINE_LEARNING,
            'data': TaskType.DATA_ANALYSIS,
            'nlp': TaskType.NATURAL_LANGUAGE,
            'cv': TaskType.COMPUTER_VISION,
            'audio': TaskType.AUDIO_PROCESSING
        }
        
        # Базовые требования к ресурсам для разных типов задач
        self.base_requirements = {
            TaskType.TEXT_PROCESSING: ResourceRequirements(1, 512, False, 100, 1.0, 30),
            TaskType.IMAGE_CLASSIFICATION: ResourceRequirements(2, 1024, True, 500, 2.0, 60),
            TaskType.VIDEO_ANALYSIS: ResourceRequirements(4, 2048, True, 1000, 5.0, 300),
            TaskType.MACHINE_LEARNING: ResourceRequirements(2, 1024, True, 200, 1.0, 120),
            TaskType.DATA_ANALYSIS: ResourceRequirements(1, 512, False, 50, 0.5, 45),
            TaskType.NATURAL_LANGUAGE: ResourceRequirements(2, 1024, False, 200, 1.0, 90),
            TaskType.COMPUTER_VISION: ResourceRequirements(4, 2048, True, 800, 3.0, 180),
            TaskType.AUDIO_PROCESSING: ResourceRequirements(2, 1024, False, 300, 2.0, 120)
        }
    
    def analyze_task(self, task_description: str, input_data_size: int, 
                    reward: float, priority: str = "normal") -> TaskType:
        """
        Анализирует описание задачи и определяет её тип
        
        Args:
            task_description: Описание задачи
            input_data_size: Размер входных данных в байтах
            reward: Вознаграждение за выполнение
            priority: Приоритет задачи
            
        Returns:
            TaskType: Определенный тип задачи
        """
        description_lower = task_description.lower()
        
        # Простая эвристика для определения типа задачи
        if any(word in description_lower for word in ['text', 'translate', 'sentiment', 'language']):
            if any(word in description_lower for word in ['nlp', 'natural', 'understanding']):
                return TaskType.NATURAL_LANGUAGE
            return TaskType.TEXT_PROCESSING
        
        elif any(word in description_lower for word in ['image', 'photo', 'picture', 'classify']):
            if any(word in description_lower for word in ['detect', 'recognition', 'vision']):
                return TaskType.COMPUTER_VISION
            return TaskType.IMAGE_CLASSIFICATION
        
        elif any(word in description_lower for word in ['video', 'movie', 'clip']):
            return TaskType.VIDEO_ANALYSIS
        
        elif any(word in description_lower for word in ['audio', 'sound', 'speech', 'music']):
            return TaskType.AUDIO_PROCESSING
        
        elif any(word in description_lower for word in ['train', 'model', 'learning', 'neural']):
            return TaskType.MACHINE_LEARNING
        
        elif any(word in description_lower for word in ['data', 'analyze', 'statistics', 'csv']):
            return TaskType.DATA_ANALYSIS
        
        # По умолчанию - обработка данных
        return TaskType.DATA_ANALYSIS
    
    def decompose_task(self, task_id: str, task_description: str, 
                      input_data_cid: str, model_cid: Optional[str],
                      reward: float, priority: str = "normal") -> TaskGraph:
        """
        Декомпозирует задачу на подзадачи и создает граф выполнения
        
        Args:
            task_id: Уникальный идентификатор задачи
            task_description: Описание задачи
            input_data_cid: CID входных данных в IPFS
            model_cid: CID модели ИИ в IPFS (опционально)
            reward: Общее вознаграждение за задачу
            priority: Приоритет задачи
            
        Returns:
            TaskGraph: Граф подзадач для выполнения
        """
        # Определяем тип задачи
        task_type = self.analyze_task(task_description, 0, reward, priority)
        priority_enum = Priority[priority.upper()]
        
        # Создаем подзадачи в зависимости от типа
        subtasks = self._create_subtasks(task_id, task_type, task_description,
                                       input_data_cid, model_cid, reward, priority_enum)
        
        # Строим граф зависимостей
        execution_order = self._build_execution_order(subtasks)
        
        # Оцениваем общее время выполнения
        total_time = self._estimate_total_time(subtasks, execution_order)
        
        return TaskGraph(
            task_id=task_id,
            subtasks=subtasks,
            execution_order=execution_order,
            total_reward=reward,
            estimated_completion_time=total_time
        )
    
    def _create_subtasks(self, task_id: str, task_type: TaskType, 
                        description: str, input_cid: str, model_cid: Optional[str],
                        reward: float, priority: Priority) -> List[SubTask]:
        """Создает список подзадач для конкретного типа задачи"""
        
        subtasks = []
        base_req = self.base_requirements[task_type]
        
        if task_type == TaskType.TEXT_PROCESSING:
            subtasks = self._create_text_processing_subtasks(
                task_id, description, input_cid, model_cid, reward, priority, base_req
            )
        elif task_type == TaskType.IMAGE_CLASSIFICATION:
            subtasks = self._create_image_classification_subtasks(
                task_id, description, input_cid, model_cid, reward, priority, base_req
            )
        elif task_type == TaskType.NATURAL_LANGUAGE:
            subtasks = self._create_nlp_subtasks(
                task_id, description, input_cid, model_cid, reward, priority, base_req
            )
        elif task_type == TaskType.MACHINE_LEARNING:
            subtasks = self._create_ml_training_subtasks(
                task_id, description, input_cid, model_cid, reward, priority, base_req
            )
        else:
            # Базовая декомпозиция для других типов
            subtasks = self._create_generic_subtasks(
                task_id, task_type, description, input_cid, model_cid, reward, priority, base_req
            )
        
        return subtasks
    
    def _create_text_processing_subtasks(self, task_id: str, description: str,
                                       input_cid: str, model_cid: Optional[str],
                                       reward: float, priority: Priority,
                                       base_req: ResourceRequirements) -> List[SubTask]:
        """Создает подзадачи для обработки текста"""
        
        subtasks = []
        
        # 1. Предобработка текста
        preprocess_id = f"{task_id}_preprocess"
        subtasks.append(SubTask(
            id=preprocess_id,
            name="Text Preprocessing",
            description="Tokenization, cleaning, and text normalization",
            task_type=TaskType.TEXT_PROCESSING,
            input_data_cid=input_cid,
            model_cid=None,
            dependencies=[],
            resources=ResourceRequirements(1, 256, False, 50, 0.5, 15),
            priority=priority,
            reward_weight=0.1
        ))
        
        # 2. Разбиение на чанки
        chunk_id = f"{task_id}_chunk"
        subtasks.append(SubTask(
            id=chunk_id,
            name="Text Chunking",
            description="Split text into processable chunks",
            task_type=TaskType.TEXT_PROCESSING,
            input_data_cid=None,  # Будет получен от предыдущей задачи
            model_cid=None,
            dependencies=[preprocess_id],
            resources=ResourceRequirements(1, 128, False, 25, 0.2, 10),
            priority=priority,
            reward_weight=0.05
        ))
        
        # 3. Параллельная обработка чанков (симулируем 4 чанка)
        chunk_tasks = []
        for i in range(4):
            process_id = f"{task_id}_process_{i}"
            chunk_tasks.append(process_id)
            subtasks.append(SubTask(
                id=process_id,
                name=f"Process Text Chunk {i+1}",
                description=f"Process text chunk {i+1} with AI model",
                task_type=TaskType.TEXT_PROCESSING,
                input_data_cid=None,
                model_cid=model_cid,
                dependencies=[chunk_id],
                resources=base_req,
                priority=priority,
                reward_weight=0.7 / 4  # Распределяем 70% награды между чанками
            ))
        
        # 4. Агрегация результатов
        aggregate_id = f"{task_id}_aggregate"
        subtasks.append(SubTask(
            id=aggregate_id,
            name="Aggregate Results",
            description="Combine processed chunks into final result",
            task_type=TaskType.TEXT_PROCESSING,
            input_data_cid=None,
            model_cid=None,
            dependencies=chunk_tasks,
            resources=ResourceRequirements(1, 512, False, 100, 1.0, 20),
            priority=priority,
            reward_weight=0.15
        ))
        
        return subtasks
    
    def _create_image_classification_subtasks(self, task_id: str, description: str,
                                            input_cid: str, model_cid: Optional[str],
                                            reward: float, priority: Priority,
                                            base_req: ResourceRequirements) -> List[SubTask]:
        """Создает подзадачи для классификации изображений"""
        
        subtasks = []
        
        # 1. Загрузка и валидация изображений
        load_id = f"{task_id}_load"
        subtasks.append(SubTask(
            id=load_id,
            name="Load and Validate Images",
            description="Load images from IPFS and validate format",
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_data_cid=input_cid,
            model_cid=None,
            dependencies=[],
            resources=ResourceRequirements(1, 512, False, 200, 2.0, 30),
            priority=priority,
            reward_weight=0.1
        ))
        
        # 2. Предобработка изображений
        preprocess_id = f"{task_id}_preprocess"
        subtasks.append(SubTask(
            id=preprocess_id,
            name="Image Preprocessing",
            description="Resize, normalize, and augment images",
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_data_cid=None,
            model_cid=None,
            dependencies=[load_id],
            resources=ResourceRequirements(2, 1024, True, 300, 1.0, 45),
            priority=priority,
            reward_weight=0.15
        ))
        
        # 3. Параллельная классификация батчей
        batch_tasks = []
        for i in range(3):  # 3 батча для параллельной обработки
            classify_id = f"{task_id}_classify_{i}"
            batch_tasks.append(classify_id)
            subtasks.append(SubTask(
                id=classify_id,
                name=f"Classify Batch {i+1}",
                description=f"Classify images in batch {i+1}",
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_data_cid=None,
                model_cid=model_cid,
                dependencies=[preprocess_id],
                resources=base_req,
                priority=priority,
                reward_weight=0.6 / 3  # 60% награды между батчами
            ))
        
        # 4. Постобработка и агрегация
        postprocess_id = f"{task_id}_postprocess"
        subtasks.append(SubTask(
            id=postprocess_id,
            name="Postprocess Results",
            description="Aggregate classification results and generate report",
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_data_cid=None,
            model_cid=None,
            dependencies=batch_tasks,
            resources=ResourceRequirements(1, 256, False, 100, 0.5, 15),
            priority=priority,
            reward_weight=0.15
        ))
        
        return subtasks
    
    def _create_nlp_subtasks(self, task_id: str, description: str,
                           input_cid: str, model_cid: Optional[str],
                           reward: float, priority: Priority,
                           base_req: ResourceRequirements) -> List[SubTask]:
        """Создает подзадачи для NLP задач"""
        
        subtasks = []
        
        # 1. Анализ и токенизация
        tokenize_id = f"{task_id}_tokenize"
        subtasks.append(SubTask(
            id=tokenize_id,
            name="Tokenization and Analysis",
            description="Tokenize text and perform linguistic analysis",
            task_type=TaskType.NATURAL_LANGUAGE,
            input_data_cid=input_cid,
            model_cid=None,
            dependencies=[],
            resources=ResourceRequirements(1, 512, False, 100, 1.0, 20),
            priority=priority,
            reward_weight=0.2
        ))
        
        # 2. Извлечение признаков
        features_id = f"{task_id}_features"
        subtasks.append(SubTask(
            id=features_id,
            name="Feature Extraction",
            description="Extract linguistic features and embeddings",
            task_type=TaskType.NATURAL_LANGUAGE,
            input_data_cid=None,
            model_cid=model_cid,
            dependencies=[tokenize_id],
            resources=base_req,
            priority=priority,
            reward_weight=0.4
        ))
        
        # 3. Семантический анализ
        semantic_id = f"{task_id}_semantic"
        subtasks.append(SubTask(
            id=semantic_id,
            name="Semantic Analysis",
            description="Perform semantic analysis and understanding",
            task_type=TaskType.NATURAL_LANGUAGE,
            input_data_cid=None,
            model_cid=model_cid,
            dependencies=[features_id],
            resources=base_req,
            priority=priority,
            reward_weight=0.3
        ))
        
        # 4. Генерация результата
        generate_id = f"{task_id}_generate"
        subtasks.append(SubTask(
            id=generate_id,
            name="Result Generation",
            description="Generate final NLP result",
            task_type=TaskType.NATURAL_LANGUAGE,
            input_data_cid=None,
            model_cid=model_cid,
            dependencies=[semantic_id],
            resources=ResourceRequirements(2, 1024, False, 200, 1.0, 30),
            priority=priority,
            reward_weight=0.1
        ))
        
        return subtasks
    
    def _create_ml_training_subtasks(self, task_id: str, description: str,
                                   input_cid: str, model_cid: Optional[str],
                                   reward: float, priority: Priority,
                                   base_req: ResourceRequirements) -> List[SubTask]:
        """Создает подзадачи для обучения ML моделей"""
        
        subtasks = []
        
        # 1. Подготовка данных
        data_prep_id = f"{task_id}_data_prep"
        subtasks.append(SubTask(
            id=data_prep_id,
            name="Data Preparation",
            description="Load, clean, and prepare training data",
            task_type=TaskType.MACHINE_LEARNING,
            input_data_cid=input_cid,
            model_cid=None,
            dependencies=[],
            resources=ResourceRequirements(2, 1024, False, 500, 2.0, 60),
            priority=priority,
            reward_weight=0.2
        ))
        
        # 2. Разделение данных
        split_id = f"{task_id}_split"
        subtasks.append(SubTask(
            id=split_id,
            name="Data Splitting",
            description="Split data into training/validation sets",
            task_type=TaskType.MACHINE_LEARNING,
            input_data_cid=None,
            model_cid=None,
            dependencies=[data_prep_id],
            resources=ResourceRequirements(1, 512, False, 100, 0.5, 15),
            priority=priority,
            reward_weight=0.05
        ))
        
        # 3. Параллельное обучение на разных узлах (федеративное обучение)
        training_tasks = []
        for i in range(5):  # 5 узлов для федеративного обучения
            train_id = f"{task_id}_train_{i}"
            training_tasks.append(train_id)
            subtasks.append(SubTask(
                id=train_id,
                name=f"Federated Training Node {i+1}",
                description=f"Train model on federated node {i+1}",
                task_type=TaskType.MACHINE_LEARNING,
                input_data_cid=None,
                model_cid=model_cid,
                dependencies=[split_id],
                resources=ResourceRequirements(4, 2048, True, 1000, 1.0, 300),
                priority=priority,
                reward_weight=0.6 / 5  # 60% награды между узлами обучения
            ))
        
        # 4. Агрегация моделей
        aggregate_id = f"{task_id}_aggregate"
        subtasks.append(SubTask(
            id=aggregate_id,
            name="Model Aggregation",
            description="Aggregate federated learning results",
            task_type=TaskType.MACHINE_LEARNING,
            input_data_cid=None,
            model_cid=None,
            dependencies=training_tasks,
            resources=ResourceRequirements(2, 1024, True, 500, 1.0, 45),
            priority=priority,
            reward_weight=0.1
        ))
        
        # 5. Валидация модели
        validate_id = f"{task_id}_validate"
        subtasks.append(SubTask(
            id=validate_id,
            name="Model Validation",
            description="Validate aggregated model performance",
            task_type=TaskType.MACHINE_LEARNING,
            input_data_cid=None,
            model_cid=None,
            dependencies=[aggregate_id],
            resources=ResourceRequirements(2, 1024, True, 200, 0.5, 30),
            priority=priority,
            reward_weight=0.05
        ))
        
        return subtasks
    
    def _create_generic_subtasks(self, task_id: str, task_type: TaskType,
                               description: str, input_cid: str, model_cid: Optional[str],
                               reward: float, priority: Priority,
                               base_req: ResourceRequirements) -> List[SubTask]:
        """Создает базовые подзадачи для неспециализированных типов"""
        
        subtasks = []
        
        # 1. Загрузка данных
        load_id = f"{task_id}_load"
        subtasks.append(SubTask(
            id=load_id,
            name="Load Data",
            description="Load input data from IPFS",
            task_type=task_type,
            input_data_cid=input_cid,
            model_cid=None,
            dependencies=[],
            resources=ResourceRequirements(1, 256, False, 100, 1.0, 20),
            priority=priority,
            reward_weight=0.1
        ))
        
        # 2. Обработка данных
        process_id = f"{task_id}_process"
        subtasks.append(SubTask(
            id=process_id,
            name="Process Data",
            description="Process data with AI model",
            task_type=task_type,
            input_data_cid=None,
            model_cid=model_cid,
            dependencies=[load_id],
            resources=base_req,
            priority=priority,
            reward_weight=0.8
        ))
        
        # 3. Сохранение результата
        save_id = f"{task_id}_save"
        subtasks.append(SubTask(
            id=save_id,
            name="Save Result",
            description="Save processed result to IPFS",
            task_type=task_type,
            input_data_cid=None,
            model_cid=None,
            dependencies=[process_id],
            resources=ResourceRequirements(1, 256, False, 50, 1.0, 15),
            priority=priority,
            reward_weight=0.1
        ))
        
        return subtasks
    
    def _build_execution_order(self, subtasks: List[SubTask]) -> List[List[str]]:
        """Строит порядок выполнения подзадач по уровням"""
        
        # Создаем граф зависимостей
        graph = nx.DiGraph()
        
        # Добавляем узлы
        for subtask in subtasks:
            graph.add_node(subtask.id)
        
        # Добавляем ребра (зависимости)
        for subtask in subtasks:
            for dep in subtask.dependencies:
                graph.add_edge(dep, subtask.id)
        
        # Проверяем на циклы
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Task graph contains cycles")
        
        # Строим топологическую сортировку по уровням
        levels = []
        remaining_nodes = set(graph.nodes())
        
        while remaining_nodes:
            # Находим узлы без входящих ребер (среди оставшихся)
            current_level = []
            for node in remaining_nodes:
                if all(pred not in remaining_nodes for pred in graph.predecessors(node)):
                    current_level.append(node)
            
            if not current_level:
                raise ValueError("Cannot resolve dependencies")
            
            levels.append(current_level)
            remaining_nodes -= set(current_level)
        
        return levels
    
    def _estimate_total_time(self, subtasks: List[SubTask], 
                           execution_order: List[List[str]]) -> int:
        """Оценивает общее время выполнения задачи"""
        
        subtask_dict = {st.id: st for st in subtasks}
        total_time = 0
        
        for level in execution_order:
            # Время уровня = максимальное время среди параллельных задач
            level_time = max(subtask_dict[task_id].resources.estimated_time_seconds 
                           for task_id in level)
            total_time += level_time
        
        return total_time
    
    def optimize_graph(self, task_graph: TaskGraph) -> TaskGraph:
        """Оптимизирует граф задач для лучшей производительности"""
        
        # Здесь можно реализовать различные оптимизации:
        # 1. Объединение мелких задач
        # 2. Разделение крупных задач
        # 3. Балансировка нагрузки
        # 4. Оптимизация использования ресурсов
        
        # Пока возвращаем исходный граф
        return task_graph
    
    def validate_graph(self, task_graph: TaskGraph) -> bool:
        """Проверяет корректность графа задач"""
        
        try:
            # Проверяем уникальность ID подзадач
            task_ids = [st.id for st in task_graph.subtasks]
            if len(task_ids) != len(set(task_ids)):
                return False
            
            # Проверяем корректность зависимостей
            for subtask in task_graph.subtasks:
                for dep in subtask.dependencies:
                    if dep not in task_ids:
                        return False
            
            # Проверяем корректность порядка выполнения
            all_tasks_in_order = set()
            for level in task_graph.execution_order:
                all_tasks_in_order.update(level)
            
            if all_tasks_in_order != set(task_ids):
                return False
            
            return True
            
        except Exception:
            return False


def main():
    """Пример использования TaskAnalyzer"""
    
    analyzer = TaskAnalyzer()
    
    # Пример задачи обработки текста
    task_graph = analyzer.decompose_task(
        task_id="task_001",
        task_description="Translate a large document from Russian to English using neural machine translation",
        input_data_cid="QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        model_cid="QmYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY",
        reward=100.0,
        priority="high"
    )
    
    # Проверяем корректность графа
    if analyzer.validate_graph(task_graph):
        print("Task graph is valid")
        print(f"Total subtasks: {len(task_graph.subtasks)}")
        print(f"Execution levels: {len(task_graph.execution_order)}")
        print(f"Estimated completion time: {task_graph.estimated_completion_time} seconds")
        
        # Выводим граф в JSON формате
        print("\nTask Graph JSON:")
        print(json.dumps(task_graph.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Task graph is invalid")


if __name__ == "__main__":
    main()
