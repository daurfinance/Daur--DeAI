"""
api.py — FastAPI для клиентского взаимодействия с Daur-AI
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from orchestrator.orchestrator import Orchestrator
from ai_core.blockchain_client import BlockchainClient
from nodes_sdk.node_client import NodeClient
import uuid


from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)

class TaskRequest(BaseModel):
    target_result: str
    data: bytes
    reward: int

# Инициализация компонентов (пример, настройте под свою инфраструктуру)
blockchain = BlockchainClient(rpc_url="http://localhost:8545", contract_address="0x...", abi_path="/path/to/abi.json")
node_clients = [NodeClient(blockchain, "<PRIVATE_KEY>")]
orch = Orchestrator(node_clients)

# Хранилище задач (в реальном проекте — БД)
tasks = {}

@app.post("/tasks")
async def create_task(request: TaskRequest):
    task_id = str(uuid.uuid4())
    task_spec = {"target_result": request.target_result, "data": request.data, "complexity": 0.5}
    agg_hash = orch.process_task(task_spec, request.data, request.reward)
    tasks[task_id] = {"status": "done", "agg_hash": agg_hash}
    return {"task_id": task_id, "agg_hash": agg_hash}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]
