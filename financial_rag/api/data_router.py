"""
Data layer introspection endpoint — 数据层可观测（横切关注点，独立于 KB CRUD）

Endpoints:
- GET /api/data-layer  → 存储归属清单 + 数据流图 + 实时状态 + 一致性体检

严格只读：不依赖 app_state._state，不触发初始化/重建，启动前也可用。
"""
from fastapi import APIRouter

from financial_rag.services.data_layer import get_data_layer_map

router = APIRouter()


@router.get("/api/data-layer")
def data_layer_map():
    """数据层全景：哪个库归谁管、每个操作落进哪几个库、各库当下状态。"""
    return get_data_layer_map()