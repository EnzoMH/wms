#!/usr/bin/env python3
"""
시스템 라우터
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from datetime import datetime
from pathlib import Path
import logging

from app.system.dto import SystemStatus, ActivityLog, HealthCheck
from app.system.service import SystemService

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/system",
    tags=["system"],
    responses={404: {"description": "Not found"}},
)

def get_system_service():
    """시스템 서비스 의존성 주입"""
    return SystemService()


@router.get("/status", response_model=SystemStatus)
async def get_system_status(
    service: SystemService = Depends(get_system_service)
):
    """시스템 리소스 현황"""
    try:
        stats = service.get_system_stats()
        return SystemStatus(**stats)
    except Exception as e:
        logger.error(f"시스템 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activities", response_model=List[ActivityLog])
async def get_activities(
    limit: int = 10,
    service: SystemService = Depends(get_system_service)
):
    """최근 활동 로그"""
    try:
        activities = service.get_recent_activities(limit)
        return [ActivityLog(**activity) for activity in activities]
    except Exception as e:
        logger.error(f"활동 로그 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """헬스체크"""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        services={
            "vectordb": Path("2_vecdb/faiss_storage/warehouse_automation_knowledge.index").exists()
        }
    )

