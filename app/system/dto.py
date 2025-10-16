#!/usr/bin/env python3
"""
시스템 DTO
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class SystemStatus(BaseModel):
    """시스템 상태"""
    gpu_memory_used: float = Field(0, description="GPU 메모리 사용량 (GB)")
    gpu_memory_total: float = Field(0, description="GPU 메모리 총량 (GB)")
    gpu_memory_percent: float = Field(0, description="GPU 메모리 사용률 (%)")
    
    ram_used: float = Field(0, description="RAM 사용량 (GB)")
    ram_total: float = Field(0, description="RAM 총량 (GB)")
    ram_percent: float = Field(0, description="RAM 사용률 (%)")
    
    cpu_percent: float = Field(0, description="CPU 사용률 (%)")
    disk_used: float = Field(0, description="디스크 사용량 (GB)")
    disk_total: float = Field(0, description="디스크 총량 (GB)")


class ActivityLog(BaseModel):
    """활동 로그"""
    message: str = Field(..., description="활동 메시지")
    timestamp: datetime = Field(default_factory=datetime.now)
    type: str = Field("info", description="로그 타입 (info/success/warning/error)")
    details: Optional[Dict[str, Any]] = Field(default={}, description="추가 정보")


class HealthCheck(BaseModel):
    """헬스체크 응답"""
    status: str = Field("healthy", description="서버 상태")
    version: str = Field("1.0.0", description="API 버전")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = Field(default={}, description="서비스 상태")

