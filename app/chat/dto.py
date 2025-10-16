#!/usr/bin/env python3
"""
채팅 DTO (Data Transfer Object)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """채팅 요청"""
    query: str = Field(..., description="사용자 질문", min_length=1)
    context_count: Optional[int] = Field(5, description="검색할 컨텍스트 수", ge=1, le=20)
    temperature: Optional[float] = Field(0.1, description="생성 온도", ge=0.0, le=2.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "AGV와 AMR의 차이점은?",
                "context_count": 5,
                "temperature": 0.1
            }
        }


class PerformanceMetricsDTO(BaseModel):
    """성능 메트릭"""
    inference_time_ms: float = Field(..., description="추론 시간 (밀리초)")
    latency_ms: float = Field(..., description="전체 지연 시간 (밀리초)")
    
    total_tokens: int = Field(..., description="총 토큰 수")
    input_tokens: int = Field(..., description="입력 토큰 수")
    output_tokens: int = Field(..., description="출력 토큰 수")
    tokens_per_second: float = Field(..., description="초당 토큰 수")
    
    memory_used_mb: float = Field(..., description="사용된 RAM (MB)")
    memory_percent: float = Field(..., description="RAM 사용률 (%)")
    gpu_memory_used_mb: Optional[float] = Field(None, description="GPU 메모리 (MB)")
    gpu_memory_percent: Optional[float] = Field(None, description="GPU 사용률 (%)")
    
    throughput_tokens_sec: float = Field(0.0, description="처리량 (토큰/초)")
    model_name: str = Field("unknown", description="모델 이름")
    device: str = Field("cpu", description="실행 디바이스")


class SourceDocument(BaseModel):
    """소스 문서"""
    content: str = Field(..., description="문서 내용")
    score: float = Field(..., description="유사도 점수", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="메타데이터")


class ChatResponse(BaseModel):
    """채팅 응답"""
    query: str = Field(..., description="사용자 질문")
    response: str = Field(..., description="AI 응답")
    sources: List[SourceDocument] = Field(default=[], description="참조 문서 목록")
    performance: Optional[PerformanceMetricsDTO] = Field(None, description="성능 메트릭")
    timestamp: datetime = Field(default_factory=datetime.now, description="응답 시간")
    success: bool = Field(True, description="성공 여부")
    error: Optional[str] = Field(None, description="에러 메시지")


class ConversationalChatRequest(BaseModel):
    """연속 대화 채팅 요청 (Memory 포함)"""
    query: str = Field(..., description="사용자 질문", min_length=1)
    session_id: str = Field("default", description="세션 ID (대화 기록 구분)")
    context_count: Optional[int] = Field(5, description="검색할 컨텍스트 수", ge=1, le=20)
    temperature: Optional[float] = Field(0.1, description="생성 온도", ge=0.0, le=2.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "그럼 AMR은?",
                "session_id": "user_12345",
                "context_count": 5,
                "temperature": 0.1
            }
        }


class ChatHistoryItem(BaseModel):
    """채팅 히스토리 항목"""
    query: str = Field(..., description="질문")
    timestamp: str = Field(..., description="시간")
    inference_time_ms: float = Field(0, description="추론 시간")
    tokens_per_second: float = Field(0, description="초당 토큰 수")
    success: bool = Field(True, description="성공 여부")

