#!/usr/bin/env python3
"""
채팅 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from typing import List
from datetime import datetime
import logging

from app.chat.dto import (
    ChatRequest, ChatResponse, PerformanceMetricsDTO, 
    SourceDocument, ChatHistoryItem, ConversationalChatRequest
)
from app.chat.service import ChatService, ConversationalChatService

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={404: {"description": "Not found"}},
)

# 전역 변수 (main.py에서 초기화)
_exaone_rag = None

def set_exaone_rag(exaone_rag):
    """EXAONE RAG 인스턴스 설정"""
    global _exaone_rag
    _exaone_rag = exaone_rag

def get_chat_service():
    """채팅 서비스 의존성 주입"""
    return ChatService(exaone_rag=_exaone_rag)

def get_conversational_service():
    """대화형 채팅 서비스 의존성 주입"""
    return ConversationalChatService(exaone_rag=_exaone_rag)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest = Body(...),
    service: ChatService = Depends(get_chat_service)
):
    """
    RAG 채팅 API
    
    EXAONE wrapper.py를 사용하여 질문에 답변
    """
    try:
        # RAG 쿼리 실행
        result = service.query_rag(
            query=request.query,
            context_count=request.context_count,
            temperature=request.temperature
        )
        
        # 성능 메트릭 변환
        performance_dto = None
        if result.get("performance"):
            perf = result["performance"]
            performance_dto = PerformanceMetricsDTO(
                inference_time_ms=perf.get("inference_time_ms", 0),
                latency_ms=perf.get("latency_ms", 0),
                total_tokens=perf.get("total_tokens", 0),
                input_tokens=perf.get("input_tokens", 0),
                output_tokens=perf.get("output_tokens", 0),
                tokens_per_second=perf.get("tokens_per_second", 0),
                memory_used_mb=perf.get("memory_used_mb", 0),
                memory_percent=perf.get("memory_percent", 0),
                gpu_memory_used_mb=perf.get("gpu_memory_used_mb"),
                gpu_memory_percent=perf.get("gpu_memory_percent"),
                throughput_tokens_sec=perf.get("throughput_tokens_sec", 0),
                model_name=perf.get("model_name", "EXAONE-4.0-1.2B"),
                device=perf.get("device", "cpu")
            )
        
        # 더미 소스 문서 (TODO: 실제 검색 결과로 대체)
        contexts = [
            "AGV는 정해진 경로를 따라 이동하는 자동화 운반차량입니다.",
            "AMR은 자율주행이 가능한 모바일 로봇입니다."
        ]
        sources = [
            SourceDocument(
                content=ctx,
                score=0.85 - (i * 0.05),
                metadata={"index": i}
            )
            for i, ctx in enumerate(contexts[:request.context_count])
        ]
        
        return ChatResponse(
            query=request.query,
            response=result.get("response", ""),
            sources=sources,
            performance=performance_dto,
            timestamp=datetime.now(),
            success=result.get("success", False),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"채팅 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=List[ChatHistoryItem])
async def get_chat_history(
    limit: int = 20,
    service: ChatService = Depends(get_chat_service)
):
    """
    채팅 히스토리 조회
    
    performance_monitor.py의 로그에서 최근 질문 목록 반환
    """
    try:
        history = service.get_chat_history(limit)
        return [ChatHistoryItem(**item) for item in history]
    except Exception as e:
        logger.error(f"히스토리 조회 오류: {e}")
        return []


@router.post("/conversational", response_model=ChatResponse)
async def chat_with_memory(
    request: ConversationalChatRequest = Body(...),
    service: ConversationalChatService = Depends(get_conversational_service)
):
    """
    연속 대화 RAG 채팅 API (Memory 포함)
    
    session_id별로 대화 기록을 유지하여 자연스러운 연속 대화 가능
    
    - **session_id**: 세션 ID (사용자별 또는 대화별로 구분)
    - **query**: 질문 (이전 대화 문맥 이해 가능)
    
    예시:
    ```
    # Turn 1
    POST /chat/conversational
    {"query": "AGV가 뭐야?", "session_id": "user_123"}
    
    # Turn 2 - "그것"이 AGV를 의미한다고 이해
    POST /chat/conversational  
    {"query": "그것의 속도는?", "session_id": "user_123"}
    ```
    """
    try:
        # 대화 기록 포함 RAG 쿼리 실행
        result = service.query_with_memory(
            query=request.query,
            session_id=request.session_id,
            context_count=request.context_count,
            temperature=request.temperature
        )
        
        # 성능 메트릭 변환
        performance_dto = None
        if result.get("performance"):
            perf = result["performance"]
            performance_dto = PerformanceMetricsDTO(
                inference_time_ms=perf.get("inference_time_ms", 0),
                latency_ms=perf.get("latency_ms", 0),
                total_tokens=perf.get("total_tokens", 0),
                input_tokens=perf.get("input_tokens", 0),
                output_tokens=perf.get("output_tokens", 0),
                tokens_per_second=perf.get("tokens_per_second", 0),
                memory_used_mb=perf.get("memory_used_mb", 0),
                memory_percent=perf.get("memory_percent", 0),
                gpu_memory_used_mb=perf.get("gpu_memory_used_mb"),
                gpu_memory_percent=perf.get("gpu_memory_percent"),
                throughput_tokens_sec=perf.get("throughput_tokens_sec", 0),
                model_name=perf.get("model_name", "EXAONE-4.0-1.2B"),
                device=perf.get("device", "cpu")
            )
        
        # 소스 문서 변환
        sources = []
        if result.get("sources"):
            for i, ctx in enumerate(result["sources"][:request.context_count]):
                sources.append(SourceDocument(
                    content=ctx if isinstance(ctx, str) else ctx.get("document", ""),
                    score=0.85 - (i * 0.05),
                    metadata={"index": i, "session_id": request.session_id}
                ))
        
        return ChatResponse(
            query=request.query,
            response=result.get("response", ""),
            sources=sources,
            performance=performance_dto,
            timestamp=datetime.now(),
            success=result.get("success", False),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Conversational 채팅 API 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

