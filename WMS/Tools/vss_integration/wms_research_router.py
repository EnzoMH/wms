#!/usr/bin/env python3
"""
WMS Research Router
==================

VSS-AI-API-dev에 추가할 WMS 연구 논문 검색 라우터
"""

from fastapi import APIRouter, Body, Depends, HTTPException
from typing import Optional, List, Dict, Any
import logging

# DTO 클래스들
from pydantic import BaseModel

class SearchRequestDto(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QuestionRequestDto(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SearchResponseDto(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_found: int

class QuestionResponseDto(BaseModel):
    answer: str
    sources: List[str]
    search_results: List[Dict[str, Any]]

class TrendsResponseDto(BaseModel):
    topic: str
    analysis_type: str
    data: Dict[str, Any]

class PaperDetailsResponseDto(BaseModel):
    paper_filename: str
    title: str
    source: str
    total_chunks: int
    total_characters: int
    similar_papers: List[str]
    first_chunk_preview: str

class ComparisonRequestDto(BaseModel):
    paper1: str
    paper2: str

# 라우터 생성
router = APIRouter(
    prefix="/wms-research",
    tags=["wms-research"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# 의존성 함수들
def get_wms_research_service():
    """WMS Research Service 의존성 주입"""
    try:
        from .wms_faiss_repository import WMSFaissRepository
        from .wms_research_service import WMSResearchService
        
        # Faiss 저장소 경로 (환경변수 또는 기본값)
        import os
        faiss_storage_path = os.getenv("WMS_FAISS_STORAGE_PATH", "./resource/wms_knowledge/faiss_storage")
        
        # Repository 초기화
        repository = WMSFaissRepository(faiss_storage_path)
        
        # LLM 초기화 (기존 VSS 시스템의 LLM 사용)
        try:
            from core.langchain_manager import model_korean_normal as llm
        except ImportError:
            logger.warning("LLM을 로드할 수 없습니다. 질의응답 기능이 제한됩니다.")
            llm = None
        
        # Service 초기화
        service = WMSResearchService(repository, llm)
        return service
        
    except Exception as e:
        logger.error(f"❌ WMS Research Service 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"서비스 초기화 실패: {str(e)}")

# ============================================================================
# API 엔드포인트들
# ============================================================================

@router.post("/search", 
             description="WMS 연구 논문 벡터 검색",
             response_model=SearchResponseDto)
async def search_papers(
    request: SearchRequestDto = Body(...),
    service = Depends(get_wms_research_service)
):
    """
    WMS 관련 연구 논문을 벡터 검색합니다.
    
    - **query**: 검색할 키워드나 질문
    - **top_k**: 반환할 결과 수 (기본값: 5)
    """
    try:
        result = service.search_papers(request.query, request.top_k)
        return SearchResponseDto(**result)
    except Exception as e:
        logger.error(f"❌ 논문 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@router.post("/ask", 
             description="WMS 연구 질의응답 (RAG)",
             response_model=QuestionResponseDto)
async def ask_question(
    request: QuestionRequestDto = Body(...),
    service = Depends(get_wms_research_service)
):
    """
    WMS 연구 논문을 기반으로 질문에 답변합니다.
    
    - **question**: 질문 내용
    - **top_k**: 참고할 문서 수 (기본값: 3)
    """
    try:
        result = service.ask_question(request.question, request.top_k)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return QuestionResponseDto(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 질의응답 실패: {e}")
        raise HTTPException(status_code=500, detail=f"질의응답 실패: {str(e)}")

@router.get("/trends", 
            description="WMS 연구 동향 분석")
async def get_research_trends(
    topic: Optional[str] = None,
    top_k: int = 10,
    service = Depends(get_wms_research_service)
):
    """
    WMS 연구 동향을 분석합니다.
    
    - **topic**: 분석할 주제 (없으면 전체 분석)
    - **top_k**: 분석할 문서 수
    """
    try:
        result = service.get_research_trends(topic, top_k)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 연구 동향 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"동향 분석 실패: {str(e)}")

@router.get("/stats", 
            description="WMS 데이터베이스 통계")
async def get_database_stats(
    service = Depends(get_wms_research_service)
):
    """
    WMS 연구 데이터베이스의 통계 정보를 조회합니다.
    """
    try:
        return service.get_database_stats()
    except Exception as e:
        logger.error(f"❌ 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@router.get("/paper/{paper_filename:path}", 
            description="특정 논문 상세 정보",
            response_model=PaperDetailsResponseDto)
async def get_paper_details(
    paper_filename: str,
    service = Depends(get_wms_research_service)
):
    """
    특정 논문의 상세 정보를 조회합니다.
    
    - **paper_filename**: 논문 파일명 (예: "001_Paper_Title.pdf")
    """
    try:
        result = service.get_paper_details(paper_filename)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return PaperDetailsResponseDto(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 논문 상세 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"논문 정보 조회 실패: {str(e)}")

@router.post("/compare", 
             description="두 논문 비교 분석")
async def compare_papers(
    request: ComparisonRequestDto = Body(...),
    service = Depends(get_wms_research_service)
):
    """
    두 논문을 비교 분석합니다.
    
    - **paper1**: 첫 번째 논문 파일명
    - **paper2**: 두 번째 논문 파일명
    """
    try:
        result = service.compare_papers(request.paper1, request.paper2)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 논문 비교 실패: {e}")
        raise HTTPException(status_code=500, detail=f"논문 비교 실패: {str(e)}")

@router.get("/sources/{source}", 
            description="소스별 논문 목록")
async def get_papers_by_source(
    source: str,
    limit: int = 10,
    service = Depends(get_wms_research_service)
):
    """
    특정 소스의 논문 목록을 조회합니다.
    
    - **source**: 논문 소스 (ArXiv, IEEE, SemanticScholar, GoogleScholar)
    - **limit**: 반환할 논문 수
    """
    try:
        result = service.search_by_source(source, limit)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 소스별 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"소스별 검색 실패: {str(e)}")

@router.get("/health", 
            description="WMS Research API 상태 확인")
async def health_check(
    service = Depends(get_wms_research_service)
):
    """
    WMS Research API의 상태를 확인합니다.
    """
    try:
        stats = service.get_database_stats()
        return {
            "status": "healthy",
            "total_documents": stats["total_documents"],
            "total_papers": stats["total_papers"],
            "embedding_model": stats["embedding_model"],
            "vector_dimension": stats["vector_dimension"]
        }
    except Exception as e:
        logger.error(f"❌ 헬스체크 실패: {e}")
        raise HTTPException(status_code=503, detail=f"서비스 불가: {str(e)}")

# ============================================================================
# 고급 검색 엔드포인트들
# ============================================================================

@router.post("/advanced-search", 
             description="고급 검색 (필터링 옵션)")
async def advanced_search(
    query: str = Body(...),
    source_filter: Optional[List[str]] = Body(None),
    min_similarity: Optional[float] = Body(0.0),
    top_k: int = Body(10),
    service = Depends(get_wms_research_service)
):
    """
    고급 검색 옵션을 제공합니다.
    
    - **query**: 검색 쿼리
    - **source_filter**: 필터링할 소스 목록
    - **min_similarity**: 최소 유사도 임계값
    - **top_k**: 반환할 결과 수
    """
    try:
        # 기본 검색 수행
        results = service.search_papers(query, top_k * 2)  # 여유있게 검색
        
        # 필터링 적용
        filtered_results = []
        for result in results["results"]:
            # 소스 필터
            if source_filter and result["metadata"]["paper_source"] not in source_filter:
                continue
            
            # 유사도 필터
            if result["similarity"] < min_similarity:
                continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return {
            "query": query,
            "filters": {
                "source_filter": source_filter,
                "min_similarity": min_similarity
            },
            "results": filtered_results,
            "total_found": len(filtered_results)
        }
        
    except Exception as e:
        logger.error(f"❌ 고급 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"고급 검색 실패: {str(e)}")

@router.get("/keywords", 
            description="추천 검색 키워드")
async def get_recommended_keywords():
    """
    WMS 연구 분야의 추천 검색 키워드를 반환합니다.
    """
    keywords = {
        "automation": ["창고 자동화", "warehouse automation", "AGV", "AMR", "로봇 피킹"],
        "systems": ["WMS", "WCS", "WES", "MES", "ERP 통합"],
        "technologies": ["RFID", "IoT", "AI", "머신러닝", "컴퓨터 비전"],
        "operations": ["재고 관리", "주문 처리", "피킹 최적화", "경로 계획"],
        "analysis": ["성능 분석", "ROI", "효율성", "비용 절감", "생산성"]
    }
    
    return {
        "recommended_keywords": keywords,
        "usage_tip": "이 키워드들을 조합하여 더 정확한 검색 결과를 얻을 수 있습니다."
    }


