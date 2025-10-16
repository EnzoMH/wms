#!/usr/bin/env python3
"""
파이프라인 라우터
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import logging
import subprocess
import sys
from pathlib import Path

from app.pipeline.dto import PipelineStatus, ScraperStats, ChunkStats, VectorDBStats
from app.pipeline.service import PipelineService

logger = logging.getLogger(__name__)

# 라우터 생성
router = APIRouter(
    prefix="/pipeline",
    tags=["pipeline"],
    responses={404: {"description": "Not found"}},
)

def get_pipeline_service():
    """파이프라인 서비스 의존성 주입"""
    return PipelineService()


@router.get("/status", response_model=PipelineStatus)
async def get_pipeline_status(
    service: PipelineService = Depends(get_pipeline_service)
):
    """전체 파이프라인 현황"""
    try:
        status = service.get_pipeline_status()
        return PipelineStatus(**status)
    except Exception as e:
        logger.error(f"파이프라인 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scrapers", response_model=ScraperStats)
async def get_scrapers():
    """크롤링 현황"""
    try:
        return ScraperStats(**PipelineService.get_scraper_stats())
    except Exception as e:
        logger.error(f"크롤링 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunks", response_model=ChunkStats)
async def get_chunks():
    """청크 처리 현황"""
    try:
        return ChunkStats(**PipelineService.get_chunk_stats())
    except Exception as e:
        logger.error(f"청크 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectordb", response_model=VectorDBStats)
async def get_vectordb():
    """벡터DB 현황"""
    try:
        return VectorDBStats(**PipelineService.get_vectordb_stats())
    except Exception as e:
        logger.error(f"벡터DB 통계 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 크롤링 트리거 API
@router.post("/trigger/scraper")
async def trigger_scraper(background_tasks: BackgroundTasks):
    """
    크롤링 실행 트리거
    
    백그라운드에서 크롤러 실행
    """
    try:
        scraper_script = Path("0_core/0_scrapers/industry_news_scraper.py")
        
        if not scraper_script.exists():
            raise HTTPException(status_code=404, detail="크롤러 스크립트 없음")
        
        # 백그라운드 작업으로 실행
        background_tasks.add_task(run_scraper, str(scraper_script))
        
        return {
            "status": "started",
            "message": "크롤링이 백그라운드에서 실행 중입니다",
            "script": str(scraper_script)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"크롤링 트리거 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/extractor")
async def trigger_extractor(background_tasks: BackgroundTasks):
    """텍스트 추출기 실행"""
    try:
        extractor_script = Path("0_core/1_extractors/text_extractor.py")
        
        if not extractor_script.exists():
            raise HTTPException(status_code=404, detail="추출기 스크립트 없음")
        
        background_tasks.add_task(run_script, str(extractor_script))
        
        return {
            "status": "started",
            "message": "텍스트 추출이 백그라운드에서 실행 중입니다"
        }
        
    except Exception as e:
        logger.error(f"추출기 트리거 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger/rebuild-vectordb")
async def trigger_rebuild_vectordb(background_tasks: BackgroundTasks):
    """벡터DB 재구축"""
    try:
        builder_script = Path("0_core/vectorDB/faiss_builder.py")
        
        if not builder_script.exists():
            raise HTTPException(status_code=404, detail="빌더 스크립트 없음")
        
        background_tasks.add_task(run_script, str(builder_script))
        
        return {
            "status": "started",
            "message": "벡터DB 재구축이 백그라운드에서 실행 중입니다"
        }
        
    except Exception as e:
        logger.error(f"벡터DB 재구축 트리거 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 백그라운드 작업 함수
def run_scraper(script_path: str):
    """크롤러 실행"""
    try:
        logger.info(f"크롤링 시작: {script_path}")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 타임아웃
        )
        
        if result.returncode == 0:
            logger.info("크롤링 완료")
        else:
            logger.error(f"크롤링 실패: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("크롤링 타임아웃")
    except Exception as e:
        logger.error(f"크롤링 실행 오류: {e}")


def run_script(script_path: str):
    """일반 스크립트 실행"""
    try:
        logger.info(f"스크립트 시작: {script_path}")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            logger.info(f"스크립트 완료: {script_path}")
        else:
            logger.error(f"스크립트 실패: {result.stderr}")
            
    except Exception as e:
        logger.error(f"스크립트 실행 오류: {e}")
