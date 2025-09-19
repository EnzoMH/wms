#!/usr/bin/env python3
"""
WMS 산업용 협동로봇 전문 시스템 - 통합 API
============================================

크롤링 → 텍스트 처리 → 벡터DB 구축의 완전 자동화 시스템
AMR, AGV, CNV, RTV 등 산업용 협동로봇 전문 지식 구축

FastAPI 기반 웹 API + 파이프라인 자동화
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime

# FastAPI 관련
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles  
from pydantic import BaseModel

# 로깅 설정
from loguru import logger
from contextlib import asynccontextmanager

# WMS 전문 모듈들 import
sys.path.append(str(Path(__file__).parent / "WMS" / "Tools"))
from enhanced_wms_keywords import EnhancedWMSKeywords

load_dotenv()

# 전역 변수
wms_pipeline = None
enhanced_keywords = None

class WMSIndustrialRobotPipeline:
    """산업용 협동로봇 전문 시스템 파이프라인"""
    
    def __init__(self):
        self.setup_directories()
        self.enhanced_keywords = EnhancedWMSKeywords()
        self.status = {"stage": "initialized", "progress": 0, "message": "시스템 초기화 완료"}
        
    def setup_directories(self):
        """필요한 디렉토리 구조 생성 (루트 기준)"""
        directories = [
            "WMS/Papers/ArXiv",
            "WMS/Papers/SemanticScholar", 
            "WMS/Papers/GoogleScholar",
            "WMS/ProcessedData",
            "WMS/VectorDB",
            "output"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ 디렉토리 구조 생성 완료")
    
    async def run_complete_pipeline(self):
        """전체 파이프라인 실행"""
        try:
            logger.info("🚀 산업용 협동로봇 전문 시스템 파이프라인 시작")
            
            # 1단계: 고도화된 키워드로 논문 크롤링
            await self.stage1_enhanced_crawling()
            
            # 2단계: 전문 용어 기반 텍스트 처리
            await self.stage2_professional_text_processing()
            
            # 3단계: Faiss 벡터DB 구축
            await self.stage3_vector_database_creation()
            
            # 4단계: 최종 검증
            await self.stage4_system_validation()
            
            logger.info("🎉 산업용 협동로봇 전문 시스템 구축 완료!")
            return {"status": "success", "message": "산업용 협동로봇 전문 시스템 구축 완료"}
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def stage1_enhanced_crawling(self):
        """1단계: 고도화된 키워드로 논문 크롤링"""
        self.status.update({"stage": "crawling", "progress": 25, "message": "AMR, AGV, CNV 등 전문 키워드로 논문 수집 중..."})
        logger.info("📡 1단계: 고도화된 전문 키워드로 논문 크롤링")
        
        try:
            scraper_script = Path("WMS/Tools/paper_scraper.py")
            cmd = [
                sys.executable, str(scraper_script),
                "--output-dir", "WMS/Papers",
                "--max-results", "150"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ 1단계 완료: 산업용 협동로봇 전문 논문 수집 완료")
            else:
                logger.warning(f"⚠️  크롤링 일부 실패: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ 1단계 실패: {e}")
            raise
    
    async def stage2_professional_text_processing(self):
        """2단계: 전문 용어 기반 텍스트 처리"""
        self.status.update({"stage": "processing", "progress": 50, "message": "WCS, WES, MES 등 제어시스템 용어 추출 중..."})
        logger.info("📝 2단계: 전문 용어 기반 텍스트 처리")
        
        try:
            extractor_script = Path("WMS/Tools/text_extractor.py")
            cmd = [
                sys.executable, str(extractor_script),
                "--papers-dir", "WMS/Papers",
                "--output-dir", "WMS/ProcessedData"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ 2단계 완료: 산업용 협동로봇 전문 용어 추출 완료")
            else:
                logger.error(f"❌ 텍스트 처리 실패: {stderr.decode()}")
                raise Exception("텍스트 처리 실패")
                
        except Exception as e:
            logger.error(f"❌ 2단계 실패: {e}")
            raise
    
    async def stage3_vector_database_creation(self):
        """3단계: Faiss 벡터DB 구축"""
        self.status.update({"stage": "vectordb", "progress": 75, "message": "산업용 협동로봇 전문지식 Faiss 벡터DB 구축 중..."})
        logger.info("🗄️ 3단계: Faiss 벡터DB 구축")
        
        try:
            builder_script = Path("WMS/Tools/faiss_builder.py")
            cmd = [
                sys.executable, str(builder_script),
                "--processed-data", "WMS/ProcessedData",
                "--vector-db", "WMS/VectorDB",
                "--action", "build"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Faiss 벡터 데이터베이스 구축 완료")
            else:
                logger.error(f"❌ Faiss 구축 실패: {stderr.decode()}")
                raise Exception("Faiss 구축 실패")
                
        except Exception as e:
            logger.error(f"❌ 3단계 실패: {e}")
            raise
    
    async def stage4_system_validation(self):
        """4단계: 시스템 검증"""
        self.status.update({"stage": "validation", "progress": 100, "message": "산업용 협동로봇 전문 시스템 검증 완료"})
        logger.info("✅ 4단계: 시스템 검증")
        
        # 생성된 파일들 확인
        output_dir = Path("WMS/VectorDB")
        required_files = ["wms_knowledge.index", "documents.json", "metadata.json"]
        
        for file_name in required_files:
            file_path = output_dir / file_name
            if file_path.exists():
                logger.info(f"✅ {file_name} 확인됨")
            else:
                logger.warning(f"⚠️ {file_name} 누락")
        
        # 전문 용어 통계 출력
        self.print_professional_statistics()
    
    def print_professional_statistics(self):
        """전문 용어 통계 출력"""
        logger.info("📊 산업용 협동로봇 전문 시스템 통계:")
        logger.info(f"🏭 핵심 WMS 용어: {len(self.enhanced_keywords.get_keywords_by_category('core_wms'))}개")
        logger.info(f"🤖 협동로봇 용어: {len(self.enhanced_keywords.get_keywords_by_category('collaborative_robot'))}개")
        logger.info(f"⚡ 자동화/IoT 용어: {len(self.enhanced_keywords.get_keywords_by_category('automation_iot'))}개")
        logger.info(f"📦 물류/공급망 용어: {len(self.enhanced_keywords.get_keywords_by_category('logistics_supply_chain'))}개")
        logger.info(f"📈 성능/최적화 용어: {len(self.enhanced_keywords.get_keywords_by_category('performance_optimization'))}개")


# Pydantic 모델들
class PipelineStatus(BaseModel):
    stage: str
    progress: int
    message: str

class PipelineResult(BaseModel):
    status: str
    message: str

class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5

# FastAPI lifespan 이벤트
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 라이프사이클 관리"""
    global wms_pipeline, enhanced_keywords
    
    logger.info("🚀 WMS 산업용 협동로봇 전문 시스템 API 시작")
    
    # 초기화
    try:
        enhanced_keywords = EnhancedWMSKeywords()
        wms_pipeline = WMSIndustrialRobotPipeline()
        logger.info("✅ WMS 시스템 초기화 완료")
        yield
    except Exception as e:
        logger.error(f"❌ 시스템 초기화 실패: {e}")
        raise e
    finally:
        logger.info("🔚 WMS 산업용 협동로봇 전문 시스템 API 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="비젼스페이스 WMS 산업용 협동로봇 전문 시스템",
    description="AMR, AGV, CNV, RTV 등 산업용 협동로봇 전문 지식 RAG 시스템",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check
@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {"status": "healthy", "message": "WMS 산업용 협동로봇 시스템 정상 작동"}

# 파이프라인 관련 API
@app.post("/pipeline/start", response_model=PipelineResult)
async def start_pipeline(background_tasks: BackgroundTasks):
    """전체 파이프라인 시작"""
    global wms_pipeline
    
    if wms_pipeline is None:
        raise HTTPException(status_code=500, detail="시스템이 초기화되지 않았습니다")
    
    # 백그라운드에서 실행
    background_tasks.add_task(wms_pipeline.run_complete_pipeline)
    
    return PipelineResult(
        status="started",
        message="산업용 협동로봇 전문 시스템 파이프라인이 백그라운드에서 시작되었습니다"
    )

@app.get("/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """파이프라인 진행 상황 조회"""
    global wms_pipeline
    
    if wms_pipeline is None:
        raise HTTPException(status_code=500, detail="시스템이 초기화되지 않았습니다")
    
    return PipelineStatus(**wms_pipeline.status)

@app.get("/keywords/professional")
async def get_professional_keywords():
    """전문 키워드 목록 조회"""
    global enhanced_keywords
    
    if enhanced_keywords is None:
        raise HTTPException(status_code=500, detail="키워드 시스템이 초기화되지 않았습니다")
    
    return {
        "core_wms": enhanced_keywords.get_keywords_by_category('core_wms')[:10],
        "collaborative_robot": enhanced_keywords.get_keywords_by_category('collaborative_robot')[:10],
        "automation_iot": enhanced_keywords.get_keywords_by_category('automation_iot')[:10],
        "total_keywords": len(enhanced_keywords.all_keywords)
    }

@app.get("/system/stats")
async def get_system_stats():
    """시스템 통계 정보"""
    try:
        papers_count = 0
        processed_count = 0
        
        # 논문 수 계산
        papers_dir = Path("WMS/Papers")
        if papers_dir.exists():
            for source_dir in papers_dir.iterdir():
                if source_dir.is_dir():
                    papers_count += len(list(source_dir.glob("*.pdf")))
        
        # 처리된 파일 수 계산
        processed_dir = Path("WMS/ProcessedData")
        if processed_dir.exists():
            processed_count = len(list(processed_dir.glob("chunks_*.json")))
        
        # 벡터DB 상태 확인
        vector_db_files = ["wms_knowledge.index", "documents.json", "metadata.json"]
        vector_db_status = {}
        for file_name in vector_db_files:
            file_path = Path("WMS/VectorDB") / file_name
            vector_db_status[file_name] = file_path.exists()
        
        return {
            "papers_collected": papers_count,
            "files_processed": processed_count,
            "processing_rate": f"{(processed_count/papers_count)*100:.1f}%" if papers_count > 0 else "0%",
            "vector_db_status": vector_db_status,
            "keywords_loaded": len(enhanced_keywords.all_keywords) if enhanced_keywords else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

@app.get("/files/download/{file_type}")
async def download_file(file_type: str):
    """생성된 파일들 다운로드"""
    file_map = {
        "summary": "WMS/ProcessedData/summary_report.md",
        "keywords": "WMS/ProcessedData/keywords.csv",
        "chunks": "WMS/ProcessedData/chunk_summary.json"
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=404, detail="지원하지 않는 파일 타입")
    
    file_path = Path(file_map[file_type])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다")
    
    return FileResponse(file_path)

# 메인 실행 부분
async def run_pipeline():
    """파이프라인 직접 실행 (CLI 모드)"""
    pipeline = WMSIndustrialRobotPipeline()
    
    print("🚀 WMS 산업용 협동로봇 전문 시스템 직접 실행")
    print("=" * 60)
    print("📡 1단계: AMR, AGV, CNV, RTV 전문 키워드로 논문 크롤링")
    print("📝 2단계: WCS, WES, MES 제어시스템 용어 추출")
    print("🗄️ 3단계: 산업용 협동로봇 전문지식 벡터DB 구축")  
    print("✅ 4단계: 시스템 검증 및 완료")
    print("=" * 60)
    
    result = await pipeline.run_complete_pipeline()
    
    print("\n🎉 최종 결과:")
    print(f"상태: {result['status']}")
    print(f"메시지: {result['message']}")
    
    if result['status'] == 'success':
        print("\n📁 생성된 파일:")
        output_dir = Path("WMS/VectorDB")
        if output_dir.exists():
            for file in output_dir.iterdir():
                if file.is_file():
                    print(f"  - {file.name}")
        
        print("\n💡 사용 방법:")
        print("  1. WMS/VectorDB/ 폴더의 파일들 확인")
        print("  2. wms_knowledge.index를 Faiss 인덱스로 로드")
        print("  3. documents.json, metadata.json과 함께 사용")
        print("\n🎯 이제 본 프로젝트가 산업용 협동로봇 전문 AI가 되었습니다!")


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="WMS 산업용 협동로봇 전문 시스템")
    parser.add_argument("--mode", choices=["api", "pipeline"], default="api", 
                       help="실행 모드: api (웹API 서버) 또는 pipeline (파이프라인 직접 실행)")
    parser.add_argument("--host", default="0.0.0.0", help="API 서버 호스트")
    parser.add_argument("--port", type=int, default=8001, help="API 서버 포트")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        # 파이프라인 직접 실행
        asyncio.run(run_pipeline())
    else:
        # FastAPI 서버 실행
        print(f"🌐 API 서버 시작: http://{args.host}:{args.port}")
        print(f"📚 API 문서: http://{args.host}:{args.port}/docs")
        print("🚀 비젼스페이스 WMS 산업용 협동로봇 전문 시스템 API 시작")
        
        uvicorn.run(
            app, 
            host=args.host, 
            port=args.port,
            reload=True,
        )
        print("🔚 비젼스페이스 WMS 산업용 협동로봇 전문 시스템 API 종료")