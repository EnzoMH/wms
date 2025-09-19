#!/usr/bin/env python3
"""
창고 자동화 시스템 전문 플랫폼
============================

크롤링 → 텍스트 처리 → 벡터DB 구축의 완전 자동화 시스템
AGV, EMS, RTV, CNV 등 창고 자동화 시스템 전문 지식 구축

실행: python main.py
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

# FastAPI 관련
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 로깅 설정
from loguru import logger
from contextlib import asynccontextmanager

# WMS 전문 모듈들 import
sys.path.append(str(Path(__file__).parent / "Tools"))
from enhanced_wms_keywords import EnhancedWMSKeywords

load_dotenv()

# 전역 변수
wms_pipeline = None
enhanced_keywords = None

class WarehouseAutomationPipeline:
    """창고 자동화 시스템(AGV, EMS, RTV, CNV) 전문 파이프라인"""
    
    def __init__(self):
        self.setup_directories()
        self.enhanced_keywords = EnhancedWMSKeywords()
        self.status = {"stage": "initialized", "progress": 0, "message": "시스템 초기화 완료"}
        
    def setup_directories(self):
        """필요한 디렉토리 구조 생성"""
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
            logger.info("🚀 창고 자동화 시스템 전문 파이프라인 시작")
            
            # 1단계: 고도화된 키워드로 논문 크롤링
            await self.stage1_enhanced_crawling()
            
            # 2단계: 전문 용어 기반 텍스트 처리
            await self.stage2_professional_text_processing()
            
            # 3단계: 벡터DB 구축 (Faiss)
            await self.stage3_vector_database_creation()
            
            # 4단계: 최종 검증
            await self.stage4_system_validation()
            
            logger.info("🎉 창고 자동화 시스템 전문 플랫폼 구축 완료!")
            return {"status": "success", "message": "창고 자동화 시스템 전문 플랫폼 구축 완료"}
            
        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return {"status": "error", "message": str(e)}
    
    async def stage1_enhanced_crawling(self):
        """1단계: 고도화된 키워드로 논문 크롤링"""
        self.status.update({"stage": "crawling", "progress": 25, "message": "AGV, EMS, RTV, CNV 등 창고 자동화 전문 키워드로 논문 수집 중..."})
        logger.info("📡 1단계: 창고 자동화 시스템 전문 키워드로 논문 크롤링")
        
        try:
            scraper_script = Path("Tools/paper_scraper.py")
            cmd = [
                sys.executable, str(scraper_script),
                "--output-dir", "WMS/Papers",
                "--max-results", "150"  # 더 많은 전문 논문 수집
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ 1단계 완료: 창고 자동화 시스템 전문 논문 수집 완료")
            else:
                logger.warning(f"⚠️ 크롤링 일부 실패: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ 1단계 실패: {e}")
            raise
    
    async def stage2_professional_text_processing(self):
        """2단계: 전문 용어 기반 텍스트 처리"""
        self.status.update({"stage": "processing", "progress": 50, "message": "창고 자동화 시스템 전문 용어 및 경로 최적화 추출 중..."})
        logger.info("📝 2단계: 창고 자동화 시스템 전문 용어 기반 텍스트 처리")
        
        try:
            extractor_script = Path("Tools/text_extractor.py")
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
                logger.info("✅ 2단계 완료: 창고 자동화 시스템 전문 용어 추출 완료")
            else:
                logger.error(f"❌ 텍스트 처리 실패: {stderr.decode()}")
                raise Exception("텍스트 처리 실패")
                
        except Exception as e:
            logger.error(f"❌ 2단계 실패: {e}")
            raise
    
    async def stage3_vector_database_creation(self):
        """3단계: Faiss 벡터DB 구축"""
        self.status.update({"stage": "vectordb", "progress": 75, "message": "창고 자동화 시스템 전문지식 Faiss 벡터DB 구축 중..."})
        logger.info("🗄️ 3단계: Faiss 벡터DB 구축")
        
        try:
            # Faiss 벡터DB 직접 구축
            builder_script = Path("Tools/faiss_builder.py")
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
        self.status.update({"stage": "validation", "progress": 100, "message": "창고 자동화 시스템 전문 플랫폼 검증 완료"})
        logger.info("✅ 4단계: 시스템 검증")
        
        # 생성된 파일들 확인
        output_dir = Path("WMS/VectorDB")
        required_files = ["warehouse_automation_knowledge.index", "documents.json", "metadata.json"]
        
        for file_name in required_files:
            file_path = output_dir / file_name
            if file_path.exists():
                logger.info(f"✅ {file_name} 생성 확인")
            else:
                logger.warning(f"⚠️ {file_name} 누락")
        
        # 전문 용어 통계 출력
        self.print_professional_statistics()
    
    def print_professional_statistics(self):
        """창고 자동화 시스템 전문 용어 통계 출력"""
        logger.info("📊 창고 자동화 시스템 전문 플랫폼 통계:")
        logger.info(f"🏭 핵심 WMS 용어: {len(self.enhanced_keywords.get_keywords_by_category('core_wms'))}개")
        logger.info(f"🤖 AGV/EMS/RTV 용어: {len(self.enhanced_keywords.get_keywords_by_category('collaborative_robot'))}개")
        logger.info(f"⚡ 자동화/IoT 용어: {len(self.enhanced_keywords.get_keywords_by_category('automation_iot'))}개")
        logger.info(f"📦 물류/공급망 용어: {len(self.enhanced_keywords.get_keywords_by_category('logistics_supply_chain'))}개")
        logger.info(f"📈 성능/최적화 용어: {len(self.enhanced_keywords.get_keywords_by_category('performance_optimization'))}개")

# Pydantic 모델들
class PipelineStatus(BaseModel):
    stage: str
    progress: int
    message: str

class PipelineResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global wms_pipeline, enhanced_keywords
    
    logger.info("🚀 창고 자동화 시스템 전문 플랫폼 시작")
    
    # 전역 파이프라인 초기화
    wms_pipeline = WarehouseAutomationPipeline()
    enhanced_keywords = EnhancedWMSKeywords()
    
    try:
        yield
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        raise e
    finally:
        logger.info("🏁 창고 자동화 시스템 전문 플랫폼 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="창고 자동화 시스템 전문 플랫폼",
    description="AGV, EMS, RTV, CNV 등 창고 자동화 시스템 전문지식 구축 플랫폼",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 엔드포인트들
@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {"status": "healthy", "message": "창고 자동화 시스템 전문 플랫폼 정상 작동"}

@app.post("/pipeline/start", response_model=PipelineResponse)
async def start_pipeline():
    """전체 파이프라인 시작"""
    global wms_pipeline
    
    if wms_pipeline is None:
        raise HTTPException(status_code=500, detail="시스템이 초기화되지 않았습니다")
    
    try:
        result = await wms_pipeline.run_complete_pipeline()
        return PipelineResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파이프라인 실행 실패: {str(e)}")

@app.get("/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """파이프라인 상태 조회"""
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


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="창고 자동화 시스템 전문 플랫폼")
    parser.add_argument("--mode", choices=["api", "pipeline"], default="pipeline",
                       help="실행 모드: api (웹서버) 또는 pipeline (직접 실행)")
    parser.add_argument("--port", type=int, default=8000, help="API 서버 포트")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        # 직접 파이프라인 실행 모드
        print("🚀 창고 자동화 시스템 전문 플랫폼 직접 실행")
        print("=" * 60)
        print("📡 1단계: AGV, EMS, RTV, CNV 전문 키워드로 논문 크롤링")
        print("📝 2단계: 창고 자동화 시스템 전문 용어 추출")
        print("🗄️ 3단계: 창고 자동화 시스템 전문지식 벡터DB 구축")
        print("✅ 4단계: 시스템 검증 및 완료")
        print("=" * 60)
        
        # 동기 실행을 위한 래퍼
        async def run_pipeline():
            pipeline = WarehouseAutomationPipeline()
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
                print("  2. warehouse_automation_knowledge.index를 Faiss 인덱스로 로드")
                print("  3. documents.json, metadata.json과 함께 사용")
                print("\n🎯 이제 본 프로젝트가 창고 자동화 시스템 전문 AI가 되었습니다!")
        
        # 비동기 실행
        asyncio.run(run_pipeline())
        
    else:
        # API 서버 모드
        print(f"🌐 창고 자동화 시스템 전문 플랫폼 API 서버 시작")
        print(f"📖 API 문서: http://localhost:{args.port}/docs")
        print(f"🚀 파이프라인 시작: POST http://localhost:{args.port}/pipeline/start")
        print(f"📊 상태 확인: GET http://localhost:{args.port}/pipeline/status")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            reload=True,
        )
        
        print("🏁 창고 자동화 시스템 전문 플랫폼 API 서버 종료")
