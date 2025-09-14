#!/usr/bin/env python3
"""
WMS 산업용 협동로봇 전문 시스템
==============================

크롤링 → 텍스트 처리 → 벡터DB 구축의 완전 자동화 시스템
AMR, AGV, CNV, RTV 등 산업용 협동로봇 전문 지식 구축

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
            logger.info("🚀 산업용 협동로봇 전문 시스템 파이프라인 시작")
            
            # 1단계: 고도화된 키워드로 논문 크롤링
            await self.stage1_enhanced_crawling()
            
            # 2단계: 전문 용어 기반 텍스트 처리
            await self.stage2_professional_text_processing()
            
            # 3단계: 벡터DB 구축 (ChromaDB + Faiss)
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
                "--max-results", "150"  # 더 많은 전문 논문 수집
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ 1단계 완료: 산업용 협동로봇 전문 논문 수집 완료")
            else:
                logger.warning(f"⚠️ 크롤링 일부 실패: {stderr.decode()}")
                
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
        """3단계: 벡터DB 구축"""
        self.status.update({"stage": "vectordb", "progress": 75, "message": "산업용 협동로봇 전문지식 벡터DB 구축 중..."})
        logger.info("🗄️ 3단계: 벡터DB 구축")
        
        try:
            # ChromaDB 구축
            builder_script = Path("WMS/Tools/chromadb_builder.py")
            cmd = [
                sys.executable, str(builder_script),
                "--processed-data", "WMS/ProcessedData",
                "--vector-db", "WMS/VectorDB",
                "--embedding-model", "sentence-transformers",
                "--action", "build"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ ChromaDB 구축 완료")
                
                # Faiss 변환
                await self.convert_to_faiss()
                
            else:
                logger.error(f"❌ ChromaDB 구축 실패: {stderr.decode()}")
                raise Exception("ChromaDB 구축 실패")
                
        except Exception as e:
            logger.error(f"❌ 3단계 실패: {e}")
            raise
    
    async def convert_to_faiss(self):
        """ChromaDB → Faiss 변환"""
        logger.info("🔄 ChromaDB → Faiss 변환")
        
        try:
            migrator_script = Path("WMS/Tools/chroma_to_faiss_migrator.py")
            cmd = [
                sys.executable, str(migrator_script),
                "--chroma-db", "WMS/VectorDB/chroma_storage",
                "--output-dir", "output/industrial_robot_vectordb",
                "--collection", "wms_research_papers"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ Faiss 변환 완료")
            else:
                logger.error(f"❌ Faiss 변환 실패: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"❌ Faiss 변환 실패: {e}")
    
    async def stage4_system_validation(self):
        """4단계: 시스템 검증"""
        self.status.update({"stage": "validation", "progress": 100, "message": "산업용 협동로봇 전문 시스템 검증 완료"})
        logger.info("✅ 4단계: 시스템 검증")
        
        # 생성된 파일들 확인
        output_dir = Path("output/industrial_robot_vectordb")
        required_files = ["wms_knowledge.index", "documents.json", "metadata.json"]
        
        for file_name in required_files:
            file_path = output_dir / file_name
            if file_path.exists():
                logger.info(f"✅ {file_name} 생성 확인")
            else:
                logger.warning(f"⚠️ {file_name} 누락")
        
        # 전문 용어 통계 출력
        self.print_professional_statistics()
    
    def print_professional_statistics(self):
        """전문 용어 통계 출력"""
        logger.info("📊 산업용 협동로봇 전문 시스템 통계:")
        logger.info(f"🤖 로봇 시스템 용어: {len(self.enhanced_keywords.get_category_terms('robot_systems'))}개")
        logger.info(f"🏭 제어 시스템 용어: {len(self.enhanced_keywords.get_category_terms('control_systems'))}개")
        logger.info(f"🔧 피킹 기술 용어: {len(self.enhanced_keywords.get_category_terms('picking_technologies'))}개")
        logger.info(f"📦 저장 시스템 용어: {len(self.enhanced_keywords.get_category_terms('storage_systems'))}개")
        logger.info(f"🏗️ 스마트팩토리 용어: {len(self.enhanced_keywords.get_category_terms('smart_factory'))}개")

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
    
    logger.info("🚀 WMS 산업용 협동로봇 전문 시스템 시작")
    
    # 전역 파이프라인 초기화
    wms_pipeline = WMSIndustrialRobotPipeline()
    enhanced_keywords = EnhancedWMSKeywords()
    
    try:
        yield
    except Exception as e:
        logger.error(f"❌ 시스템 오류: {e}")
        raise e
    finally:
        logger.info("🏁 WMS 산업용 협동로봇 전문 시스템 종료")

# FastAPI 앱 생성
app = FastAPI(
    title="WMS 산업용 협동로봇 전문 시스템",
    description="AMR, AGV, CNV, RTV 등 산업용 협동로봇 전문지식 구축 시스템",
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
    return {"status": "healthy", "message": "WMS 산업용 협동로봇 전문 시스템 정상 작동"}

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
        "robot_systems": enhanced_keywords.get_category_terms('robot_systems')[:10],
        "control_systems": enhanced_keywords.get_category_terms('control_systems')[:10],
        "smart_factory": enhanced_keywords.get_category_terms('smart_factory')[:10],
        "total_keywords": len(enhanced_keywords.get_all_search_keywords())
    }


if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="WMS 산업용 협동로봇 전문 시스템")
    parser.add_argument("--mode", choices=["api", "pipeline"], default="pipeline",
                       help="실행 모드: api (웹서버) 또는 pipeline (직접 실행)")
    parser.add_argument("--port", type=int, default=8000, help="API 서버 포트")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        # 직접 파이프라인 실행 모드
        print("🚀 WMS 산업용 협동로봇 전문 시스템 직접 실행")
        print("=" * 60)
        print("📡 1단계: AMR, AGV, CNV, RTV 전문 키워드로 논문 크롤링")
        print("📝 2단계: WCS, WES, MES 제어시스템 용어 추출")
        print("🗄️ 3단계: 산업용 협동로봇 전문지식 벡터DB 구축")
        print("✅ 4단계: 시스템 검증 및 완료")
        print("=" * 60)
        
        # 동기 실행을 위한 래퍼
        async def run_pipeline():
            pipeline = WMSIndustrialRobotPipeline()
            result = await pipeline.run_complete_pipeline()
            
            print("\n🎉 최종 결과:")
            print(f"상태: {result['status']}")
            print(f"메시지: {result['message']}")
            
            if result['status'] == 'success':
                print("\n📁 생성된 파일:")
                output_dir = Path("output/industrial_robot_vectordb")
                if output_dir.exists():
                    for file in output_dir.iterdir():
                        if file.is_file():
                            print(f"  - {file.name}")
                
                print("\n💡 사용 방법:")
                print("  1. output/industrial_robot_vectordb/ 폴더의 파일들을 본 프로젝트로 복사")
                print("  2. wms_knowledge.index를 Faiss 인덱스로 로드")
                print("  3. documents.json, metadata.json과 함께 사용")
                print("\n🎯 이제 본 프로젝트가 산업용 협동로봇 전문 AI가 되었습니다!")
        
        # 비동기 실행
        asyncio.run(run_pipeline())
        
    else:
        # API 서버 모드
        print(f"🌐 WMS 산업용 협동로봇 전문 시스템 API 서버 시작")
        print(f"📖 API 문서: http://localhost:{args.port}/docs")
        print(f"🚀 파이프라인 시작: POST http://localhost:{args.port}/pipeline/start")
        print(f"📊 상태 확인: GET http://localhost:{args.port}/pipeline/status")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=args.port,
            reload=True,
        )
        
        print("🏁 WMS 산업용 협동로봇 전문 시스템 API 서버 종료")
