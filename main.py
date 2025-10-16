#!/usr/bin/env python3
"""
WMS RAG API 메인 서버
===================

EXAONE 기반 RAG 시스템
"""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 로깅 설정을 가장 먼저 수행
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 라우터 import
from app.chat.router import router as chat_router, set_exaone_rag
from app.pipeline.router import router as pipeline_router
from app.system.router import router as system_router

# 전역 변수
exaone_rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    global exaone_rag
    
    logger.info("=" * 60)
    logger.info("WMS RAG API 서버 시작")
    logger.info("=" * 60)
    
    # EXAONE RAG 시스템 초기화
    try:
        from _0_1_core.rag.wrapper import ExaoneRAGWrapper
        
        # 실행 모드 선택 (우선순위: Ollama > vLLM > Transformers)
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        use_vllm = os.getenv("USE_VLLM", "false").lower() == "true"
        ollama_model = os.getenv("OLLAMA_MODEL", "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M")
        vllm_url = os.getenv("VLLM_URL", "http://localhost:8080/v1")
        
        logger.info(f"[*] EXAONE RAG 초기화 중...")
        
        if use_ollama:
            logger.info(f"    모드: Ollama (GPU 부담 적음, 권장)")
            logger.info(f"    모델: {ollama_model}")
        elif use_vllm:
            logger.info(f"    모드: vLLM")
            logger.info(f"    vLLM URL: {vllm_url}")
        else:
            logger.info(f"    모드: Transformers (직접)")
        
        exaone_rag = ExaoneRAGWrapper(
            use_ollama=use_ollama,
            use_vllm=use_vllm,
            ollama_model=ollama_model,
            vllm_url=vllm_url
        )
        
        # chat router에 설정
        set_exaone_rag(exaone_rag)
        
        logger.info("[COMPLETE] EXAONE RAG 시스템 초기화 완료")
                
    except Exception as e:
        logger.error(f"[ERROR] EXAONE RAG 초기화 실패: {e}")
        logger.warning("[WARNING] RAG 기능이 제한될 수 있습니다")
        exaone_rag = None
    
    yield
    
    # 종료 시 정리
    logger.info("WMS RAG API 서버 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="WMS RAG API",
    description="창고 자동화 RAG 시스템 - EXAONE-4.0-1.2B 기반",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 헬스체크 (루트)
@app.get("/health")
async def health_get():
    return {"message": "OK"}


@app.post("/health")
async def health_post():
    return {"message": "OK"}


# 라우터 등록
app.include_router(chat_router)
app.include_router(pipeline_router)
app.include_router(system_router)


if __name__ == "__main__":
    import uvicorn
    import socket
    
    def find_available_port(start_port):
        """사용 가능한 포트 찾기 (local 환경용)"""
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
                port += 1
    
    # PORT 환경 변수가 있으면 사용, 없으면 8000부터 자동 탐색
    port_env = os.getenv("PORT")
    if port_env:
        # 환경 변수로 포트 지정됨 (Docker 권장)
        port = int(port_env)
        logger.info(f"환경 변수 PORT={port} 사용")
    else:
        # 8000부터 자동 탐색 (Local/Docker 모두 가능)
        port = find_available_port(8000)
        logger.info(f"사용 가능한 포트 {port} 자동 선택")
    
    print(f"api docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port)
