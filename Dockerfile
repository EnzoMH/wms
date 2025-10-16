# WMS RAG System - FastAPI + EXAONE
# Python 3.12 기반 경량화 이미지
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 파일 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV USE_OLLAMA=false
ENV USE_VLLM=false
ENV OLLAMA_MODEL=exaone-rag
# ENV PORT=8000  # 선택사항: 지정하지 않으면 8000부터 자동 탐색

# 포트 노출 (8000-8010 범위)
EXPOSE 8000-8010

# 헬스체크 (포트는 docker-compose에서 지정)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# 실행 명령
CMD ["python", "main.py"]
