# WMS RAG API

창고 자동화 시스템 RAG 질의응답 서버

## 프로젝트 개요

EXAONE-4.0-1.2B 기반의 창고 자동화 전문 RAG 시스템입니다. AGV, AMR, WMS, 물류 자동화 관련 논문 및 문서를 검색하여 질문에 답변합니다.

## 주요 기능

- **RAG 채팅**: FAISS 벡터 검색 + EXAONE 답변 생성
- **대화형 채팅**: 세션별 대화 기록 유지
- **파이프라인 관리**: 크롤링, 텍스트 추출, 벡터DB 구축 트리거
- **시스템 모니터링**: 리소스 현황, 활동 로그, 헬스체크

## 아키텍처

```
wms/
├── main.py                     # FastAPI 서버 진입점
├── app/
│   ├── chat/                   # 채팅 API
│   │   ├── router.py           # /chat 엔드포인트
│   │   ├── service.py          # RAG 쿼리 처리
│   │   └── dto.py              # 요청/응답 모델
│   ├── pipeline/               # 파이프라인 API
│   │   ├── router.py           # /pipeline 엔드포인트
│   │   ├── service.py          # 크롤링/추출/벡터DB 관리
│   │   └── dto.py
│   ├── system/                 # 시스템 API
│   │   ├── router.py           # /system 엔드포인트
│   │   ├── service.py          # 모니터링
│   │   └── dto.py
│   └── vectorstore/            # 벡터스토어 관리
│       └── manager.py          # FAISS 로드/검색
├── _0_1_core/
│   └── rag/
│       ├── wrapper.py          # EXAONE RAG Wrapper
│       └── performance_monitor.py
├── 0_core/                     # 데이터 파이프라인
│   ├── 0_scrapers/             # 논문/뉴스 크롤링
│   ├── 1_extractors/           # PDF 텍스트 추출
│   ├── 1.5_chunk_optimizer/    # 청크 최적화
│   ├── 2_reranker/             # 재순위화
│   └── vectorDB/               # FAISS 벡터DB 구축
├── 1_data/                     # 데이터
│   ├── 0_crawled/              # 원본 논문 PDF
│   ├── 1_chunks/               # 추출 텍스트 청크
│   └── 1_chunks_optimized/     # 최적화 청크
├── 2_vecdb/
│   └── faiss_storage/          # FAISS 인덱스 저장소
└── z_llm_data/                 # LLM 학습 데이터셋
```

## 빠른 시작

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 서버 실행

#### 로컬 환경 (Ollama 권장)

```bash
# Ollama 설치 후 EXAONE 모델 pull
ollama pull hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M

# 서버 실행 (Ollama 모드)
USE_OLLAMA=true python main.py
```

#### Docker 환경

```bash
docker-compose up -d
```

### 3. API 문서 확인

서버 실행 후 브라우저에서 접속:

```
http://localhost:8000/docs
```

## API 엔드포인트

### 채팅 API

#### POST /chat

기본 RAG 채팅

```json
{
  "query": "AGV와 AMR의 차이가 뭐야?",
  "context_count": 5,
  "temperature": 0.1
}
```

응답:
```json
{
  "query": "AGV와 AMR의 차이가 뭐야?",
  "response": "AGV는 정해진 경로를 따라 이동하는 반면...",
  "sources": [
    {
      "content": "AGV는 자동 유도 차량으로...",
      "score": 0.85,
      "metadata": {}
    }
  ],
  "performance": {
    "inference_time_ms": 1234,
    "tokens_per_second": 12.5
  },
  "timestamp": "2025-10-15T12:34:56",
  "success": true
}
```

#### POST /chat/conversational

대화형 채팅 (세션 기억)

```json
{
  "query": "그것의 장점은?",
  "session_id": "user_123",
  "context_count": 5
}
```

#### GET /chat/history

채팅 히스토리 조회

```
GET /chat/history?limit=20
```

### 파이프라인 API

#### GET /pipeline/status

파이프라인 전체 현황

```json
{
  "scrapers": {
    "total_papers": 465,
    "total_news": 1
  },
  "chunks": {
    "total_chunks": 12345,
    "optimized_chunks": 11234
  },
  "vectordb": {
    "total_documents": 11234,
    "index_size_mb": 150.5
  }
}
```

#### POST /pipeline/trigger/scraper

크롤링 실행 트리거 (백그라운드)

#### POST /pipeline/trigger/rebuild-vectordb

벡터DB 재구축 트리거 (백그라운드)

### 시스템 API

#### GET /system/status

시스템 리소스 현황

```json
{
  "cpu_percent": 45.2,
  "memory_percent": 68.1,
  "disk_percent": 52.3,
  "gpu_available": true,
  "gpu_memory_used_mb": 2048
}
```

#### GET /system/health

헬스체크

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "vectordb": true
  }
}
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `USE_OLLAMA` | `false` | Ollama 사용 (권장) |
| `USE_VLLM` | `false` | vLLM 사용 |
| `OLLAMA_MODEL` | `hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M` | Ollama 모델 |
| `VLLM_URL` | `http://localhost:8080/v1` | vLLM 서버 URL |
| `PORT` | `8000` | 서버 포트 |

## 실행 모드

### 1. Ollama (권장)

가장 쉽고 GPU 부담이 적음

```bash
USE_OLLAMA=true python main.py
```

### 2. vLLM

고성능 추론 서버 (별도 실행 필요)

```bash
# vLLM 서버 실행
vllm serve LGAI-EXAONE/EXAONE-4.0-1.2B-Instruct

# API 서버 실행
USE_VLLM=true python main.py
```

### 3. Transformers

직접 모델 로드 (GPU 메모리 필요)

```bash
python main.py
```

## 데이터 파이프라인

### 1. 논문 크롤링

```bash
python 0_core/0_scrapers/paper_scraper.py
```

ArXiv, Semantic Scholar에서 창고 자동화 관련 논문 수집

### 2. 텍스트 추출

```bash
python 0_core/1_extractors/text_extractor.py
```

PDF에서 텍스트 추출 및 청킹

### 3. 청크 최적화

```bash
python 0_core/1.5_chunk_optimizer/chunk_optimizer.py
```

중복 제거, 품질 필터링

### 4. 벡터DB 구축

```bash
python 0_core/vectorDB/faiss_builder.py
```

FAISS 벡터 인덱스 생성

## 주요 기술 스택

### 백엔드

- **FastAPI**: REST API 서버
- **Uvicorn**: ASGI 서버

### RAG

- **EXAONE-4.0-1.2B**: LG AI 한국어 LLM
- **LangChain**: RAG 파이프라인
- **FAISS**: 벡터 검색 (HNSW 인덱스)
- **jhgan/ko-sroberta-multitask**: 한국어 임베딩

### 데이터 파이프라인

- **ArXiv API**: 논문 수집
- **PyMuPDF, pdfplumber**: PDF 파싱
- **BeautifulSoup**: 웹 스크래핑

### 모니터링

- **psutil**: 시스템 리소스 모니터링
- **performance_monitor.py**: RAG 성능 추적

## 벡터스토어 구조

### FAISS 저장소 구조

```
2_vecdb/faiss_storage/
├── warehouse_automation_knowledge.index  # FAISS 인덱스
├── documents.json                        # 원본 문서
├── metadata.json                         # 메타데이터
└── config.json                           # 설정
```

### config.json

```json
{
  "dimension": 768,
  "total_documents": 11234,
  "embedding_model": "jhgan/ko-sroberta-multitask",
  "index_type": "HNSW",
  "index_name": "warehouse_automation_knowledge",
  "created_at": "2025-10-15T12:00:00"
}
```

## 테스트

### 단위 테스트

```bash
pytest 3_test/
```

### API 테스트

```bash
# 서버 실행 후
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "AGV란?", "context_count": 3}'
```

## 로깅

### 로그 파일

- `_0_1_core/rag/performance_log.json`: RAG 성능 로그
- `1_data/0_crawled/scraping_report.txt`: 크롤링 리포트

### 로그 레벨

```python
logging.basicConfig(level=logging.INFO)
```

## 개발 정보

- **프로젝트**: 창고 자동화 RAG 시스템
- **버전**: 1.0.0
- **기술 스택**: FastAPI, LangChain, FAISS, EXAONE
- **라이선스**: MIT

## 참고 자료

- [EXAONE 모델](https://huggingface.co/LGAI-EXAONE)
- [FastAPI 문서](https://fastapi.tiangolo.com)
- [LangChain 문서](https://python.langchain.com)
- [FAISS 문서](https://github.com/facebookresearch/faiss)