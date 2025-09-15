# WMS Faiss 시스템 VSS-AI-API-dev 통합 가이드

## 🎯 통합 개요

우리가 만든 WMS Faiss 벡터 데이터베이스를 VSS-AI-API-dev 프로젝트에 통합하여 WMS 연구 논문 검색 및 질의응답 기능을 추가합니다.

## 📁 파일 구조

### 1. VSS-AI-API-dev에 추가할 파일들

```
VSS-AI-API-dev/
├── app/
│   └── wms_research/                    # 새로 추가
│       └── v1/
│           ├── __init__.py
│           ├── model/
│           │   ├── __init__.py
│           │   └── dto/
│           │       ├── __init__.py
│           │       ├── request_dto.py
│           │       └── response_dto.py
│           ├── repository/
│           │   ├── __init__.py
│           │   └── wms_faiss_repository.py
│           ├── router/
│           │   ├── __init__.py
│           │   └── wms_research_router.py
│           └── service/
│               ├── __init__.py
│               └── wms_research_service.py
├── resource/
│   └── wms_knowledge/                   # 새로 추가
│       └── faiss_storage/               # 우리가 만든 Faiss 시스템 복사
│           ├── wms_knowledge.index
│           ├── documents.json
│           ├── metadata.json
│           └── config.json
└── core/
    └── wms_vector_store.py              # 새로 추가 (옵션)
```

## 🚀 배포 단계

### 단계 1: 파일 복사

```bash
# 1. WMS Faiss 데이터 복사
cp -r C:/Users/user/pjt/wms/WMS/VectorDB/faiss_storage C:/Users/user/VSS-AI-API-dev/resource/wms_knowledge/

# 2. 통합 코드 복사
cp -r WMS/Tools/vss_integration/* C:/Users/user/VSS-AI-API-dev/app/wms_research/v1/
```

### 단계 2: 환경 설정

```bash
# .env 파일에 추가
echo "WMS_FAISS_STORAGE_PATH=./resource/wms_knowledge/faiss_storage" >> .env
```

### 단계 3: 의존성 설치

```bash
# requirements.txt에 추가
pip install faiss-cpu torch transformers langchain-huggingface
```

### 단계 4: 메인 앱에 라우터 등록

```python
# main.py 또는 app.py에 추가
from app.wms_research.v1.router.wms_research_router import router as wms_research_router

app.include_router(wms_research_router, prefix="/api/v1")
```

## 🔧 설정 파일

### 1. 환경 변수 (.env)

```bash
# WMS Faiss 설정
WMS_FAISS_STORAGE_PATH=./resource/wms_knowledge/faiss_storage

# 기존 설정들...
OPENAI_API_KEY=your_api_key
VSS_FUNCTION_PATH=./resource/vss_functions
VSS_FUNCTION_VECTOR_PATH=./resource/vss_function_vector
```

### 2. 로깅 설정

```python
# logging.conf 또는 main.py에 추가
import logging

logging.getLogger("app.wms_research").setLevel(logging.INFO)
```

## 📡 API 엔드포인트

### 기본 검색

```http
POST /api/v1/wms-research/search
Content-Type: application/json

{
    "query": "창고 자동화 시스템",
    "top_k": 5
}
```

### RAG 질의응답

```http
POST /api/v1/wms-research/ask
Content-Type: application/json

{
    "question": "AGV와 AMR의 차이점은 무엇인가요?",
    "top_k": 3
}
```

### 연구 동향 분석

```http
GET /api/v1/wms-research/trends?topic=로봇%20피킹&top_k=10
```

### 데이터베이스 통계

```http
GET /api/v1/wms-research/stats
```

### 논문 상세 정보

```http
GET /api/v1/wms-research/paper/047_Fast_Autonomous_Flight_in_Warehouses_for_Inventory.pdf
```

## 🧪 테스트

### 1. 헬스체크

```bash
curl http://localhost:8000/api/v1/wms-research/health
```

### 2. 기본 검색 테스트

```bash
curl -X POST http://localhost:8000/api/v1/wms-research/search \
  -H "Content-Type: application/json" \
  -d '{"query": "warehouse automation", "top_k": 3}'
```

### 3. 질의응답 테스트

```bash
curl -X POST http://localhost:8000/api/v1/wms-research/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "창고에서 로봇이 어떻게 사용되나요?"}'
```

## 🔄 기존 시스템과의 통합

### 방안 1: 독립적 운영 (추천)

- WMS Research API는 `/wms-research` 경로에서 독립적으로 운영
- 기존 VSS Bot API와 분리되어 서로 영향 없음
- 각각의 전문 분야에 특화된 서비스 제공

### 방안 2: VSS Bot에 기능 추가

```python
# vss_bot_router.py에 추가
@router.post("/query-wms-research")
async def query_wms_research(
    request: QueryWithLlmDto = Body(...),
    service: VssBotService = Depends(get_vss_bot_service)
):
    # WMS 연구 검색 기능 추가
    pass
```

### 방안 3: 하이브리드 검색

```python
# 통합 검색 서비스
class UnifiedSearchService:
    def __init__(self, vss_store, wms_store):
        self.vss_store = vss_store
        self.wms_store = wms_store
    
    def unified_search(self, query: str, search_type: str = "auto"):
        if search_type == "auto":
            # 쿼리 분석하여 자동 판단
            if self._is_function_query(query):
                return self.vss_store.similarity_search(query)
            else:
                return self.wms_store.search(query)
```

## 🛠️ 유지보수

### 1. 데이터 업데이트

```python
# 새로운 논문 추가 시
python WMS/Tools/faiss_builder.py --action build --processed-data new_papers/
```

### 2. 인덱스 재구축

```python
# 전체 인덱스 재구축
python WMS/Tools/faiss_builder.py --action build --force-rebuild
```

### 3. 성능 모니터링

```python
# 검색 성능 테스트
python WMS/Tools/inspect_faiss_index.py
```

## 🚨 주의사항

### 1. 메모리 사용량

- Faiss 인덱스: ~7MB
- 임베딩 모델: ~500MB (첫 로딩 시)
- 총 메모리 사용량: ~1GB

### 2. 초기 로딩 시간

- 첫 번째 요청 시 임베딩 모델 로딩으로 10-30초 소요
- 이후 요청은 3-5ms로 빠름

### 3. GPU 사용

- CUDA 사용 가능 시 자동으로 GPU 활용
- CPU만 사용 시에도 정상 동작

## 📈 확장 계획

### 1. 실시간 논문 추가

```python
@router.post("/add-paper")
async def add_new_paper(paper_file: UploadFile):
    # 새 논문을 실시간으로 인덱스에 추가
    pass
```

### 2. 사용자 피드백

```python
@router.post("/feedback")
async def submit_feedback(search_id: str, rating: int, comment: str):
    # 검색 결과에 대한 사용자 피드백 수집
    pass
```

### 3. 개인화 검색

```python
@router.post("/personalized-search")
async def personalized_search(user_id: str, query: str):
    # 사용자 검색 이력 기반 개인화 검색
    pass
```

이제 VSS-AI-API-dev에 WMS Faiss 시스템이 완전히 통합되어 강력한 연구 논문 검색 및 분석 기능을 제공할 수 있습니다! 🚀
