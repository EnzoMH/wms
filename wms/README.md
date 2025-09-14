# WMS 산업용 협동로봇 전문 시스템 🤖

**AMR, AGV, CNV, RTV 등 산업용 협동로봇 전문지식을 자동으로 수집하고 벡터DB를 구축하는 시스템**

## 🎯 **시스템 목적**

본 시스템은 **본 프로젝트의 Langchain Application에 물류/WMS 전문지식을 주입**하기 위한 데이터 파이프라인입니다.

- **일반적인 창고 관리** → **산업용 협동로봇 전문 시스템**
- **범용 자동화 키워드** → **AMR, CNV, RTV, WCS, WES 등 실무 전문 용어**
- **단순 벡터DB** → **스마트팩토리 Industry 4.0 통합 지식베이스**

## 🚀 **빠른 시작**

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행 (권장)

```bash
python main.py
```

### 3. API 서버 모드

```bash
python main.py --mode api --port 8000
```

## 📋 **실행 결과**

실행 완료 후 `output/industrial_robot_vectordb/` 폴더에 다음 파일들이 생성됩니다:

- **`wms_knowledge.index`**: Faiss 벡터 인덱스 (본 프로젝트에서 사용)
- **`documents.json`**: 전문 논문 텍스트 데이터
- **`metadata.json`**: 논문 메타데이터 (출처, 키워드 등)
- **`migration_info.json`**: 마이그레이션 상세 정보

## 🔧 **전문 키워드 체계**

### 🤖 **로봇 시스템**

- **AMR** (Autonomous Mobile Robot)
- **AGV** (Automated Guided Vehicle)
- **CNV** (Conveyor System)
- **RTV** (Return to Vendor)
- **Collaborative Robot (Cobot)**

### 🏭 **제어 시스템**

- **WCS** (Warehouse Control System)
- **WES** (Warehouse Execution System)
- **MES** (Manufacturing Execution System)
- **SCADA**, **PLC**

### 🏗️ **스마트팩토리**

- **Industry 4.0**
- **Digital Twin**
- **Cyber Physical System**
- **IoT Integration**

## 📊 **파이프라인 단계**

```
📡 1단계: 전문 키워드 크롤링
    ├── AMR, AGV, CNV 등 로봇 시스템
    ├── WCS, WES, MES 등 제어 시스템
    └── 스마트팩토리 통합 기술

📝 2단계: 전문 용어 추출
    ├── 9개 전문 카테고리 분류
    ├── 산업용 협동로봇 용어 특화
    └── 스마트 청킹 (1000자 단위)

🗄️ 3단계: 벡터DB 구축
    ├── ChromaDB 구축 (384차원)
    ├── Faiss 변환 (호환성 확인)
    └── 본 프로젝트 통합 준비

✅ 4단계: 시스템 검증
    ├── 파일 생성 확인
    ├── 전문 용어 통계
    └── 통합 가이드 제공
```

## 🛠️ **본 프로젝트 통합**

### 1. 파일 복사

```bash
# 생성된 벡터DB 파일들을 본 프로젝트로 복사
cp output/industrial_robot_vectordb/* /path/to/main/project/vectordb/
```

### 2. Faiss 로드 코드 (본 프로젝트에서)

```python
import faiss
import json

# WMS 전문지식 인덱스 로드
index = faiss.read_index('vectordb/wms_knowledge.index')

# 문서와 메타데이터 로드
with open('vectordb/documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
with open('vectordb/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 검색 예시
query_vector = ...  # 쿼리 벡터
distances, indices = index.search(query_vector, k=5)
```

### 3. 전문 검색 예시

- "AMR fleet management optimization"
- "collaborative robot safety standards"
- "WCS WES integration architecture"
- "digital twin warehouse simulation"
- "slotting optimization algorithm"

## 📁 **프로젝트 구조**

```
wms/
├── main.py                           # 🚀 메인 실행 파일
├── requirements.txt                  # 📦 필수 패키지
├── WMS/
│   ├── Tools/
│   │   ├── paper_scraper.py         # 📡 논문 크롤링 (고도화된 키워드)
│   │   ├── text_extractor.py        # 📝 텍스트 추출 (전문 용어)
│   │   ├── chromadb_builder.py      # 🗄️ ChromaDB 구축
│   │   ├── chroma_to_faiss_migrator.py  # 🔄 Faiss 변환
│   │   ├── enhanced_wms_keywords.py # 🎯 전문 키워드 체계
│   │   └── vector_dimension_analyzer.py # 📊 차원 호환성 분석
│   ├── Papers/                      # 📚 수집된 논문들
│   ├── ProcessedData/               # 📄 처리된 텍스트
│   └── VectorDB/                    # 🗃️ ChromaDB 저장소
└── output/
    └── industrial_robot_vectordb/   # 🎯 최종 결과물
```

## 🎉 **최종 결과**

이 시스템을 실행하면:

1. **148+ 전문 논문** 자동 수집 (AMR, AGV, CNV, RTV 중심)
2. **9개 전문 카테고리**로 용어 분류 및 추출
3. **산업용 협동로봇 전문 벡터DB** 구축
4. **본 프로젝트 즉시 통합 가능**한 Faiss 인덱스 생성

**결과**: 본 프로젝트의 Langchain Application이 **물류/WMS 분야의 진정한 전문 AI**로 변신! 🎯

## 📞 **문의사항**

- 시스템 관련: WMS 연구팀
- 기술 지원: 비젼스페이스 개발팀
- 버전: 2.0.0 (산업용 협동로봇 전문화)
