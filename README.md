# WMS 창고 자동화 시스템

**키워드 기반으로 새롭게 정리된 창고 자동화 시스템 (AGV, EMS, RTV, CNV) 연구 플랫폼**

## 🎯 프로젝트 개요

AGV(자동 유도 차량), EMS(전동 모노레일 시스템), RTV(로봇 운송 차량), CNV(컨베이어) 등 창고 자동화 시스템에 특화된 연구 논문 수집, 처리, 분석 및 질의응답 시스템입니다.

## 📁 새로운 프로젝트 구조

```
📦 WMS (Root)
├── 🔧 Core/                    # 핵심 처리 도구들
│   ├── Scrapers/               # 논문 수집
│   │   └── paper_scraper.py    # 논문 스크래퍼
│   ├── Extractors/             # 텍스트 처리
│   │   └── text_extractor.py   # 텍스트 추출기
│   ├── VectorDB/               # 벡터 데이터베이스
│   │   ├── faiss_builder.py    # Faiss DB 구축기
│   │   └── advanced_rag.py     # RAG 시스템
│   └── Analyzers/              # 분석 도구
│       ├── citation_analyzer.py # 인용 분석
│       └── trend_visualizer.py  # 트렌드 시각화
│
├── 📊 Data/                    # 모든 데이터
│   ├── Papers/                 # 원본 논문들
│   │   ├── ArXiv/             # arXiv 논문들
│   │   ├── IEEE/              # IEEE 논문들
│   │   ├── SemanticScholar/   # Semantic Scholar
│   │   └── GoogleScholar/     # Google Scholar
│   ├── Processed/              # 처리된 텍스트 청크
│   ├── VectorDB/               # Faiss 벡터 데이터베이스
│   └── Analysis/               # 분석 결과
│
├── ⚙️ Utils/                   # 유틸리티
│   ├── Config/                 # 설정 파일
│   │   └── enhanced_wms_keywords.py
│   └── Scripts/                # 실행 스크립트
│
├── 📦 Legacy/                  # 이전 구조 백업
│   ├── WMS_original/           # 기존 WMS 폴더
│   └── WMS_duplicate/          # 중복 구조
│
└── 🚀 wms_launcher.py         # 통합 런처
```

## 🚀 빠른 시작

### 1. 전체 파이프라인 실행
```bash
python wms_launcher.py full
```

### 2. 개별 단계 실행
```bash
# 논문 수집
python wms_launcher.py scrape

# 텍스트 추출
python wms_launcher.py extract

# Faiss 벡터DB 구축
python wms_launcher.py build

# 인용 분석
python wms_launcher.py analyze

# 트렌드 시각화
python wms_launcher.py visualize
```

### 3. 프로젝트 구조 확인
```bash
python wms_launcher.py structure
```

## 🔧 핵심 구성 요소

### Core/Scrapers/paper_scraper.py
- **기능**: ArXiv, IEEE, Semantic Scholar에서 창고 자동화 관련 논문 수집
- **키워드**: AGV, EMS, RTV, CNV, 경로 최적화, 스마트팩토리 등
- **출력**: `Data/Papers/` 디렉토리에 PDF와 메타데이터 저장

### Core/Extractors/text_extractor.py
- **기능**: PDF 논문에서 텍스트 추출 및 스마트 청킹
- **처리**: PyMuPDF, pdfplumber, PyPDF2를 순서대로 시도
- **출력**: `Data/Processed/` 디렉토리에 JSON 청크 파일 생성

### Core/VectorDB/faiss_builder.py
- **기능**: 처리된 텍스트를 Faiss 벡터 데이터베이스로 변환
- **임베딩**: 한국어 특화 `jhgan/ko-sroberta-multitask` 모델 사용
- **출력**: `Data/VectorDB/` 디렉토리에 검색 가능한 벡터DB 생성

### Core/Analyzers/
- **citation_analyzer.py**: 논문 간 인용 관계 네트워크 분석
- **trend_visualizer.py**: AGV/EMS/RTV/CNV 트렌드 시각화

## 🎯 특화 키워드

### AGV (Automated Guided Vehicle)
- AGV path planning, fleet management
- Multi-AGV coordination, SLAM navigation
- Collision avoidance, scheduling optimization

### EMS (Electric Monorail System)  
- Rail-based picking robot
- Overhead rail robot system
- Ceiling-mounted picking robot

### RTV (Robotic Transfer Vehicle)
- Robotic transfer vehicle
- Autonomous transfer robot  
- RTV material handling

### CNV (Conveyor Systems)
- Intelligent conveyor system
- Smart conveyor belt
- Adaptive conveyor network

### 경로 최적화
- A* algorithm warehouse
- Path optimization warehouse
- Dynamic path planning factory

## 📋 요구 사항

```bash
pip install arxiv scholarly bibtexparser requests
pip install PyMuPDF pdfplumber PyPDF2 
pip install nltk scikit-learn wordcloud matplotlib pandas
pip install faiss-cpu langchain-huggingface torch
```

## 🔍 검색 및 질의응답

Faiss 벡터DB 구축 후:

```python
from Core.VectorDB.faiss_builder import WarehouseAutomationFaissBuilder

builder = WarehouseAutomationFaissBuilder()
builder.load_existing_index()

# AGV 경로 계획 관련 논문 검색
results = builder.test_search("AGV 경로 계획", top_k=5)
```

## 📊 분석 결과

- **인용 네트워크**: 논문 간 관계 그래프
- **트렌드 분석**: 시간별 기술 발전 추이  
- **키워드 클러스터링**: 연구 주제 분류
- **성능 벤치마크**: 알고리즘 비교

## 🔄 데이터 흐름

1. **수집**: 학술 데이터베이스 → `Data/Papers/`
2. **추출**: PDF → 텍스트 청크 → `Data/Processed/`
3. **임베딩**: 텍스트 → 벡터 → `Data/VectorDB/`
4. **분석**: 벡터DB → 인사이트 → `Data/Analysis/`

## 🛠️ 개발자 정보

- **프로젝트**: 창고 자동화 시스템 연구 플랫폼
- **키워드 중심 재구조화**: 2025년 9월 19일
- **기술 스택**: Python, Faiss, LangChain, NLTK, PyMuPDF

## 📝 참고사항

- **Legacy/**: 기존 구조의 백업이 보관되어 있습니다
- **경로 업데이트**: 모든 스크립트는 새로운 구조에 맞게 경로가 수정되었습니다  
- **통합 런처**: `wms_launcher.py`로 모든 기능을 쉽게 실행할 수 있습니다

## 🎉 새로운 구조의 장점

1. **키워드 기반 분류**: 기능별로 명확한 구분
2. **데이터 통합**: 모든 데이터가 `Data/` 하위에 집중
3. **경로 일관성**: 상대 경로를 통한 이식성 향상
4. **유지보수성**: 각 모듈의 독립성 보장
5. **확장성**: 새로운 분석 도구 추가 용이

