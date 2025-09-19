# WMS (Warehouse Management System) Research Collection

이 프로젝트는 창고 관리 시스템 관련 학술 자료를 수집하고 분석하는 프로젝트입니다.

## 폴더 구조
- **Papers/**: 각 플랫폼별 논문 수집
  - ArXiv/: arXiv 논문들
  - IEEE/: IEEE Xplore 논문들  
  - SemanticScholar/: Semantic Scholar 논문들
  - GoogleScholar/: Google Scholar 논문들
- **ProcessedData/**: 텍스트 추출 및 키워드 분석 결과
- **Analysis/**: 연구 동향 및 분석 결과
- **Tools/**: 데이터 수집 및 분석 도구들

## 사용법
1. `pip install -r requirements.txt`
2. `python Tools/paper_scraper.py`로 논문 수집 시작
3. `python Tools/text_extractor.py`로 텍스트 추출
4. `python Tools/citation_analyzer.py`로 인용 분석

## 참고사항
- 논문 다운로드 시 저작권 정책을 준수해주세요
- 메타데이터는 JSON 형식으로 저장됩니다
