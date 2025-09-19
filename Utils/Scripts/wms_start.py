import os
import json
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler('wms_start.log'), 
                              logging.StreamHandler()])

logger = logging.getLogger(__name__)

def create_wms_structure():
    """WMS 프로젝트 폴더 구조 생성"""
    
    logging.info("Creating WMS project structure...")
    
    # WMS 폴더 구조
    wms_folders = [
        "WMS/Papers/ArXiv",
        "WMS/Papers/IEEE", 
        "WMS/Papers/SemanticScholar",
        "WMS/Papers/GoogleScholar",
        "WMS/ProcessedData",
        "WMS/Analysis",
        "WMS/Tools"
    ]
    
    # 폴더 생성
    for folder in wms_folders:
        os.makedirs(folder, exist_ok=True)
        logging.info(f"✓ Created: {folder}")
    
    # README 파일들 생성
    wms_readmes = {
        "WMS/README.md": """# WMS (Warehouse Management System) Research Collection

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
""",
        "WMS/Papers/README.md": "# 논문 저장 폴더\n\n각 플랫폼별로 수집된 WMS 관련 논문들이 저장됩니다.",
        "WMS/ProcessedData/README.md": "# 처리된 데이터\n\n논문에서 추출된 텍스트, 키워드, 메타데이터가 저장됩니다.",
        "WMS/Analysis/README.md": "# 분석 결과\n\n연구 동향, 토픽 모델링, 인용 네트워크 분석 결과가 저장됩니다.",
        "WMS/Tools/README.md": "# 분석 도구\n\n데이터 수집, 처리, 분석을 위한 Python 스크립트들이 저장됩니다."
    }
    
    for file_path, content in wms_readmes.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"✓ Created: {file_path}")
    
    # requirements.txt 생성
    wms_requirements = """arxiv
scholarly
requests
beautifulsoup4
pandas
numpy
matplotlib
seaborn
networkx
nltk
scikit-learn
PyPDF2
bibtexparser"""
    
    with open("WMS/requirements.txt", 'w') as f:
        f.write(wms_requirements)
    logging.info("✓ Created: WMS/requirements.txt")
    
    # 샘플 메타데이터 파일 생성
    sample_metadata = {
        "collection_info": {
            "created_date": "2024-01-01",
            "total_papers": 0,
            "platforms": ["ArXiv", "IEEE", "SemanticScholar", "GoogleScholar"],
            "keywords": ["warehouse management", "WMS", "automation", "AGV", "robotics"]
        },
        "papers": []
    }
    
    with open("WMS/Papers/ArXiv/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(sample_metadata, f, ensure_ascii=False, indent=2)
    logging.info("✓ Created: WMS/Papers/ArXiv/metadata.json")
    
    logging.info("WMS project structure created successfully!\n")