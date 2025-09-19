#!/usr/bin/env python3
"""
WMS (창고 관리 시스템) 통합 런처
==========================

새로 정리된 구조에서 WMS 시스템의 주요 기능들을 실행할 수 있는 통합 런처입니다.

키워드 기반 새 구조:
- Core/Scrapers/: 논문 수집기 (paper_scraper.py)
- Core/Extractors/: 텍스트 추출기 (text_extractor.py)  
- Core/VectorDB/: 벡터DB 구축기 (faiss_builder.py)
- Core/Analyzers/: 분석 도구들 (citation_analyzer.py, trend_visualizer.py)
- Data/: 모든 데이터 (Papers, Processed, VectorDB, Analysis)
- Utils/Config/: 설정 파일들 (enhanced_wms_keywords.py)

작성자: WMS 개발팀
날짜: 2025년 9월 19일
"""

import sys
import os
from pathlib import Path
import argparse

# 현재 디렉토리를 기준으로 경로 설정
current_dir = Path(__file__).parent
core_dir = current_dir / "Core"
data_dir = current_dir / "Data"
utils_dir = current_dir / "Utils"

# Python 경로에 Core 디렉토리 추가
sys.path.insert(0, str(core_dir))


def run_paper_scraper():
    """논문 수집기를 실행합니다."""
    print("📚 창고 자동화 시스템 논문 수집기 실행 중...")
    
    scraper_path = core_dir / "Scrapers" / "paper_scraper.py"
    papers_output = data_dir / "Papers"
    
    if scraper_path.exists():
        # 경로 업데이트하여 실행
        cmd = f'python "{scraper_path}" --output-dir "{papers_output}"'
        print(f"실행: {cmd}")
        os.system(cmd)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {scraper_path}")


def run_text_extractor():
    """텍스트 추출기를 실행합니다."""
    print("📄 창고 자동화 시스템 텍스트 추출기 실행 중...")
    
    extractor_path = core_dir / "Extractors" / "text_extractor.py"
    papers_input = data_dir / "Papers"
    processed_output = data_dir / "Processed"
    
    if extractor_path.exists():
        cmd = f'python "{extractor_path}" --papers-dir "{papers_input}" --output-dir "{processed_output}"'
        print(f"실행: {cmd}")
        os.system(cmd)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {extractor_path}")


def run_faiss_builder():
    """Faiss 벡터 데이터베이스 구축기를 실행합니다."""
    print("🚀 창고 자동화 시스템 Faiss 벡터DB 구축기 실행 중...")
    
    builder_path = core_dir / "VectorDB" / "faiss_builder.py"
    processed_input = data_dir / "Processed"
    vector_output = data_dir / "VectorDB"
    
    if builder_path.exists():
        cmd = f'python "{builder_path}" --processed-data "{processed_input}" --vector-db "{vector_output}"'
        print(f"실행: {cmd}")
        os.system(cmd)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {builder_path}")


def run_citation_analyzer():
    """인용 분석기를 실행합니다."""
    print("🔍 창고 자동화 시스템 인용 분석기 실행 중...")
    
    analyzer_path = core_dir / "Analyzers" / "citation_analyzer.py"
    papers_input = data_dir / "Papers"
    analysis_output = data_dir / "Analysis"
    
    if analyzer_path.exists():
        cmd = f'python "{analyzer_path}" --papers-dir "{papers_input}" --output-dir "{analysis_output}"'
        print(f"실행: {cmd}")
        os.system(cmd)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {analyzer_path}")


def run_trend_visualizer():
    """트렌드 시각화기를 실행합니다."""
    print("📊 창고 자동화 시스템 트렌드 시각화기 실행 중...")
    
    visualizer_path = core_dir / "Analyzers" / "trend_visualizer.py"
    processed_input = data_dir / "Processed"
    analysis_output = data_dir / "Analysis"
    
    if visualizer_path.exists():
        cmd = f'python "{visualizer_path}" --processed-data "{processed_input}" --output-dir "{analysis_output}"'
        print(f"실행: {cmd}")
        os.system(cmd)
    else:
        print(f"❌ 파일을 찾을 수 없습니다: {visualizer_path}")


def run_full_pipeline():
    """전체 파이프라인을 순서대로 실행합니다."""
    print("🔄 창고 자동화 시스템 전체 파이프라인 실행 중...")
    print("=" * 60)
    
    # 1단계: 논문 수집
    print("1단계: 논문 수집")
    run_paper_scraper()
    
    print("\n" + "=" * 60)
    
    # 2단계: 텍스트 추출
    print("2단계: 텍스트 추출")
    run_text_extractor()
    
    print("\n" + "=" * 60)
    
    # 3단계: 벡터DB 구축
    print("3단계: Faiss 벡터DB 구축")
    run_faiss_builder()
    
    print("\n" + "=" * 60)
    
    # 4단계: 분석
    print("4단계: 인용 분석")
    run_citation_analyzer()
    
    print("\n" + "=" * 60)
    
    # 5단계: 시각화
    print("5단계: 트렌드 시각화")
    run_trend_visualizer()
    
    print("\n" + "=" * 60)
    print("🎉 전체 파이프라인 완료!")


def show_structure():
    """새로운 프로젝트 구조를 보여줍니다."""
    print("📁 새로 정리된 WMS 프로젝트 구조:")
    print("=" * 50)
    print("""
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
│   ├── Processed/              # 처리된 텍스트
│   ├── VectorDB/               # 벡터 데이터베이스
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
└── 🚀 wms_launcher.py         # 통합 런처 (이 파일)
    """)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="WMS 창고 자동화 시스템 통합 런처")
    parser.add_argument("action", choices=[
        "scrape", "extract", "build", "analyze", "visualize", 
        "full", "structure"
    ], help="실행할 작업")
    
    args = parser.parse_args()
    
    print("🏭 WMS 창고 자동화 시스템 통합 런처")
    print("=" * 50)
    
    if args.action == "scrape":
        run_paper_scraper()
    elif args.action == "extract":
        run_text_extractor()
    elif args.action == "build":
        run_faiss_builder()
    elif args.action == "analyze":
        run_citation_analyzer()
    elif args.action == "visualize":
        run_trend_visualizer()
    elif args.action == "full":
        run_full_pipeline()
    elif args.action == "structure":
        show_structure()


if __name__ == "__main__":
    main()
