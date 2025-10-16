#!/usr/bin/env python3
"""
창고 자동화 시스템 논문 수집기
========================

ArXiv에서 AGV, EMS, RTV, CNV 등
창고 자동화 및 스마트팩토리 관련 연구 논문을 수집하는 도구입니다.

작성자: 신명호
날짜: 2025년 10월 13일
버전: 1.1.0
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
import logging

# ArXiv 라이브러리 임포트
try:
    import arxiv
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("실행: pip install arxiv")
    exit(1)


class WarehouseAutomationPaperScraper:
    """ArXiv에서 창고 자동화 시스템(AGV, EMS, RTV, CNV) 관련 연구 논문을 수집하는 메인 클래스입니다."""
    
    def __init__(self, output_dir: str = "../../1_data/0_crawled"):
        """
        논문 수집기를 초기화합니다.
        
        Args:
            output_dir: 다운로드된 논문과 메타데이터를 저장할 디렉토리
        """
        self.output_dir = output_dir
        self.setup_logging()
        self.setup_directories()
        
        # 중복 방지를 위한 기존 논문 정보 로드
        self.existing_papers = self.load_existing_papers()
        
        self.warehouse_automation_keywords = [ 
            # === AGV (Automated Guided Vehicle) 관련 ===
            "automated guided vehicle",
            "AGV path planning",
            "AGV fleet management", 
            "AGV navigation system",
            "AGV collision avoidance",
            "AGV scheduling optimization",
            "multi-AGV coordination",
            "AGV SLAM navigation",
            
            # === EMS (Electric Monorail System) 관련 ===
            "rail-based picking robot",
            "overhead rail robot system",
            "EMS picking automation",
            "rail-guided robot warehouse",
            "ceiling-mounted picking robot",
            "rail robot material handling",
            "automated picking rail system",
            "overhead crane robot picking",
            "rail-based storage retrieval",
            "gantry robot warehouse",
            
            # === RTV (Robotic Transfer Vehicle) 관련 ===
            "robotic transfer vehicle",
            "RTV material handling",
            "automated material transport",
            "robotic logistics system",
            "autonomous transfer robot",
            "RTV warehouse automation",
            
            # === CNV (Conveyor) 시스템 관련 ===
            "intelligent conveyor system",
            "smart conveyor belt",
            "automated conveyor control",
            "conveyor sorting system",
            "adaptive conveyor network",
            "conveyor AGV integration",
            
            # === 경로 최적화 및 A* 알고리즘 ===
            "A* algorithm warehouse",
            "path optimization warehouse",
            "warehouse route planning",
            "dynamic path planning factory",
            "multi-robot path planning",
            "warehouse navigation algorithm",
            "shortest path warehouse logistics",
            "Dijkstra algorithm warehouse",
            "RRT path planning warehouse",
            "genetic algorithm warehouse optimization",
            
            # === 창고 최적화 동선 ===
            "warehouse layout optimization",
            "optimal warehouse design",
            "warehouse traffic flow optimization",
            "storage location optimization",
            "warehouse space utilization",
            "picking route optimization",
            "warehouse congestion management",
            "material flow optimization",
            
            # === 스마트팩토리 통합 시스템 ===
            "smart factory automation",
            "Industry 4.0 robotics",
            "intelligent manufacturing system",
            "cyber-physical system factory",
            "IoT smart factory",
            "digital twin warehouse",
            "smart factory logistics",
            "autonomous manufacturing system",
            
            # === 로봇 협업 및 조정 ===
            "multi-robot coordination",
            "robot fleet management",
            "collaborative robot system",
            "swarm robotics warehouse", 
            "distributed robot control",
            "robot task allocation",
            
            # === 실시간 제어 및 모니터링 ===
            "real-time warehouse monitoring",
            "predictive maintenance AGV",
            "warehouse digital twin",
            "smart sensor warehouse",
            "RFID warehouse tracking",
            "computer vision warehouse",
            
            # === 머신러닝 및 AI 최적화 ===
            "machine learning warehouse optimization",
            "AI-driven logistics",
            "reinforcement learning AGV",
            "neural network path planning",
            "deep learning warehouse management",
            "predictive analytics warehouse"
        ]
        
        # 도메인 키워드 매핑 (분류용)
        self.domain_keywords = {
            'AGV': ['agv', 'automated guided vehicle', 'autonomous mobile robot', 'amr', 'mobile robot navigation'],
            'EMS': ['ems', 'rail-based', 'monorail', 'overhead rail', 'ceiling-mounted', 'gantry'],
            'RTV': ['rtv', 'robotic transfer vehicle', 'transfer robot', 'material transport'],
            'CNV': ['conveyor', 'conveyor belt', 'sorting system', 'conveyor network'],
            'PathOpt': ['path planning', 'path optimization', 'route planning', 'a* algorithm', 'dijkstra', 'navigation'],
            'SmartFactory': ['smart factory', 'industry 4.0', 'digital twin', 'iot', 'cyber-physical'],
            'AI_ML': ['machine learning', 'deep learning', 'reinforcement learning', 'neural network', 'ai-driven'],
            'Warehouse': ['warehouse layout', 'warehouse optimization', 'storage', 'picking', 'inventory']
        }
        
        # 도메인별 검색 키워드 (수집용) - 각 도메인마다 다양한 키워드 사용
        self.domain_search_keywords = {
            'AGV': [
                "automated guided vehicle",
                "AGV path planning",
                "multi-AGV coordination",
                "mobile robot warehouse",
                "autonomous mobile robot AMR"
            ],
            'PathOpt': [
                "path planning warehouse",
                "route optimization logistics",
                "A* algorithm robot",
                "warehouse navigation",
                "trajectory planning mobile robot",
                "motion planning warehouse",
                "collision avoidance warehouse"
            ],
            'SmartFactory': [
                "Industry 4.0",
                "smart factory",
                "digital twin manufacturing",
                "cyber-physical system",
                "IoT manufacturing",
                "intelligent manufacturing",
                "factory automation"
            ],
            'AI_ML': [
                "machine learning optimization",
                "reinforcement learning robot",
                "deep learning manufacturing",
                "neural network logistics",
                "AI warehouse",
                "predictive analytics factory"
            ]
        }
    
    def classify_paper_domain(self, title: str, abstract: str) -> str:
        """
        논문의 제목과 초록을 분석하여 도메인을 분류합니다.
        
        Args:
            title: 논문 제목
            abstract: 논문 초록
            
        Returns:
            도메인 라벨 (예: 'AGV', 'EMS', 'PathOpt' 등)
        """
        text = (title + " " + abstract).lower()
        
        # 각 도메인별 키워드 매칭 점수 계산
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score
        
        # 가장 높은 점수의 도메인 반환
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'General'
    
    def setup_logging(self):
        """수집기의 로깅을 설정합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('paper_scraper.log', encoding='utf-8', errors='ignore'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """논문 저장을 위한 필수 디렉토리를 생성합니다."""
        arxiv_dir = f"{self.output_dir}/ArXiv"
        os.makedirs(arxiv_dir, exist_ok=True)
        self.logger.info(f"Created directory: {arxiv_dir}")
    
    def load_existing_papers(self) -> Dict[str, set]:
        """
        기존 논문들의 정보를 로드하여 중복 방지에 사용합니다.
        
        Returns:
            기존 논문의 제목, URL, ID를 담은 딕셔너리
        """
        existing = {
            'titles': set(),
            'urls': set(), 
            'ids': set(),
            'files': set()
        }
        
        try:
            source_path = f"{self.output_dir}/ArXiv"
            
            # 메타데이터 파일 확인
            metadata_file = f"{source_path}/metadata.json"
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        papers = data.get('papers', [])
                        
                        for paper in papers:
                            # 제목 추가 (정규화)
                            title = paper.get('title', '').strip().lower()
                            if title:
                                existing['titles'].add(title)
                            
                            # URL 추가
                            url = paper.get('url') or paper.get('pdf_url')
                            if url:
                                existing['urls'].add(url)
                            
                            # ID 추가
                            paper_id = paper.get('id')
                            if paper_id:
                                existing['ids'].add(str(paper_id))
                                
                except Exception as e:
                    self.logger.warning(f"메타데이터 로드 실패 {metadata_file}: {e}")
            
            # 기존 파일들 확인
            if os.path.exists(source_path):
                for file in os.listdir(source_path):
                    if file.endswith('.pdf'):
                        existing['files'].add(file)
            
            self.logger.info(f"기존 논문 정보 로드 완료:")
            self.logger.info(f"  - 제목: {len(existing['titles'])}개")
            self.logger.info(f"  - URL: {len(existing['urls'])}개") 
            self.logger.info(f"  - ID: {len(existing['ids'])}개")
            self.logger.info(f"  - 파일: {len(existing['files'])}개")
            
        except Exception as e:
            self.logger.error(f"기존 논문 정보 로드 중 오류: {e}")
        
        return existing
    
    def is_duplicate_paper(self, paper_data: Dict) -> Tuple[bool, str]:
        """
        논문이 중복인지 확인합니다.
        
        Args:
            paper_data: 논문 메타데이터
            
        Returns:
            (is_duplicate, reason) 튜플
        """
        title = paper_data.get('title', '').strip().lower()
        url = paper_data.get('url') or paper_data.get('pdf_url')
        paper_id = paper_data.get('id')
        
        # 제목으로 중복 검사
        if title and title in self.existing_papers['titles']:
            return True, f"중복 제목: {title[:50]}..."
        
        # URL로 중복 검사
        if url and url in self.existing_papers['urls']:
            return True, f"중복 URL: {url}"
        
        # ID로 중복 검사  
        if paper_id and str(paper_id) in self.existing_papers['ids']:
            return True, f"중복 ID: {paper_id}"
        
        return False, ""
    
    def add_to_existing_papers(self, paper_data: Dict):
        """
        새로운 논문 정보를 기존 논문 목록에 추가합니다.
        
        Args:
            paper_data: 논문 메타데이터
        """
        title = paper_data.get('title', '').strip().lower()
        url = paper_data.get('url') or paper_data.get('pdf_url')
        paper_id = paper_data.get('id')
        
        if title:
            self.existing_papers['titles'].add(title)
        if url:
            self.existing_papers['urls'].add(url)
        if paper_id:
            self.existing_papers['ids'].add(str(paper_id))
    
    def scrape_arxiv_papers(self, max_results: int = 300) -> List[Dict]:
        """
        ArXiv에서 논문을 수집합니다.
        
        Args:
            max_results: 수집할 최대 논문 수 (도메인당)
            
        Returns:
            논문 메타데이터 딕셔너리 리스트
        """
        self.logger.info("ArXiv에서 창고 자동화 시스템 논문 수집을 시작합니다...")
        self.logger.info(f"도메인별 검색 모드: 각 도메인당 최대 {max_results}편씩 수집")
        papers = []
        
        try:
            # 도메인별로 개별 검색 수행
            for domain, keywords in self.domain_search_keywords.items():
                self.logger.info(f"\n[{domain}] 도메인 검색 시작...")
                self.logger.info(f"  검색 키워드: {', '.join(keywords)}")
                
                # 도메인별 키워드를 OR로 결합
                search_query = " OR ".join([f'"{keyword}"' for keyword in keywords])
                
                search = arxiv.Search(
                    query=search_query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                
                domain_paper_count = 0
                
                for result in search.results():
                    paper_data = {
                        "id": result.entry_id,
                        "title": result.title,
                        "authors": [author.name for author in result.authors],
                        "abstract": result.summary,
                        "published": result.published.strftime("%Y-%m-%d"),
                        "categories": result.categories,
                        "pdf_url": result.pdf_url,
                        "source": "ArXiv"
                    }
                    
                    # 중복 검사 (임시 비활성화)
                    is_duplicate, reason = self.is_duplicate_paper(paper_data)
                    if is_duplicate:
                        self.logger.info(f"중복 논문 스킵: {reason}")
                        continue
                    
                    # 도메인 분류 (검색된 도메인으로 강제 설정)
                    paper_data['domain'] = domain
                    
                    papers.append(paper_data)
                    domain_paper_count += 1
                    
                    # 기존 논문 목록에 추가
                    self.add_to_existing_papers(paper_data)
                    
                    # PDF 다운로드 (도메인 포함 파일명)
                    try:
                        clean_title = result.title[:40].replace(" ", "_")
                        filename = f"{len(papers):03d}_{domain}_{clean_title}.pdf"
                        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                        
                        file_path = f"{self.output_dir}/ArXiv/{filename}"
                        if os.path.exists(file_path):
                            self.logger.info(f"파일 이미 존재: {filename}")
                        else:
                            result.download_pdf(dirpath=f"{self.output_dir}/ArXiv", filename=filename)
                            self.logger.info(f"[{domain}] 다운로드 완료: {filename}")
                            self.existing_papers['files'].add(filename)
                    except Exception as e:
                        self.logger.warning(f"PDF 다운로드 실패: {e}")
                    
                    time.sleep(1)
                
                self.logger.info(f"[{domain}] 도메인 수집 완료: {domain_paper_count}편")
            
            # 메타데이터 저장
            with open(f"{self.output_dir}/ArXiv/metadata.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "collection_info": {
                        "created_date": datetime.now().strftime("%Y-%m-%d"),
                        "total_papers": len(papers),
                        "source": "ArXiv",
                        "search_keywords": self.warehouse_automation_keywords
                    },
                    "papers": papers
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"ArXiv 논문 수집 완료. {len(papers)}개 논문을 수집했습니다.")
            
        except Exception as e:
            self.logger.error(f"ArXiv 수집 실패: {e}")
        
        return papers
    
    def generate_scraping_report(self, papers: List[Dict]):
        """논문 수집 과정의 요약 보고서를 생성합니다."""
        total_papers = len(papers)
        
        # 도메인별 통계
        domain_stats = {}
        for paper in papers:
            domain = paper.get('domain', 'General')
            domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        domain_breakdown = '\n'.join([f'- {domain}: {count}편' for domain, count in sorted(domain_stats.items(), key=lambda x: x[1], reverse=True)])
        
        report = f"""
창고 자동화 시스템 논문 수집 보고서
=================================

수집 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
총 논문 수: {total_papers}

도메인별 분류:
{domain_breakdown}

키워드 카테고리: AGV, EMS, RTV, CNV, 경로최적화, 스마트팩토리, AI/ML

수집 소스:
- ArXiv: {total_papers} 논문

검색 키워드:
{chr(10).join([f'- {keyword}' for keyword in self.warehouse_automation_keywords[:10]])}
... 및 {len(self.warehouse_automation_keywords) - 10}개 더

생성된 파일:
- ArXiv/metadata.json (메타데이터)
- ArXiv/*.pdf (PDF 파일들, 도메인별로 분류됨)

파일명 형식: 숫자_도메인_논문제목.pdf
예: 001_AGV_Automated_Guided_Vehicle_Path.pdf

다음 단계:
1. text_extractor.py를 실행하여 PDF에서 텍스트 추출
2. 추출된 데이터를 FAISS 벡터DB에 저장
3. RAG 시스템을 통한 연구 질의응답 시스템 구축
"""
        
        report_file = f"{self.output_dir}/scraping_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"수집 보고서 저장: {report_file}")


def main():
    """창고 자동화 시스템 논문 수집기를 실행하는 메인 함수입니다."""
    parser = argparse.ArgumentParser(description="창고 자동화 시스템 연구 논문 수집기 (ArXiv)")
    parser.add_argument("--output-dir", default="../../1_data/0_crawled", help="논문 저장 디렉토리")
    parser.add_argument("--max-results", type=int, default=100, help="도메인당 최대 수집 논문 수")
    
    args = parser.parse_args()
    
    scraper = WarehouseAutomationPaperScraper(args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"창고 자동화 시스템 논문 수집기")
    print(f"{'='*60}")
    print(f"도메인별 검색 모드: 각 도메인당 최대 {args.max_results}편")
    print(f"총 예상 수집량: 최대 {args.max_results * len(scraper.domain_search_keywords)}편")
    print(f"{'='*60}\n")
    
    papers = scraper.scrape_arxiv_papers(args.max_results)
    scraper.generate_scraping_report(papers)


if __name__ == "__main__":
    main()
