#!/usr/bin/env python3
"""
창고 자동화 시스템 논문 수집기
========================

ArXiv, IEEE Xplore, Semantic Scholar, Google Scholar 등
여러 학술 데이터베이스에서 AGV, EMS, RTV, CNV 등
창고 자동화 및 스마트팩토리 관련 연구 논문을 수집하는 포괄적인 도구입니다.

작성자: 신명호
날짜: 2025년 9월 3일
버전: 1.0.0
"""

import os
import json
import csv
import re
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# 서드파티 라이브러리 임포트
try:
    import arxiv
    import scholarly
    from scholarly import scholarly as gs
    import bibtexparser
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("실행: pip install arxiv scholarly bibtexparser requests")
    exit(1)


class WarehouseAutomationPaperScraper:
    """여러 소스에서 창고 자동화 시스템(AGV, EMS, RTV, CNV) 관련 연구 논문을 수집하는 메인 클래스입니다."""
    
    def __init__(self, output_dir: str = "../Papers"):
        """
        논문 수집기를 초기화합니다.
        
        Args:
            output_dir: 다운로드된 논문과 메타데이터를 저장할 디렉토리
        """
        self.output_dir = output_dir
        self.setup_logging()
        self.setup_directories()
        self.session = self.setup_session()
        
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
        
        # 기존 WMS 일반 키워드 (주석 처리됨)
        # 기존 일반적인 WMS 키워드 (현재는 사용하지 않음)
        # self.warehouse_automation_keywords = [
        #     "warehouse management system",
        #     "WMS optimization", 
        #     "warehouse automation",
        #     "inventory management system",
        #     "automated storage retrieval",
        #     "AGV warehouse",
        #     "warehouse robotics",
        #     "smart warehouse", 
        #     "warehouse IoT",
        #     "supply chain automation"
        # ]
        # 현재는 위의 AGV, EMS, RTV, CNV 등 더 구체적인 창고 자동화 시스템 키워드를 사용
    
    def setup_logging(self):
        """수집기의 로깅을 설정합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('paper_scraper.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """논문 저장을 위한 필수 디렉토리를 생성합니다."""
        directories = [
            f"{self.output_dir}/ArXiv",
            f"{self.output_dir}/IEEE", 
            f"{self.output_dir}/SemanticScholar",
            f"{self.output_dir}/GoogleScholar"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Created directory: {directory}")
    
    def setup_session(self) -> requests.Session:
        """재시도 전략을 포함한 HTTP 세션을 설정합니다."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
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
            # 각 소스 디렉토리에서 기존 메타데이터 확인
            sources = ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']
            
            for source in sources:
                source_path = f"{self.output_dir}/{source}"
                
                # 메타데이터 파일들 확인
                metadata_files = [
                    f"{source_path}/metadata.json",
                    f"{source_path}/search_results.json"
                ]
                
                for metadata_file in metadata_files:
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
                                    paper_id = paper.get('id') or paper.get('paperId')
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
        paper_id = paper_data.get('id') or paper_data.get('paperId')
        
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
        paper_id = paper_data.get('id') or paper_data.get('paperId')
        
        if title:
            self.existing_papers['titles'].add(title)
        if url:
            self.existing_papers['urls'].add(url)
        if paper_id:
            self.existing_papers['ids'].add(str(paper_id))
    
    def scrape_arxiv_papers(self, max_results: int = 50) -> List[Dict]:
        """
        Scrape papers from ArXiv.
        
        Args:
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of paper metadata dictionaries
        """
        self.logger.info("ArXiv에서 창고 자동화 시스템 논문 수집을 시작합니다...")
        papers = []
        
        try:
            # 여러 검색 조건을 결합
            search_query = " OR ".join([f'"{keyword}"' for keyword in self.warehouse_automation_keywords[:5]])
            
            search = arxiv.Search(
                query=search_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
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
                
                # 중복 검사
                is_duplicate, reason = self.is_duplicate_paper(paper_data)
                if is_duplicate:
                    self.logger.info(f"⏭️ 중복 논문 스킵: {reason}")
                    continue
                
                papers.append(paper_data)
                
                # 기존 논문 목록에 추가
                self.add_to_existing_papers(paper_data)
                
                # 가능한 경우 PDF 다운로드
                try:
                    filename = f"{len(papers):03d}_{result.title[:50]}.pdf".replace(" ", "_")
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)  # 파일명 정규화
                    
                    # 파일이 이미 존재하는지 확인
                    file_path = f"{self.output_dir}/ArXiv/{filename}"
                    if os.path.exists(file_path):
                        self.logger.info(f"📄 파일 이미 존재: {filename}")
                    else:
                        result.download_pdf(dirpath=f"{self.output_dir}/ArXiv", filename=filename)
                        self.logger.info(f"✅ 다운로드 완료: {filename}")
                        self.existing_papers['files'].add(filename)
                except Exception as e:
                    self.logger.warning(f"❌ PDF 다운로드 실패: {e}")
                
                time.sleep(1)  # 속도 제한
            
            # Save metadata
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
            
            self.logger.info(f"ArXiv 창고 자동화 논문 수집 완료. {len(papers)}개 논문을 발견했습니다.")
            
        except Exception as e:
            self.logger.error(f"ArXiv scraping failed: {e}")
        
        return papers
    
    def scrape_semantic_scholar(self, max_results: int = 50) -> List[Dict]:
        """
        Scrape papers from Semantic Scholar API.
        
        Args:
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of paper metadata dictionaries
        """
        self.logger.info("Semantic Scholar에서 창고 자동화 시스템 논문 수집을 시작합니다...")
        papers = []
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        try:
            # Rate limiting 개선: 더 적은 키워드와 더 긴 대기 시간
            for keyword in self.warehouse_automation_keywords[:2]:  # Limit to 2 keywords to avoid rate limits
                params = {
                    "query": keyword,
                    "limit": min(max_results // 2, 50),  # 더 작은 배치 크기
                    "fields": "paperId,title,authors,year,venue,citationCount,abstract,url"
                }
                
                self.logger.info(f"Querying Semantic Scholar for: {keyword}")
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                for paper in data.get('data', []):
                    paper_data = {
                        "id": paper.get("paperId"),
                        "title": paper.get("title"),
                        "authors": [author.get("name", "") for author in paper.get("authors", [])],
                        "abstract": paper.get("abstract", ""),
                        "year": paper.get("year"),
                        "venue": paper.get("venue"),
                        "citation_count": paper.get("citationCount", 0),
                        "url": paper.get("url"),
                        "source": "Semantic Scholar",
                        "search_keyword": keyword
                    }
                    
                    # 🔍 중복 검사
                    is_duplicate, reason = self.is_duplicate_paper(paper_data)
                    if is_duplicate:
                        self.logger.info(f"⏭️ Semantic Scholar 중복 스킵: {reason}")
                        continue
                    
                    papers.append(paper_data)
                    # 기존 논문 목록에 추가
                    self.add_to_existing_papers(paper_data)
                
                self.logger.info(f"Found {len(data.get('data', []))} papers for keyword: {keyword}")
                time.sleep(3)  # 더 긴 대기 시간으로 rate limiting 회피
            
            # Save search results
            with open(f"{self.output_dir}/SemanticScholar/search_results.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "search_metadata": {
                        "query_date": datetime.now().strftime("%Y-%m-%d"),
                        "total_results": len(papers),
                        "source": "Semantic Scholar",
                        "search_keywords": self.warehouse_automation_keywords[:3]
                    },
                    "papers": papers
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Semantic Scholar 창고 자동화 논문 수집 완료. {len(papers)}개 논문을 발견했습니다.")
            
        except Exception as e:
            self.logger.error(f"Semantic Scholar scraping failed: {e}")
        
        return papers
    
    def scrape_google_scholar(self, max_results: int = 30) -> List[Dict]:
        """
        Scrape papers from Google Scholar using scholarly library.
        
        Args:
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of paper metadata dictionaries
        """
        self.logger.info("Google Scholar에서 창고 자동화 시스템 논문 수집을 시작합니다...")
        papers = []
        abstracts_text = []
        
        try:
            for keyword in self.warehouse_automation_keywords[:2]:  # Limit to avoid blocking
                search_query = gs.search_pubs(keyword)
                
                count = 0
                for paper in search_query:
                    if count >= max_results // 2:
                        break
                    
                    try:
                        paper_data = {
                            "title": paper.get("bib", {}).get("title", ""),
                            "authors": paper.get("bib", {}).get("author", []),
                            "year": paper.get("bib", {}).get("pub_year"),
                            "venue": paper.get("bib", {}).get("venue", ""),
                            "citation_count": paper.get("num_citations", 0),
                            "url": paper.get("pub_url", ""),
                            "abstract": paper.get("bib", {}).get("abstract", ""),
                            "source": "Google Scholar",
                            "search_keyword": keyword
                        }
                        
                        # 🔍 중복 검사
                        is_duplicate, reason = self.is_duplicate_paper(paper_data)
                        if is_duplicate:
                            self.logger.info(f"⏭️ Google Scholar 중복 스킵: {reason}")
                            continue
                        
                        papers.append(paper_data)
                        # 기존 논문 목록에 추가
                        self.add_to_existing_papers(paper_data)
                        
                        # Collect abstracts for text file
                        if paper_data["abstract"]:
                            abstracts_text.append(f"Title: {paper_data['title']}")
                            abstracts_text.append(f"Authors: {', '.join(paper_data['authors']) if isinstance(paper_data['authors'], list) else paper_data['authors']}")
                            abstracts_text.append(f"Year: {paper_data['year']}")
                            abstracts_text.append(f"Abstract: {paper_data['abstract']}")
                            abstracts_text.append("-" * 80)
                        
                        count += 1
                        time.sleep(2)  # 속도 제한 for Google Scholar
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing paper: {e}")
                        continue
            
            # Save abstracts to text file
            with open(f"{self.output_dir}/GoogleScholar/abstracts.txt", 'w', encoding='utf-8') as f:
                f.write("Google Scholar Search Results - 창고 자동화 시스템 연구 논문\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"Search Date: {datetime.now().strftime('%Y-%m-%d')}\\n")
                f.write(f"Total Papers: {len(papers)}\\n\\n")
                f.write("\\n".join(abstracts_text))
            
            self.logger.info(f"Google Scholar 창고 자동화 논문 수집 완료. {len(papers)}개 논문을 발견했습니다.")
            
        except Exception as e:
            self.logger.error(f"Google Scholar scraping failed: {e}")
        
        return papers
    
    def generate_citation_file(self, papers: List[Dict], source: str):
        """Generate BibTeX citation file for IEEE papers."""
        if source == "IEEE":
            citations = []
            for i, paper in enumerate(papers):
                citation = f"""@article{{paper_{i+1:03d},
  title={{{paper.get('title', 'Unknown Title')}}},
  author={{{', '.join(paper.get('authors', ['Unknown Author']))}}},
  journal={{{paper.get('venue', 'Unknown Journal')}}},
  year={{{paper.get('year', 'Unknown Year')}}},
  publisher={{IEEE}}
}}"""
                citations.append(citation)
            
            with open(f"{self.output_dir}/IEEE/citations.bib", 'w', encoding='utf-8') as f:
                f.write("\\n\\n".join(citations))
    
    def run_full_scraping(self, max_results_per_source: int = 50):
        """
        모든 소스에서 창고 자동화 시스템 논문을 수집하는 완전한 과정을 실행합니다.
        
        Args:
            max_results_per_source: 소스당 최대 논문 수
        """
        self.logger.info("창고 자동화 시스템 논문 종합 수집을 시작합니다...")
        
        all_papers = {}
        
        # Scrape from all sources
        all_papers['arxiv'] = self.scrape_arxiv_papers(max_results_per_source)
        time.sleep(5)  # Break between sources
        
        all_papers['semantic_scholar'] = self.scrape_semantic_scholar(max_results_per_source)
        time.sleep(5)
        
        # Google Scholar는 captcha 문제로 비활성화
        # all_papers['google_scholar'] = self.scrape_google_scholar(max_results_per_source)
        self.logger.info("Google Scholar 수집은 captcha 문제로 비활성화됨")
        all_papers['google_scholar'] = []
        
        # Generate citation files
        self.generate_citation_file(all_papers.get('google_scholar', []), "IEEE")
        
        # Generate summary report
        self.generate_scraping_report(all_papers)
        
        self.logger.info("Scraping process completed successfully!")
    
    def generate_scraping_report(self, papers_by_source: Dict[str, List]):
        """창고 자동화 시스템 논문 수집 과정의 요약 보고서를 생성합니다."""
        total_papers = sum(len(papers) for papers in papers_by_source.values())
        
        report = f"""
창고 자동화 시스템 논문 수집 보고서
=================================

수집 날짜 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
총 논문 수 : {total_papers}

키워드 카테고리: AGV, EMS, RTV, CNV, 경로최적화, 스마트팩토리, AI/ML

소스별 분할:
- ArXiv(아카이브논문): {len(papers_by_source.get('arxiv', []))} 논문
- Semantic Scholar(시맨틱 셀러): {len(papers_by_source.get('semantic_scholar', []))} 논문  
- Google Scholar(구글 스쿨러): {len(papers_by_source.get('google_scholar', []))} 논문

Search Keywords Used:
{chr(10).join([f'- {keyword}' for keyword in self.warehouse_automation_keywords])}

Files Generated:
- ArXiv/metadata.json(메타데이터)
- SemanticScholar/search_results.json(검색 결과)
- GoogleScholar/abstracts.txt(초록)
- IEEE/citations.bib (from Google Scholar data)

다음 단계:
1. text_extractor.py를 실행하여 PDF에서 전체 텍스트를 추출합니다.
2. citation_analyzer.py를 실행하여 인용 네트워크를 분석합니다.
3. trend_visualizer.py를 실행하여 AGV/EMS/RTV/CNV 트렌드 시각화를 생성합니다.
4. FAISS 벡터DB에 저장하여 창고 자동화 시스템 연구 질의응답 시스템 구축합니다.
"""
        
        with open(f"{self.output_dir}/../scraping_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)


def main():
    """창고 자동화 시스템 논문 수집기를 실행하는 메인 함수입니다."""
    parser = argparse.ArgumentParser(description="창고 자동화 시스템 연구 논문 수집기")
    parser.add_argument("--output-dir", default="../Papers", help="Output directory for papers")
    parser.add_argument("--max-results", type=int, default=50, help="Maximum results per source")
    parser.add_argument("--source", choices=['arxiv', 'semantic', 'google', 'all'], 
                       default='all', help="Source to scrape from")
    
    args = parser.parse_args()
    
    scraper = WarehouseAutomationPaperScraper(args.output_dir)
    
    if args.source == 'arxiv':
        scraper.scrape_arxiv_papers(args.max_results)
    elif args.source == 'semantic':
        scraper.scrape_semantic_scholar(args.max_results)
    elif args.source == 'google':
        scraper.scrape_google_scholar(args.max_results)
    else:
        scraper.run_full_scraping(args.max_results)


if __name__ == "__main__":
    main()
