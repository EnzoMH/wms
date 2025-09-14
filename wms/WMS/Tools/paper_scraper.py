#!/usr/bin/env python3
"""
WMS 논문 수집기
==============

ArXiv, IEEE Xplore, Semantic Scholar, Google Scholar 등
여러 학술 데이터베이스에서 연구 논문을 수집하는 포괄적인 도구입니다.

작성자: 신명호
날짜: 2025년 9월 3일
버전: 1.0.0
"""

import os
import json
import csv
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


class WMSPaperScraper:
    """여러 소스에서 WMS 관련 연구 논문을 수집하는 메인 클래스입니다."""
    
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
        
        # WMS 연구를 위한 고도화된 검색 키워드
        self.wms_keywords = [
            # 핵심 로봇 기술
            "AMR autonomous mobile robot warehouse",
            "AGV automated guided vehicle logistics", 
            "CNV conveyor system automation",
            "RTV return to vendor process",
            "collaborative robot cobot warehouse",
            "palletizing robot automation",
            
            # 스마트팩토리 통합
            "smart factory warehouse integration",
            "Industry 4.0 warehouse management",
            "digital twin warehouse simulation",
            "cyber physical system logistics",
            
            # 고급 제어 시스템
            "WCS warehouse control system",
            "WES warehouse execution system", 
            "MES manufacturing execution system warehouse",
            
            # 첨단 피킹 기술
            "pick to light system optimization",
            "voice picking technology",
            "vision guided picking robot",
            "goods to person automation",
            
            # 자동창고 시스템
            "AS/RS automated storage retrieval system",
            "VLM vertical lift module",
            "shuttle system warehouse",
            
            # 최적화 알고리즘
            "slotting optimization algorithm",
            "wave planning optimization",
            "batch picking optimization",
            
            # 통합 시스템
            "ERP WMS integration",
            "TMS transportation management integration"
        ]
    
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
    
    def scrape_arxiv_papers(self, max_results: int = 50) -> List[Dict]:
        """
        Scrape papers from ArXiv.
        
        Args:
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of paper metadata dictionaries
        """
        self.logger.info("ArXiv 논문 수집을 시작합니다...")
        papers = []
        
        try:
            # 여러 검색 조건을 결합
            search_query = " OR ".join([f'"{keyword}"' for keyword in self.wms_keywords[:5]])
            
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
                papers.append(paper_data)
                
                # 가능한 경우 PDF 다운로드
                try:
                    filename = f"{len(papers):03d}_{result.title[:50]}.pdf".replace(" ", "_")
                    result.download_pdf(dirpath=f"{self.output_dir}/ArXiv", filename=filename)
                    self.logger.info(f"Downloaded: {filename}")
                except Exception as e:
                    self.logger.warning(f"Failed to download PDF: {e}")
                
                time.sleep(1)  # 속도 제한
            
            # Save metadata
            with open(f"{self.output_dir}/ArXiv/metadata.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "collection_info": {
                        "created_date": datetime.now().strftime("%Y-%m-%d"),
                        "total_papers": len(papers),
                        "source": "ArXiv",
                        "search_keywords": self.wms_keywords
                    },
                    "papers": papers
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"ArXiv scraping completed. Found {len(papers)} papers.")
            
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
        self.logger.info("Starting Semantic Scholar paper scraping...")
        papers = []
        
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        
        try:
            for keyword in self.wms_keywords[:3]:  # Limit to avoid rate limits
                params = {
                    "query": keyword,
                    "limit": min(max_results // 3, 100),
                    "fields": "paperId,title,authors,year,venue,citationCount,abstract,url"
                }
                
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
                    papers.append(paper_data)
                
                time.sleep(1)  # 속도 제한
            
            # Save search results
            with open(f"{self.output_dir}/SemanticScholar/search_results.json", 'w', encoding='utf-8') as f:
                json.dump({
                    "search_metadata": {
                        "query_date": datetime.now().strftime("%Y-%m-%d"),
                        "total_results": len(papers),
                        "source": "Semantic Scholar",
                        "search_keywords": self.wms_keywords[:3]
                    },
                    "papers": papers
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Semantic Scholar scraping completed. Found {len(papers)} papers.")
            
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
        self.logger.info("Starting Google Scholar paper scraping...")
        papers = []
        abstracts_text = []
        
        try:
            for keyword in self.wms_keywords[:2]:  # Limit to avoid blocking
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
                        
                        papers.append(paper_data)
                        
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
                f.write("Google Scholar Search Results - WMS Research Papers\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"Search Date: {datetime.now().strftime('%Y-%m-%d')}\\n")
                f.write(f"Total Papers: {len(papers)}\\n\\n")
                f.write("\\n".join(abstracts_text))
            
            self.logger.info(f"Google Scholar scraping completed. Found {len(papers)} papers.")
            
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
        Run complete scraping process for all sources.
        
        Args:
            max_results_per_source: Maximum papers per source
        """
        self.logger.info("Starting comprehensive WMS paper scraping...")
        
        all_papers = {}
        
        # Scrape from all sources
        all_papers['arxiv'] = self.scrape_arxiv_papers(max_results_per_source)
        time.sleep(5)  # Break between sources
        
        all_papers['semantic_scholar'] = self.scrape_semantic_scholar(max_results_per_source)
        time.sleep(5)
        
        all_papers['google_scholar'] = self.scrape_google_scholar(max_results_per_source)
        
        # Generate citation files
        self.generate_citation_file(all_papers.get('google_scholar', []), "IEEE")
        
        # Generate summary report
        self.generate_scraping_report(all_papers)
        
        self.logger.info("Scraping process completed successfully!")
    
    def generate_scraping_report(self, papers_by_source: Dict[str, List]):
        """Generate a summary report of the scraping process."""
        total_papers = sum(len(papers) for papers in papers_by_source.values())
        
        report = f"""
WMS 논문 수집 보고서
========================

수집 날짜 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
총 논문 수 : {total_papers}

소스별 분할:
- ArXiv(아카이브논문): {len(papers_by_source.get('arxiv', []))} 논문
- Semantic Scholar(시맨틱 셀러): {len(papers_by_source.get('semantic_scholar', []))} 논문  
- Google Scholar(구글 스쿨러): {len(papers_by_source.get('google_scholar', []))} 논문

Search Keywords Used:
{chr(10).join([f'- {keyword}' for keyword in self.wms_keywords])}

Files Generated:
- ArXiv/metadata.json(메타데이터)
- SemanticScholar/search_results.json(검색 결과)
- GoogleScholar/abstracts.txt(초록)
- IEEE/citations.bib (from Google Scholar data)

다음 단계:
1. text_extractor.py를 실행하여 PDF에서 전체 텍스트를 추출합니다.
2. citation_analyzer.py를 실행하여 인용 네트워크를 분석합니다.
3. trend_visualizer.py를 실행하여 시각화를 생성합니다.
"""
        
        with open(f"{self.output_dir}/../scraping_report.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)


def main():
    """논문 수집기를 실행하는 메인 함수입니다."""
    parser = argparse.ArgumentParser(description="WMS Research Paper Scraper")
    parser.add_argument("--output-dir", default="../Papers", help="Output directory for papers")
    parser.add_argument("--max-results", type=int, default=50, help="Maximum results per source")
    parser.add_argument("--source", choices=['arxiv', 'semantic', 'google', 'all'], 
                       default='all', help="Source to scrape from")
    
    args = parser.parse_args()
    
    scraper = WMSPaperScraper(args.output_dir)
    
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
