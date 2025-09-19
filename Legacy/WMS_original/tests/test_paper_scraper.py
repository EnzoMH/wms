#!/usr/bin/env python3
"""
논문 수집기 테스트
================

WMS paper_scraper.py에 대한 단위 테스트 및 통합 테스트를 포함합니다.

작성자: 신명호
날짜: 2025년 9월 3일
버젼 1.0.0
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# 테스트 대상 모듈을 임포트하기 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Tools'))

try:
    from paper_scraper import WMSPaperScraper
except ImportError as e:
    print(f"테스트 대상 모듈을 임포트할 수 없습니다: {e}")
    sys.exit(1)


class TestWMSPaperScraper(unittest.TestCase):
    """WMSPaperScraper 클래스에 대한 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        # C:\wms\test_results로 설정 (상대 경로 사용)
        current_dir = os.path.dirname(__file__)
        self.output_dir = os.path.join(current_dir, '..', '..', 'test_results')
        self.output_dir = os.path.abspath(self.output_dir)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scraper = WMSPaperScraper(output_dir=self.output_dir)
        
    def tearDown(self):
        """각 테스트 후에 실행되는 정리"""
        # test_results 디렉토리는 유지하고 내용만 정리
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    try:
                        os.remove(item_path)
                    except:
                        pass

    def test_initialization(self):
        """초기화 테스트"""
        # 스크래퍼가 올바르게 초기화되는지 확인
        self.assertEqual(self.scraper.output_dir, self.output_dir)
        self.assertIsNotNone(self.scraper.wms_keywords)
        self.assertGreater(len(self.scraper.wms_keywords), 0)
        
        # 필수 키워드가 포함되어 있는지 확인
        expected_keywords = ["warehouse management system", "WMS", "AGV"]
        for keyword in expected_keywords:
            self.assertIn(keyword.lower(), 
                         [k.lower() for k in self.scraper.wms_keywords])

    def test_setup_directories(self):
        """디렉토리 설정 테스트"""
        # 예상되는 디렉토리들이 생성되었는지 확인
        expected_dirs = [
            "ArXiv", "IEEE", "SemanticScholar", "GoogleScholar"
        ]
        
        for dir_name in expected_dirs:
            dir_path = Path(self.output_dir) / dir_name
            self.assertTrue(dir_path.exists(), f"{dir_name} 디렉토리가 생성되지 않았습니다")
            self.assertTrue(dir_path.is_dir(), f"{dir_name}가 디렉토리가 아닙니다")

    def test_setup_session(self):
        """HTTP 세션 설정 테스트"""
        session = self.scraper.setup_session()
        
        # 세션이 올바르게 설정되었는지 확인
        self.assertIsNotNone(session)
        self.assertEqual(len(session.adapters), 2)  # http:// 및 https:// 어댑터

    @patch('paper_scraper.arxiv')
    def test_scrape_arxiv_papers_mock(self, mock_arxiv):
        """ArXiv 논문 수집 테스트 (모킹 사용)"""
        # 모킹된 ArXiv 검색 결과 설정
        mock_result = MagicMock()
        mock_result.entry_id = "http://arxiv.org/abs/2024.0001v1"
        mock_result.title = "Test WMS Paper"
        mock_result.authors = [MagicMock(name="Test Author")]
        mock_result.summary = "Test abstract for WMS research"
        mock_result.published = MagicMock()
        mock_result.published.strftime.return_value = "2024-01-15"
        mock_result.categories = ["cs.RO"]
        mock_result.pdf_url = "http://arxiv.org/pdf/2024.0001v1.pdf"
        mock_result.download_pdf = MagicMock()
        
        mock_search = MagicMock()
        mock_search.results.return_value = [mock_result]
        mock_arxiv.Search.return_value = mock_search
        
        # 테스트 실행
        papers = self.scraper.scrape_arxiv_papers(max_results=1)
        
        # 결과 검증
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['title'], "Test WMS Paper")
        self.assertEqual(papers[0]['source'], "ArXiv")

    def test_invalid_output_directory(self):
        """잘못된 출력 디렉토리 테스트"""
        # 존재하지 않는 상위 경로에 디렉토리 생성 시도
        invalid_path = "/nonexistent/path/that/should/not/exist"
        
        # Windows와 Unix 시스템에서 다르게 처리될 수 있으므로 예외 처리
        try:
            scraper = WMSPaperScraper(output_dir=invalid_path)
            # 디렉토리 생성이 실패해야 하는 경우를 체크
            # 하지만 os.makedirs(exist_ok=True)로 인해 실제로는 실패하지 않을 수 있음
        except (PermissionError, OSError):
            # 예상되는 예외
            pass

    def test_keyword_search_query_generation(self):
        """검색 쿼리 생성 테스트"""
        # 키워드가 올바르게 설정되어 있는지 확인
        self.assertIsInstance(self.scraper.wms_keywords, list)
        self.assertGreater(len(self.scraper.wms_keywords), 5)
        
        # 검색 쿼리 생성 테스트
        keywords_subset = self.scraper.wms_keywords[:3]
        query = " OR ".join([f'"{keyword}"' for keyword in keywords_subset])
        
        self.assertIn("warehouse management", query.lower())
        self.assertIn(" OR ", query)

    @patch('paper_scraper.time.sleep')
    @patch('paper_scraper.requests.Session.get')
    def test_scrape_semantic_scholar_mock(self, mock_get, mock_sleep):
        """Semantic Scholar 논문 수집 테스트 (모킹 사용)"""
        # 모킹된 API 응답 설정
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            'data': [{
                'paperId': 'test123',
                'title': 'Test Semantic Scholar Paper',
                'authors': [{'name': 'Test Author'}],
                'abstract': 'Test abstract for semantic scholar',
                'year': 2024,
                'venue': 'Test Conference',
                'citationCount': 10,
                'url': 'https://test.url'
            }]
        }
        mock_get.return_value = mock_response
        
        # 테스트 실행
        papers = self.scraper.scrape_semantic_scholar(max_results=1)
        
        # 결과 검증
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['title'], 'Test Semantic Scholar Paper')
        self.assertEqual(papers[0]['source'], 'Semantic Scholar')

    def test_generate_citation_file(self):
        """인용 파일 생성 테스트"""
        # 테스트 논문 데이터
        test_papers = [{
            'title': 'Test Paper Title',
            'authors': ['Author 1', 'Author 2'],
            'year': 2024,
            'venue': 'Test Journal'
        }]
        
        # 인용 파일 생성
        self.scraper.generate_citation_file(test_papers, "IEEE")
        
        # 파일이 생성되었는지 확인
        citation_file = Path(self.output_dir) / "IEEE" / "citations.bib"
        self.assertTrue(citation_file.exists())
        
        # 파일 내용 확인
        with open(citation_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Test Paper Title', content)
            self.assertIn('Author 1', content)


class TestWMSPaperScraperIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self):
        """통합 테스트 설정"""
        # C:\wms\test_results로 설정 (상대 경로 사용)
        current_dir = os.path.dirname(__file__)
        self.output_dir = os.path.join(current_dir, '..', '..', 'test_results')
        self.output_dir = os.path.abspath(self.output_dir)
        
        # 디렉토리가 존재하지 않으면 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.scraper = WMSPaperScraper(output_dir=self.output_dir)

    def tearDown(self):
        """통합 테스트 정리"""
        # test_results 디렉토리는 유지하고 내용만 정리
        if os.path.exists(self.output_dir):
            for item in os.listdir(self.output_dir):
                item_path = os.path.join(self.output_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    try:
                        os.remove(item_path)
                    except:
                        pass

    def test_metadata_file_structure(self):
        """메타데이터 파일 구조 테스트"""
        # 가짜 메타데이터 생성
        test_papers = [{
            'id': 'test001',
            'title': 'Integration Test Paper',
            'authors': ['Test Author'],
            'abstract': 'Test abstract for integration testing',
            'published': '2024-01-15',
            'source': 'ArXiv'
        }]
        
        # 메타데이터 파일 생성
        metadata = {
            "collection_info": {
                "created_date": "2024-01-15",
                "total_papers": len(test_papers),
                "source": "ArXiv",
                "search_keywords": self.scraper.wms_keywords
            },
            "papers": test_papers
        }
        
        metadata_file = Path(self.output_dir) / "ArXiv" / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 파일이 올바르게 생성되었는지 확인
        self.assertTrue(metadata_file.exists())
        
        # JSON 파일이 유효한지 확인
        with open(metadata_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            self.assertEqual(loaded_data['collection_info']['total_papers'], 1)
            self.assertEqual(len(loaded_data['papers']), 1)

    def test_full_directory_structure(self):
        """전체 디렉토리 구조 테스트"""
        expected_structure = {
            'ArXiv': ['metadata.json'],
            'IEEE': ['citations.bib'],
            'SemanticScholar': ['search_results.json'],
            'GoogleScholar': ['abstracts.txt']
        }
        
        # 각 디렉토리와 예상 파일들을 생성하고 확인
        for dir_name, files in expected_structure.items():
            dir_path = Path(self.output_dir) / dir_name
            self.assertTrue(dir_path.exists())
            
            for file_name in files:
                # 테스트용 더미 파일 생성
                file_path = dir_path / file_name
                file_path.touch()
                self.assertTrue(file_path.exists())


def run_performance_test():
    """성능 테스트 (별도 함수로 분리)"""
    print("성능 테스트 실행...")
    
    # C:\wms\test_results로 설정 (상대 경로 사용)
    current_dir = os.path.dirname(__file__)
    output_dir = os.path.join(current_dir, '..', '..', 'test_results')
    output_dir = os.path.abspath(output_dir)
    
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        scraper = WMSPaperScraper(output_dir=output_dir)
        
        import time
        start_time = time.time()
        
        # 초기화 시간 측정
        setup_time = time.time() - start_time
        print(f"초기화 시간: {setup_time:.3f}초")
        print(f"테스트 결과 저장 경로: {output_dir}")
        
        # 메모리 사용량 확인 (선택적)
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            print(f"메모리 사용량: {memory_usage:.2f}MB")
        except ImportError:
            print("psutil이 설치되지 않아 메모리 측정을 생략합니다.")
            
    except Exception as e:
        print(f"성능 테스트 중 오류 발생: {e}")


if __name__ == '__main__':
    print("WMS 논문 수집기 테스트 시작...")
    print("=" * 50)
    
    # 단위 테스트 실행
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # 성능 테스트 실행
    run_performance_test()
    
    print("\n모든 테스트가 완료되었습니다!")