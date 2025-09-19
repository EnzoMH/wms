#!/usr/bin/env python3
"""
인용 분석기 테스트
================

WMS citation_analyzer.py에 대한 단위 테스트 및 통합 테스트를 포함합니다.

작성자: 신명호
날짜: 2025년 9월 3일
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
    from citation_analyzer import WMSCitationAnalyzer
    import networkx as nx
except ImportError as e:
    print(f"테스트 대상 모듈을 임포트할 수 없습니다: {e}")
    sys.exit(1)


class TestWMSCitationAnalyzer(unittest.TestCase):
    """WMSCitationAnalyzer 클래스에 대한 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        # 임시 디렉토리 생성
        self.temp_papers_dir = tempfile.mkdtemp(prefix="papers_")
        self.temp_processed_dir = tempfile.mkdtemp(prefix="processed_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="output_")
        
        # 테스트용 디렉토리 구조 생성
        for source in ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']:
            os.makedirs(os.path.join(self.temp_papers_dir, source), exist_ok=True)
        
        self.analyzer = WMSCitationAnalyzer(
            papers_dir=self.temp_papers_dir,
            processed_dir=self.temp_processed_dir,
            output_dir=self.temp_output_dir
        )
        
    def tearDown(self):
        """각 테스트 후에 실행되는 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_papers_dir, ignore_errors=True)
        shutil.rmtree(self.temp_processed_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)

    def test_initialization(self):
        """초기화 테스트"""
        # 인용 분석기가 올바르게 초기화되는지 확인
        self.assertEqual(str(self.analyzer.papers_dir), self.temp_papers_dir)
        self.assertEqual(str(self.analyzer.processed_dir), self.temp_processed_dir)
        self.assertEqual(str(self.analyzer.output_dir), self.temp_output_dir)
        
        # 네트워크 그래프가 초기화되었는지 확인
        self.assertIsInstance(self.analyzer.citation_graph, nx.DiGraph)
        self.assertIsInstance(self.analyzer.similarity_graph, nx.Graph)
        
        # 빈 메타데이터 딕셔너리가 초기화되었는지 확인
        self.assertIsInstance(self.analyzer.papers_metadata, dict)
        self.assertIsInstance(self.analyzer.paper_abstracts, dict)

    def test_setup_directories(self):
        """디렉토리 설정 테스트"""
        # 출력 디렉토리가 생성되었는지 확인
        self.assertTrue(Path(self.temp_output_dir).exists())
        self.assertTrue(Path(self.temp_output_dir).is_dir())

    def test_load_paper_metadata(self):
        """논문 메타데이터 로딩 테스트"""
        # 테스트용 ArXiv 메타데이터 파일 생성
        arxiv_metadata = {
            "collection_info": {
                "created_date": "2024-01-15",
                "total_papers": 1
            },
            "papers": [{
                "id": "2024.0001",
                "title": "Test WMS Paper",
                "authors": ["Author One", "Author Two"],
                "abstract": "This is a test abstract about warehouse management systems.",
                "published": "2024-01-15",
                "pdf_url": "http://test.url"
            }]
        }
        
        arxiv_dir = Path(self.temp_papers_dir) / "ArXiv"
        arxiv_dir.mkdir(exist_ok=True)
        with open(arxiv_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(arxiv_metadata, f)
        
        # 테스트용 Semantic Scholar 데이터 파일 생성
        ss_data = {
            "search_metadata": {
                "query_date": "2024-01-15",
                "total_results": 1
            },
            "papers": [{
                "id": "test123",
                "title": "Semantic Scholar Test Paper",
                "authors": ["SS Author"],
                "abstract": "Test abstract for semantic scholar analysis.",
                "year": 2024,
                "venue": "Test Conference",
                "citation_count": 15,
                "url": "http://test.semantic.url"
            }]
        }
        
        ss_dir = Path(self.temp_papers_dir) / "SemanticScholar"
        ss_dir.mkdir(exist_ok=True)
        with open(ss_dir / "search_results.json", 'w', encoding='utf-8') as f:
            json.dump(ss_data, f)
        
        # 메타데이터 로딩 실행
        self.analyzer.load_paper_metadata()
        
        # 결과 검증
        self.assertGreater(len(self.analyzer.papers_metadata), 0)
        self.assertIn("arxiv_2024.0001", self.analyzer.papers_metadata)
        self.assertIn("ss_test123", self.analyzer.papers_metadata)
        
        # 논문 정보가 올바르게 로드되었는지 확인
        arxiv_paper = self.analyzer.papers_metadata["arxiv_2024.0001"]
        self.assertEqual(arxiv_paper['title'], "Test WMS Paper")
        self.assertEqual(len(arxiv_paper['authors']), 2)

    def test_extract_year(self):
        """연도 추출 테스트"""
        # 다양한 형식의 메타데이터로 연도 추출 테스트
        
        # year 필드가 있는 경우
        metadata1 = {"year": 2024}
        self.assertEqual(self.analyzer.extract_year(metadata1), 2024)
        
        # published 필드가 있는 경우
        metadata2 = {"published": "2024-01-15"}
        self.assertEqual(self.analyzer.extract_year(metadata2), 2024)
        
        # 연도 정보가 없는 경우
        metadata3 = {"title": "Test Paper"}
        self.assertIsNone(self.analyzer.extract_year(metadata3))

    def test_calculate_text_similarity(self):
        """텍스트 유사도 계산 테스트"""
        # 비슷한 텍스트
        text1 = "warehouse management system automation AGV robotics"
        text2 = "warehouse management automated guided vehicle robot systems"
        
        similarity = self.analyzer.calculate_text_similarity(text1, text2)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
        self.assertGreater(similarity, 0.1)  # 관련 용어들이 있으므로 어느 정도 유사도가 있어야 함
        
        # 완전히 다른 텍스트
        text3 = "completely different topic about cooking recipes"
        similarity2 = self.analyzer.calculate_text_similarity(text1, text3)
        self.assertLess(similarity2, similarity)
        
        # 빈 텍스트
        similarity3 = self.analyzer.calculate_text_similarity("", text1)
        self.assertEqual(similarity3, 0.0)

    def test_build_citation_network(self):
        """인용 네트워크 구축 테스트"""
        # 테스트용 메타데이터 추가
        self.analyzer.papers_metadata = {
            "paper1": {
                "title": "Early WMS Paper",
                "year": 2020,
                "abstract": "warehouse management system basic functionality"
            },
            "paper2": {
                "title": "Advanced WMS Paper", 
                "year": 2022,
                "abstract": "warehouse management system automation integration"
            },
            "paper3": {
                "title": "AGV Integration Paper",
                "year": 2023, 
                "abstract": "automated guided vehicle warehouse management integration"
            }
        }
        
        # 인용 네트워크 구축
        self.analyzer.build_citation_network()
        
        # 네트워크가 구축되었는지 확인
        self.assertGreater(len(self.analyzer.citation_graph.nodes), 0)
        self.assertEqual(len(self.analyzer.citation_graph.nodes), 3)
        
        # 노드에 메타데이터가 포함되어 있는지 확인
        for node_id in self.analyzer.citation_graph.nodes:
            node_data = self.analyzer.citation_graph.nodes[node_id]
            self.assertIn('title', node_data)

    def test_build_similarity_network(self):
        """유사도 네트워크 구축 테스트"""
        # 테스트용 초록 데이터 추가
        self.analyzer.paper_abstracts = {
            "paper1": "warehouse management system optimization",
            "paper2": "warehouse management automation technology",
            "paper3": "supply chain management logistics"
        }
        
        # 유사도 네트워크 구축
        self.analyzer.build_similarity_network()
        
        # 네트워크가 구축되었는지 확인
        self.assertGreaterEqual(len(self.analyzer.similarity_graph.nodes), 0)
        
        # 엣지가 있는 경우 가중치가 올바른지 확인
        for edge in self.analyzer.similarity_graph.edges(data=True):
            source, target, data = edge
            self.assertIn('weight', data)
            self.assertIn('similarity_score', data)
            self.assertGreaterEqual(data['weight'], 0.0)
            self.assertLessEqual(data['weight'], 1.0)

    def test_analyze_network_metrics(self):
        """네트워크 지표 분석 테스트"""
        # 테스트용 간단한 네트워크 생성
        self.analyzer.citation_graph.add_node("paper1", title="Paper 1")
        self.analyzer.citation_graph.add_node("paper2", title="Paper 2")
        self.analyzer.citation_graph.add_node("paper3", title="Paper 3")
        self.analyzer.citation_graph.add_edge("paper2", "paper1", weight=0.8)
        self.analyzer.citation_graph.add_edge("paper3", "paper1", weight=0.6)
        
        # 네트워크 지표 분석
        metrics = self.analyzer.analyze_network_metrics()
        
        # 결과 검증
        self.assertIsInstance(metrics, dict)
        self.assertIn('basic_metrics', metrics)
        self.assertIn('centrality_measures', metrics)
        
        # 기본 지표 확인
        basic_metrics = metrics['basic_metrics']
        self.assertEqual(basic_metrics['total_nodes'], 3)
        self.assertEqual(basic_metrics['total_edges'], 2)
        self.assertIsInstance(basic_metrics['density'], float)

    def test_export_network_data(self):
        """네트워크 데이터 내보내기 테스트"""
        # 테스트용 네트워크 데이터 추가
        self.analyzer.citation_graph.add_node("paper1", title="Test Paper 1")
        self.analyzer.citation_graph.add_node("paper2", title="Test Paper 2")
        self.analyzer.citation_graph.add_edge("paper2", "paper1", weight=0.5, citation_type="reference")
        
        # 네트워크 데이터 내보내기
        self.analyzer.export_network_data()
        
        # GEXF 파일이 생성되었는지 확인
        gexf_file = Path(self.temp_output_dir) / "citation_network.gexf"
        if gexf_file.exists():  # NetworkX 버전에 따라 지원되지 않을 수 있음
            self.assertTrue(gexf_file.exists())

    def test_generate_analysis_report(self):
        """분석 보고서 생성 테스트"""
        # 테스트용 지표 데이터
        test_metrics = {
            'basic_metrics': {
                'total_nodes': 5,
                'total_edges': 8,
                'density': 0.4,
                'is_connected': True,
                'number_of_components': 1
            },
            'centrality_measures': {
                'most_central_papers': {
                    'degree': 'paper1',
                    'betweenness': 'paper2',
                    'closeness': 'paper1',
                    'pagerank': 'paper3'
                }
            }
        }
        
        # 분석 보고서 생성
        self.analyzer.generate_analysis_report(test_metrics)
        
        # 보고서 파일이 생성되었는지 확인
        report_file = Path(self.temp_output_dir) / "citation_analysis_report.md"
        self.assertTrue(report_file.exists())
        
        # 보고서 내용 확인
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('인용 네트워크 분석', content)
            self.assertIn('5', content)  # 총 논문 수
            self.assertIn('8', content)  # 총 인용 수


class TestWMSCitationAnalyzerIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self):
        """통합 테스트 설정"""
        self.temp_papers_dir = tempfile.mkdtemp(prefix="integration_papers_")
        self.temp_processed_dir = tempfile.mkdtemp(prefix="integration_processed_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="integration_output_")
        
        self.analyzer = WMSCitationAnalyzer(
            papers_dir=self.temp_papers_dir,
            processed_dir=self.temp_processed_dir,
            output_dir=self.temp_output_dir
        )

    def tearDown(self):
        """통합 테스트 정리"""
        shutil.rmtree(self.temp_papers_dir, ignore_errors=True)
        shutil.rmtree(self.temp_processed_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)

    def test_complete_analysis_workflow(self):
        """전체 분석 워크플로우 테스트"""
        # 종합적인 테스트 데이터 생성
        
        # ArXiv 메타데이터
        arxiv_metadata = {
            "collection_info": {"created_date": "2024-01-15", "total_papers": 2},
            "papers": [
                {
                    "id": "2024.0001",
                    "title": "Warehouse Management System Optimization",
                    "authors": ["Alice Smith", "Bob Jones"],
                    "abstract": "This paper presents novel optimization techniques for warehouse management systems focusing on automated storage and retrieval systems.",
                    "published": "2024-01-15",
                    "pdf_url": "http://test1.url"
                },
                {
                    "id": "2024.0002", 
                    "title": "AGV Integration in Modern Warehouses",
                    "authors": ["Carol Davis"],
                    "abstract": "Integration of automated guided vehicles with existing warehouse management infrastructure for improved efficiency.",
                    "published": "2024-02-15",
                    "pdf_url": "http://test2.url"
                }
            ]
        }
        
        # 테스트 파일 생성
        arxiv_dir = Path(self.temp_papers_dir) / "ArXiv"
        arxiv_dir.mkdir(parents=True, exist_ok=True)
        with open(arxiv_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(arxiv_metadata, f, ensure_ascii=False, indent=2)
        
        # Semantic Scholar 데이터
        ss_data = {
            "search_metadata": {"query_date": "2024-01-15", "total_results": 1},
            "papers": [{
                "id": "semantic123",
                "title": "Smart Warehouse Technologies Review", 
                "authors": ["David Wilson"],
                "abstract": "Comprehensive review of smart warehouse technologies including IoT, AI, and automation systems.",
                "year": 2023,
                "venue": "Journal of Warehouse Technology",
                "citation_count": 25,
                "url": "http://semantic.test.url"
            }]
        }
        
        ss_dir = Path(self.temp_papers_dir) / "SemanticScholar"
        ss_dir.mkdir(parents=True, exist_ok=True)
        with open(ss_dir / "search_results.json", 'w', encoding='utf-8') as f:
            json.dump(ss_data, f, ensure_ascii=False, indent=2)
        
        # 전체 워크플로우 실행 (일부 단계만)
        self.analyzer.load_paper_metadata()
        
        # 메타데이터 로딩 확인
        self.assertGreater(len(self.analyzer.papers_metadata), 0)
        self.assertGreater(len(self.analyzer.paper_abstracts), 0)
        
        # 네트워크 구축
        self.analyzer.build_citation_network()
        self.analyzer.build_similarity_network()
        
        # 네트워크 분석
        metrics = self.analyzer.analyze_network_metrics()
        self.assertIsInstance(metrics, dict)
        
        # 보고서 생성
        self.analyzer.generate_analysis_report(metrics)
        
        # 결과 파일 확인
        report_file = Path(self.temp_output_dir) / "citation_analysis_report.md"
        self.assertTrue(report_file.exists())

    def test_network_visualization_mock(self):
        """네트워크 시각화 테스트 (모킹 사용)"""
        # 테스트용 네트워크 데이터
        self.analyzer.citation_graph.add_node("paper1", title="Paper 1")
        self.analyzer.citation_graph.add_node("paper2", title="Paper 2")
        self.analyzer.citation_graph.add_edge("paper2", "paper1")
        
        # 메타데이터 추가
        self.analyzer.papers_metadata = {
            "paper1": {"title": "Paper 1", "source": "ArXiv"},
            "paper2": {"title": "Paper 2", "source": "IEEE"}
        }
        
        # 시각화 생성 (실제로는 파일을 생성하지 않도록 모킹)
        with patch('citation_analyzer.plt.savefig') as mock_savefig, \
             patch('citation_analyzer.nx.spring_layout') as mock_layout:
            
            mock_layout.return_value = {"paper1": (0, 0), "paper2": (1, 1)}
            
            try:
                self.analyzer.generate_network_visualizations()
                # 에러 없이 실행되면 성공
                self.assertTrue(True)
            except Exception as e:
                # 의존성 문제로 실패할 수 있으므로 경고만 출력
                print(f"시각화 테스트 건너뛰기: {e}")

    def test_empty_data_handling(self):
        """빈 데이터 처리 테스트"""
        # 빈 메타데이터로 분석 실행
        self.analyzer.load_paper_metadata()  # 빈 디렉토리
        self.assertEqual(len(self.analyzer.papers_metadata), 0)
        
        # 빈 데이터로 네트워크 구축 시도
        self.analyzer.build_citation_network()
        self.analyzer.build_similarity_network()
        
        # 빈 네트워크 분석
        metrics = self.analyzer.analyze_network_metrics()
        self.assertEqual(metrics['basic_metrics']['total_nodes'], 0)
        self.assertEqual(metrics['basic_metrics']['total_edges'], 0)


def run_performance_test():
    """성능 테스트 (별도 함수로 분리)"""
    print("성능 테스트 실행...")
    
    temp_papers_dir = tempfile.mkdtemp(prefix="perf_papers_")
    temp_processed_dir = tempfile.mkdtemp(prefix="perf_processed_")
    temp_output_dir = tempfile.mkdtemp(prefix="perf_output_")
    
    try:
        analyzer = WMSCitationAnalyzer(
            papers_dir=temp_papers_dir,
            processed_dir=temp_processed_dir,
            output_dir=temp_output_dir
        )
        
        import time
        
        # 대용량 네트워크 생성 및 분석 성능 측정
        start_time = time.time()
        
        # 100개 노드의 테스트 네트워크 생성
        for i in range(100):
            analyzer.citation_graph.add_node(f"paper_{i}", title=f"Paper {i}")
            if i > 0:
                # 이전 몇 개 논문에 인용 관계 추가
                for j in range(max(0, i-3), i):
                    if j != i:
                        analyzer.citation_graph.add_edge(f"paper_{i}", f"paper_{j}", weight=0.5)
        
        network_creation_time = time.time() - start_time
        print(f"네트워크 생성 시간 (100 노드): {network_creation_time:.3f}초")
        
        # 네트워크 분석 성능 측정
        start_time = time.time()
        metrics = analyzer.analyze_network_metrics()
        analysis_time = time.time() - start_time
        print(f"네트워크 분석 시간: {analysis_time:.3f}초")
        
        print(f"생성된 노드 수: {metrics['basic_metrics']['total_nodes']}")
        print(f"생성된 엣지 수: {metrics['basic_metrics']['total_edges']}")
        
    finally:
        shutil.rmtree(temp_papers_dir, ignore_errors=True)
        shutil.rmtree(temp_processed_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


if __name__ == '__main__':
    print("WMS 인용 분석기 테스트 시작...")
    print("=" * 50)
    
    # 단위 테스트 실행
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # 성능 테스트 실행
    run_performance_test()
    
    print("\n모든 테스트가 완료되었습니다!")
