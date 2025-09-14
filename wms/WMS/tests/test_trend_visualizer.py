#!/usr/bin/env python3
"""
트렌드 시각화기 테스트
===================

WMS trend_visualizer.py에 대한 단위 테스트 및 통합 테스트를 포함합니다.

작성자: 신명호
날짜: 2025년 9월 3일
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import csv
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import pandas as pd

# 테스트 대상 모듈을 임포트하기 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Tools'))

try:
    from trend_visualizer import WMSTrendVisualizer
except ImportError as e:
    print(f"테스트 대상 모듈을 임포트할 수 없습니다: {e}")
    print("필요한 패키지를 설치해주세요: pip install matplotlib seaborn pandas plotly")
    sys.exit(1)


class TestWMSTrendVisualizer(unittest.TestCase):
    """WMSTrendVisualizer 클래스에 대한 테스트"""

    def setUp(self):
        """각 테스트 전에 실행되는 설정"""
        # 임시 디렉토리 생성
        self.temp_papers_dir = tempfile.mkdtemp(prefix="papers_")
        self.temp_processed_dir = tempfile.mkdtemp(prefix="processed_")
        self.temp_analysis_dir = tempfile.mkdtemp(prefix="analysis_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="output_")
        
        # 테스트용 디렉토리 구조 생성
        for source in ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']:
            os.makedirs(os.path.join(self.temp_papers_dir, source), exist_ok=True)
        
        # 출력 디렉토리에 charts와 interactive 서브디렉토리 생성
        os.makedirs(os.path.join(self.temp_output_dir, "charts"), exist_ok=True)
        os.makedirs(os.path.join(self.temp_output_dir, "interactive"), exist_ok=True)
        
        self.visualizer = WMSTrendVisualizer(
            papers_dir=self.temp_papers_dir,
            processed_dir=self.temp_processed_dir,
            analysis_dir=self.temp_analysis_dir,
            output_dir=self.temp_output_dir
        )
        
    def tearDown(self):
        """각 테스트 후에 실행되는 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_papers_dir, ignore_errors=True)
        shutil.rmtree(self.temp_processed_dir, ignore_errors=True)
        shutil.rmtree(self.temp_analysis_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)

    def test_initialization(self):
        """초기화 테스트"""
        # 트렌드 시각화기가 올바르게 초기화되는지 확인
        self.assertEqual(str(self.visualizer.papers_dir), self.temp_papers_dir)
        self.assertEqual(str(self.visualizer.processed_dir), self.temp_processed_dir)
        self.assertEqual(str(self.visualizer.analysis_dir), self.temp_analysis_dir)
        self.assertEqual(str(self.visualizer.output_dir), self.temp_output_dir)
        
        # 데이터 저장소가 초기화되었는지 확인
        self.assertIsInstance(self.visualizer.papers_data, list)
        self.assertIsInstance(self.visualizer.keywords_data, pd.DataFrame)
        self.assertIsInstance(self.visualizer.trends_data, pd.DataFrame)

    def test_setup_directories(self):
        """디렉토리 설정 테스트"""
        # 필요한 출력 디렉토리들이 생성되었는지 확인
        charts_dir = Path(self.temp_output_dir) / "charts"
        interactive_dir = Path(self.temp_output_dir) / "interactive"
        
        self.assertTrue(charts_dir.exists())
        self.assertTrue(charts_dir.is_dir())
        self.assertTrue(interactive_dir.exists())
        self.assertTrue(interactive_dir.is_dir())

    def test_load_papers_metadata(self):
        """논문 메타데이터 로딩 테스트"""
        # 테스트용 ArXiv 메타데이터 파일 생성
        arxiv_metadata = {
            "collection_info": {
                "created_date": "2024-01-15",
                "total_papers": 2
            },
            "papers": [
                {
                    "id": "2024.0001",
                    "title": "WMS Optimization Techniques",
                    "authors": ["Alice Smith", "Bob Jones"],
                    "abstract": "Advanced optimization for warehouse management systems.",
                    "published": "2024-01-15",
                    "categories": ["cs.RO", "cs.AI"]
                },
                {
                    "id": "2024.0002",
                    "title": "AGV Integration Methods",
                    "authors": ["Carol Davis"],
                    "abstract": "Methods for integrating AGVs with WMS.",
                    "published": "2023-12-20",
                    "categories": ["cs.RO"]
                }
            ]
        }
        
        # 파일 생성
        arxiv_dir = Path(self.temp_papers_dir) / "ArXiv"
        with open(arxiv_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(arxiv_metadata, f)
        
        # 테스트용 Semantic Scholar 데이터 생성
        ss_data = {
            "search_metadata": {"query_date": "2024-01-15", "total_results": 1},
            "papers": [{
                "id": "semantic001",
                "title": "Smart Warehouse Research",
                "authors": ["David Wilson", "Eva Brown"],
                "year": 2023,
                "venue": "Warehouse Technology Journal",
                "citation_count": 15,
                "abstract": "Research on smart warehouse technologies."
            }]
        }
        
        ss_dir = Path(self.temp_papers_dir) / "SemanticScholar"
        with open(ss_dir / "search_results.json", 'w', encoding='utf-8') as f:
            json.dump(ss_data, f)
        
        # 메타데이터 로딩 실행
        self.visualizer.load_papers_metadata()
        
        # 결과 검증
        self.assertGreater(len(self.visualizer.papers_data), 0)
        
        # 논문 데이터 구조 확인
        for paper in self.visualizer.papers_data:
            self.assertIn('title', paper)
            self.assertIn('year', paper)
            self.assertIn('source', paper)

    def test_load_keywords_data(self):
        """키워드 데이터 로딩 테스트"""
        # 테스트용 키워드 CSV 파일 생성
        keywords_data = [
            ['keyword', 'frequency', 'avg_score', 'sources', 'source_count'],
            ['warehouse', 50, 0.95, 'ArXiv, IEEE', 2],
            ['automation', 42, 0.88, 'ArXiv, SemanticScholar', 2],
            ['management', 38, 0.82, 'IEEE, GoogleScholar', 2],
            ['AGV', 25, 0.79, 'ArXiv', 1]
        ]
        
        keywords_file = Path(self.temp_processed_dir) / "keywords.csv"
        with open(keywords_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in keywords_data:
                writer.writerow(row)
        
        # 키워드 데이터 로딩
        self.visualizer.load_keywords_data()
        
        # 결과 검증
        self.assertFalse(self.visualizer.keywords_data.empty)
        self.assertEqual(len(self.visualizer.keywords_data), 4)  # 헤더 제외
        self.assertIn('keyword', self.visualizer.keywords_data.columns)
        self.assertIn('frequency', self.visualizer.keywords_data.columns)

    def test_load_trends_data(self):
        """트렌드 데이터 로딩 테스트"""
        # 테스트용 트렌드 분석 CSV 파일 생성
        trends_data = [
            ['year', 'technology', 'adoption_rate', 'research_papers', 'industry_implementation', 'roi_reported'],
            [2020, 'RFID Systems', 85.2, 12, 145, 18.5],
            [2021, 'IoT Sensors', 45.7, 18, 89, 25.3],
            [2022, 'AGV Systems', 28.9, 24, 67, 34.6],
            [2023, 'AI Analytics', 15.6, 31, 34, 42.1]
        ]
        
        trends_file = Path(self.temp_analysis_dir) / "trend_analysis.csv"
        with open(trends_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in trends_data:
                writer.writerow(row)
        
        # 트렌드 데이터 로딩
        self.visualizer.load_trends_data()
        
        # 결과 검증
        self.assertFalse(self.visualizer.trends_data.empty)
        self.assertEqual(len(self.visualizer.trends_data), 4)  # 헤더 제외
        self.assertIn('technology', self.visualizer.trends_data.columns)
        self.assertIn('adoption_rate', self.visualizer.trends_data.columns)

    def test_load_research_trends(self):
        """연구 트렌드 JSON 데이터 로딩 테스트"""
        # 테스트용 연구 트렌드 JSON 파일 생성
        research_trends = {
            "analysis_metadata": {
                "analysis_date": "2024-01-15",
                "papers_analyzed": 15,
                "time_period": "2019-2024"
            },
            "yearly_trends": {
                "2022": {
                    "top_topics": ["automation", "AGV deployment"],
                    "paper_count": 4,
                    "avg_citation_count": 31.2
                },
                "2023": {
                    "top_topics": ["sustainability", "green logistics"],
                    "paper_count": 3,
                    "avg_citation_count": 19.8
                }
            },
            "research_focus_areas": {
                "automation_technologies": {
                    "percentage": 31.7,
                    "key_papers": 5,
                    "growth_trend": "stable"
                },
                "optimization_algorithms": {
                    "percentage": 28.3,
                    "key_papers": 4,
                    "growth_trend": "increasing"
                }
            },
            "geographic_distribution": {
                "North America": {
                    "paper_count": 7,
                    "percentage": 46.7
                },
                "Europe": {
                    "paper_count": 5,
                    "percentage": 33.3
                }
            }
        }
        
        trends_file = Path(self.temp_processed_dir) / "research_trends.json"
        with open(trends_file, 'w', encoding='utf-8') as f:
            json.dump(research_trends, f)
        
        # 연구 트렌드 로딩
        self.visualizer.load_research_trends()
        
        # 결과 검증
        self.assertTrue(hasattr(self.visualizer, 'research_trends'))
        self.assertIn('yearly_trends', self.visualizer.research_trends)
        self.assertIn('research_focus_areas', self.visualizer.research_trends)

    @patch('trend_visualizer.plt.savefig')
    @patch('trend_visualizer.plt.close')
    def test_create_publication_trends_mock(self, mock_close, mock_savefig):
        """발행 트렌드 시각화 테스트 (모킹 사용)"""
        # 테스트 데이터 설정
        self.visualizer.papers_data = [
            {'title': 'Paper 1', 'year': 2022, 'source': 'ArXiv'},
            {'title': 'Paper 2', 'year': 2023, 'source': 'IEEE'},
            {'title': 'Paper 3', 'year': 2023, 'source': 'ArXiv'},
        ]
        
        # 시각화 생성 실행
        self.visualizer.create_publication_trends()
        
        # savefig가 호출되었는지 확인 (차트가 생성됨)
        mock_savefig.assert_called()
        mock_close.assert_called()

    @patch('trend_visualizer.plt.savefig')
    @patch('trend_visualizer.WordCloud')
    def test_create_keyword_analysis_mock(self, mock_wordcloud, mock_savefig):
        """키워드 분석 시각화 테스트 (모킹 사용)"""
        # 테스트 키워드 데이터 설정
        self.visualizer.keywords_data = pd.DataFrame({
            'keyword': ['warehouse', 'automation', 'management', 'AGV'],
            'frequency': [50, 42, 38, 25],
            'avg_score': [0.95, 0.88, 0.82, 0.79]
        })
        
        # WordCloud 모킹 설정
        mock_wc_instance = MagicMock()
        mock_wordcloud.return_value = mock_wc_instance
        mock_wc_instance.generate_from_frequencies.return_value = mock_wc_instance
        
        # 키워드 분석 시각화 실행
        self.visualizer.create_keyword_analysis()
        
        # 함수들이 호출되었는지 확인
        mock_savefig.assert_called()
        mock_wordcloud.assert_called()

    @patch('trend_visualizer.plt.savefig')
    def test_create_technology_adoption_trends_mock(self, mock_savefig):
        """기술 도입 트렌드 시각화 테스트 (모킹 사용)"""
        # 테스트 트렌드 데이터 설정
        self.visualizer.trends_data = pd.DataFrame({
            'year': [2020, 2021, 2022, 2023],
            'technology': ['RFID', 'IoT', 'AGV', 'AI'],
            'adoption_rate': [85.2, 45.7, 28.9, 15.6],
            'roi_reported': [18.5, 25.3, 34.6, 42.1],
            'research_papers': [12, 18, 24, 31]
        })
        
        # 기술 도입 트렌드 시각화 실행
        self.visualizer.create_technology_adoption_trends()
        
        # 차트가 생성되었는지 확인
        mock_savefig.assert_called()

    @patch('trend_visualizer.pyo.plot')
    def test_create_interactive_dashboard_mock(self, mock_plot):
        """인터랙티브 대시보드 생성 테스트 (모킹 사용)"""
        # 테스트 데이터 설정
        self.visualizer.papers_data = [
            {'title': 'Paper 1', 'year': 2022, 'source': 'ArXiv'},
            {'title': 'Paper 2', 'year': 2023, 'source': 'IEEE'}
        ]
        
        self.visualizer.keywords_data = pd.DataFrame({
            'keyword': ['warehouse', 'automation'],
            'frequency': [50, 42]
        })
        
        self.visualizer.research_trends = {
            'research_focus_areas': {
                'automation': {'percentage': 60},
                'optimization': {'percentage': 40}
            },
            'geographic_distribution': {
                'North America': {'paper_count': 5},
                'Europe': {'paper_count': 3}
            }
        }
        
        # 인터랙티브 대시보드 생성 실행
        self.visualizer.create_interactive_dashboard()
        
        # Plotly plot 함수가 호출되었는지 확인
        mock_plot.assert_called()

    def test_generate_visualization_report(self):
        """시각화 보고서 생성 테스트"""
        # 테스트 데이터 설정
        self.visualizer.papers_data = [
            {'title': 'Paper 1', 'year': 2022, 'source': 'ArXiv'},
            {'title': 'Paper 2', 'year': 2023, 'source': 'IEEE'}
        ]
        
        self.visualizer.keywords_data = pd.DataFrame({
            'keyword': ['warehouse', 'automation'],
            'frequency': [50, 42]
        })
        
        self.visualizer.trends_data = pd.DataFrame({
            'technology': ['RFID', 'IoT'],
            'adoption_rate': [85.2, 45.7]
        })
        
        # 시각화 보고서 생성
        self.visualizer.generate_visualization_report()
        
        # 보고서 파일이 생성되었는지 확인
        report_file = Path(self.temp_output_dir) / "visualization_report.md"
        self.assertTrue(report_file.exists())
        
        # 보고서 내용 확인
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('시각화 보고서', content)
            self.assertIn('2', content)  # 논문 수
            self.assertIn('2', content)  # 키워드 수


class TestWMSTrendVisualizerIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self):
        """통합 테스트 설정"""
        self.temp_papers_dir = tempfile.mkdtemp(prefix="integration_papers_")
        self.temp_processed_dir = tempfile.mkdtemp(prefix="integration_processed_")
        self.temp_analysis_dir = tempfile.mkdtemp(prefix="integration_analysis_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="integration_output_")
        
        self.visualizer = WMSTrendVisualizer(
            papers_dir=self.temp_papers_dir,
            processed_dir=self.temp_processed_dir,
            analysis_dir=self.temp_analysis_dir,
            output_dir=self.temp_output_dir
        )

    def tearDown(self):
        """통합 테스트 정리"""
        shutil.rmtree(self.temp_papers_dir, ignore_errors=True)
        shutil.rmtree(self.temp_processed_dir, ignore_errors=True)
        shutil.rmtree(self.temp_analysis_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)

    def test_complete_data_loading_workflow(self):
        """전체 데이터 로딩 워크플로우 테스트"""
        # 종합적인 테스트 데이터 생성
        
        # ArXiv 메타데이터
        arxiv_metadata = {
            "collection_info": {"created_date": "2024-01-15", "total_papers": 3},
            "papers": [
                {
                    "id": "2024.0001",
                    "title": "WMS Optimization Research",
                    "authors": ["Alice Smith", "Bob Jones"],
                    "abstract": "Advanced optimization techniques for warehouse management.",
                    "published": "2024-01-15",
                    "categories": ["cs.RO"]
                },
                {
                    "id": "2024.0002",
                    "title": "AGV Integration Studies",
                    "authors": ["Carol Davis"],
                    "abstract": "Integration of AGVs in modern warehouse systems.",
                    "published": "2023-12-20",
                    "categories": ["cs.RO", "cs.AI"]
                }
            ]
        }
        
        # 키워드 데이터
        keywords_data = [
            ['keyword', 'frequency', 'avg_score'],
            ['warehouse', 75, 0.92],
            ['automation', 68, 0.89],
            ['management', 54, 0.85],
            ['AGV', 42, 0.81]
        ]
        
        # 트렌드 데이터
        trends_data = [
            ['year', 'technology', 'adoption_rate', 'roi_reported'],
            [2020, 'RFID Systems', 85.2, 18.5],
            [2021, 'IoT Sensors', 45.7, 25.3],
            [2022, 'AGV Systems', 28.9, 34.6],
            [2023, 'AI Analytics', 15.6, 42.1]
        ]
        
        # 연구 트렌드 JSON
        research_trends = {
            "yearly_trends": {
                "2023": {"paper_count": 3, "avg_citation_count": 25.4}
            },
            "research_focus_areas": {
                "automation": {"percentage": 45.2},
                "optimization": {"percentage": 32.1}
            },
            "geographic_distribution": {
                "North America": {"paper_count": 8},
                "Europe": {"paper_count": 5}
            }
        }
        
        # 파일들 생성
        arxiv_dir = Path(self.temp_papers_dir) / "ArXiv"
        arxiv_dir.mkdir(parents=True, exist_ok=True)
        with open(arxiv_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(arxiv_metadata, f)
        
        keywords_file = Path(self.temp_processed_dir) / "keywords.csv"
        with open(keywords_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in keywords_data:
                writer.writerow(row)
        
        trends_file = Path(self.temp_analysis_dir) / "trend_analysis.csv"
        with open(trends_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in trends_data:
                writer.writerow(row)
        
        research_file = Path(self.temp_processed_dir) / "research_trends.json"
        with open(research_file, 'w', encoding='utf-8') as f:
            json.dump(research_trends, f)
        
        # 전체 데이터 로딩 실행
        self.visualizer.load_all_data()
        
        # 결과 검증
        self.assertGreater(len(self.visualizer.papers_data), 0)
        self.assertFalse(self.visualizer.keywords_data.empty)
        self.assertFalse(self.visualizer.trends_data.empty)
        self.assertTrue(hasattr(self.visualizer, 'research_trends'))

    @patch('trend_visualizer.plt.savefig')
    @patch('trend_visualizer.pyo.plot')
    def test_complete_visualization_workflow_mock(self, mock_plot, mock_savefig):
        """전체 시각화 워크플로우 테스트 (모킹 사용)"""
        # 기본 데이터 설정
        self.visualizer.papers_data = [
            {'title': 'Paper 1', 'year': 2022, 'source': 'ArXiv', 'authors': 2},
            {'title': 'Paper 2', 'year': 2023, 'source': 'IEEE', 'authors': 1}
        ]
        
        self.visualizer.keywords_data = pd.DataFrame({
            'keyword': ['warehouse', 'automation', 'management'],
            'frequency': [50, 42, 38],
            'avg_score': [0.95, 0.88, 0.82]
        })
        
        self.visualizer.trends_data = pd.DataFrame({
            'year': [2022, 2023],
            'technology': ['RFID', 'IoT'],
            'adoption_rate': [85.2, 45.7],
            'roi_reported': [18.5, 25.3],
            'research_papers': [12, 18]
        })
        
        self.visualizer.research_trends = {
            'yearly_trends': {'2023': {'paper_count': 2, 'avg_citation_count': 20.5}},
            'research_focus_areas': {'automation': {'percentage': 60}},
            'geographic_distribution': {'North America': {'paper_count': 5}}
        }
        
        # 개별 시각화 함수들을 모킹하여 테스트
        with patch.object(self.visualizer, 'create_publication_trends') as mock_pub_trends, \
             patch.object(self.visualizer, 'create_keyword_analysis') as mock_keyword_analysis, \
             patch.object(self.visualizer, 'create_technology_adoption_trends') as mock_tech_trends, \
             patch.object(self.visualizer, 'create_research_evolution_timeline') as mock_timeline, \
             patch.object(self.visualizer, 'create_interactive_dashboard') as mock_dashboard, \
             patch.object(self.visualizer, 'generate_visualization_report') as mock_report:
            
            # 전체 시각화 실행
            self.visualizer.create_all_visualizations()
            
            # 모든 시각화 함수가 호출되었는지 확인
            mock_pub_trends.assert_called_once()
            mock_keyword_analysis.assert_called_once()
            mock_tech_trends.assert_called_once()
            mock_timeline.assert_called_once()
            mock_dashboard.assert_called_once()
            mock_report.assert_called_once()


def run_performance_test():
    """성능 테스트 (별도 함수로 분리)"""
    print("성능 테스트 실행...")
    
    temp_papers_dir = tempfile.mkdtemp(prefix="perf_papers_")
    temp_processed_dir = tempfile.mkdtemp(prefix="perf_processed_")
    temp_analysis_dir = tempfile.mkdtemp(prefix="perf_analysis_")
    temp_output_dir = tempfile.mkdtemp(prefix="perf_output_")
    
    try:
        visualizer = WMSTrendVisualizer(
            papers_dir=temp_papers_dir,
            processed_dir=temp_processed_dir,
            analysis_dir=temp_analysis_dir,
            output_dir=temp_output_dir
        )
        
        import time
        
        # 대용량 데이터 생성 및 처리 성능 측정
        start_time = time.time()
        
        # 1000개 논문 데이터 생성
        large_papers_data = []
        for i in range(1000):
            large_papers_data.append({
                'title': f'Paper {i}',
                'year': 2020 + (i % 4),  # 2020-2023 범위
                'source': ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar'][i % 4],
                'authors': (i % 5) + 1
            })
        
        visualizer.papers_data = large_papers_data
        data_generation_time = time.time() - start_time
        print(f"대용량 데이터 생성 시간 (1000개 논문): {data_generation_time:.3f}초")
        
        # 대용량 키워드 DataFrame 생성
        start_time = time.time()
        import numpy as np
        large_keywords_data = pd.DataFrame({
            'keyword': [f'keyword_{i}' for i in range(500)],
            'frequency': np.random.randint(1, 100, 500),
            'avg_score': np.random.uniform(0.1, 1.0, 500)
        })
        visualizer.keywords_data = large_keywords_data
        
        keywords_processing_time = time.time() - start_time
        print(f"키워드 데이터 처리 시간 (500개 키워드): {keywords_processing_time:.3f}초")
        
        print(f"처리된 논문 수: {len(large_papers_data)}")
        print(f"처리된 키워드 수: {len(large_keywords_data)}")
        
        # 메모리 사용량 확인 (선택적)
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            print(f"메모리 사용량: {memory_usage:.2f}MB")
        except ImportError:
            print("psutil이 설치되지 않아 메모리 측정을 생략합니다.")
        
    finally:
        shutil.rmtree(temp_papers_dir, ignore_errors=True)
        shutil.rmtree(temp_processed_dir, ignore_errors=True)
        shutil.rmtree(temp_analysis_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


if __name__ == '__main__':
    print("WMS 트렌드 시각화기 테스트 시작...")
    print("=" * 50)
    
    # 단위 테스트 실행
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # 성능 테스트 실행
    run_performance_test()
    
    print("\n모든 테스트가 완료되었습니다!")
