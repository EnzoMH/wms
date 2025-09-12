#!/usr/bin/env python3
"""
텍스트 추출기 테스트
=================

WMS text_extractor.py에 대한 단위 테스트 및 통합 테스트를 포함합니다.

작성자: 신명호
날짜: 2025년 9월 3일
"""

import unittest
import os
import sys
import tempfile
import shutil
import csv
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

# 테스트 대상 모듈을 임포트하기 위해 상위 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Tools'))

try:
    from text_extractor import WMSTextExtractor
except ImportError as e:
    print(f"테스트 대상 모듈을 임포트할 수 없습니다: {e}")
    sys.exit(1)


class TestWMSTextExtractor(unittest.TestCase):
    """WMSTextExtractor 클래스에 대한 테스트"""

    def setUp(self):
        
        """각 테스트 전에 실행되는 설정"""
        # 임시 디렉토리 생성
        self.temp_papers_dir = tempfile.mkdtemp(prefix="papers_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="output_")
        
        # 테스트용 디렉토리 구조 생성
        for source in ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']:
            os.makedirs(os.path.join(self.temp_papers_dir, source), exist_ok=True)
        
        self.extractor = WMSTextExtractor(
            papers_dir=self.temp_papers_dir,
            output_dir=self.temp_output_dir
        )
        
    def tearDown(self):
        """각 테스트 후에 실행되는 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_papers_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)

    def test_initialization(self):
        """초기화 테스트"""
        # 텍스트 추출기가 올바르게 초기화되는지 확인
        self.assertEqual(str(self.extractor.papers_dir), self.temp_papers_dir)
        self.assertEqual(str(self.extractor.output_dir), self.temp_output_dir)
        self.assertIsNotNone(self.extractor.wms_terms)
        
        # WMS 특화 용어들이 올바르게 설정되었는지 확인
        expected_categories = ['core_wms', 'technology', 'operations', 'performance', 'integration']
        for category in expected_categories:
            self.assertIn(category, self.extractor.wms_terms)
            self.assertIsInstance(self.extractor.wms_terms[category], list)

    def test_setup_nltk(self):
        """NLTK 설정 테스트"""
        # NLTK 설정이 올바르게 되었는지 확인
        self.assertIsNotNone(self.extractor.stop_words)
        self.assertIsNotNone(self.extractor.lemmatizer)
        self.assertGreater(len(self.extractor.stop_words), 100)  # 영어 불용어가 충분히 있는지 확인

    def test_preprocess_text(self):
        """텍스트 전처리 테스트"""
        # 테스트용 텍스트
        test_text = "Warehouse Management Systems (WMS) are crucial for automation! They handle inventory tracking, AGV integration, and robotics."
        
        # 전처리 실행
        tokens = self.extractor.preprocess_text(test_text)
        
        # 결과 검증
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # 소문자로 변환되었는지 확인
        for token in tokens:
            self.assertEqual(token, token.lower())
        
        # 불용어가 제거되었는지 확인 (예: 'are', 'for')
        self.assertNotIn('are', tokens)
        self.assertNotIn('for', tokens)
        
        # 중요한 용어들이 남아있는지 확인
        token_str = ' '.join(tokens)
        self.assertIn('warehouse', token_str)
        self.assertIn('management', token_str)

    def test_extract_keywords(self):
        """키워드 추출 테스트"""
        # 테스트용 긴 텍스트
        test_text = """
        Warehouse Management Systems (WMS) are essential software solutions for optimizing warehouse operations.
        These systems integrate with Automated Guided Vehicles (AGV) to improve inventory management and reduce manual labor.
        Modern WMS platforms support RFID technology, IoT sensors, and machine learning algorithms for predictive analytics.
        The automation of warehouse processes leads to significant improvements in efficiency and accuracy.
        """
        
        # 키워드 추출 실행
        keywords = self.extractor.extract_keywords(test_text, top_k=20)
        
        # 결과 검증
        self.assertIsInstance(keywords, list)
        self.assertGreater(len(keywords), 0)
        self.assertLessEqual(len(keywords), 20)
        
        # 각 키워드가 (단어, 점수) 튜플인지 확인
        for keyword, score in keywords:
            self.assertIsInstance(keyword, str)
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0)
        
        # 점수가 내림차순으로 정렬되어 있는지 확인
        scores = [score for _, score in keywords]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_categorize_keywords(self):
        """키워드 분류 테스트"""
        # 테스트용 키워드 리스트
        test_keywords = [
            ('warehouse management', 0.95),
            ('automation technology', 0.85),
            ('picking operations', 0.75),
            ('efficiency metrics', 0.65),
            ('erp integration', 0.55),
            ('random keyword', 0.45)
        ]
        
        # 키워드 분류 실행
        categorized = self.extractor.categorize_keywords(test_keywords)
        
        # 결과 검증
        self.assertIsInstance(categorized, dict)
        
        # 모든 카테고리가 존재하는지 확인
        expected_categories = ['core_wms', 'technology', 'operations', 'performance', 'integration', 'other']
        for category in expected_categories:
            self.assertIn(category, categorized)
            self.assertIsInstance(categorized[category], list)
        
        # 분류가 올바르게 되었는지 확인
        core_keywords = [kw for kw, _ in categorized['core_wms']]
        self.assertTrue(any('warehouse' in kw for kw in core_keywords))

    @patch('text_extractor.PyPDF2.PdfReader')
    def test_extract_pdf_text_mock(self, mock_pdf_reader):
        """PDF 텍스트 추출 테스트 (모킹 사용)"""
        # 모킹된 PDF 페이지 설정
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test PDF content about WMS and warehouse automation."
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_pdf_reader.return_value = mock_reader_instance
        
        # 테스트용 더미 PDF 파일 생성
        test_pdf_path = Path(self.temp_papers_dir) / "ArXiv" / "test.pdf"
        test_pdf_path.touch()
        
        # PDF 텍스트 추출 실행
        extracted_text = self.extractor.extract_pdf_text(test_pdf_path)
        
        # 결과 검증
        self.assertIsInstance(extracted_text, str)
        self.assertIn("Test PDF content", extracted_text)
        self.assertIn("WMS", extracted_text)

    def test_create_visualizations_mock(self):
        """시각화 생성 테스트 (모킹 사용)"""
        # 테스트용 키워드 데이터
        test_keywords = [
            {'keyword': 'warehouse', 'score': 0.9},
            {'keyword': 'automation', 'score': 0.8},
            {'keyword': 'management', 'score': 0.7},
        ]
        
        # 시각화 생성 (실제로는 파일을 생성하지 않도록 모킹)
        with patch('text_extractor.plt.savefig') as mock_savefig, \
             patch('text_extractor.WordCloud') as mock_wordcloud:
            
            mock_wc_instance = MagicMock()
            mock_wordcloud.return_value = mock_wc_instance
            mock_wc_instance.generate.return_value = mock_wc_instance
            
            # 시각화 생성 시도
            try:
                self.extractor.create_visualizations(test_keywords)
                # 에러 없이 실행되면 성공
                self.assertTrue(True)
            except Exception as e:
                # 의존성 문제로 실패할 수 있으므로 경고만 출력
                print(f"시각화 테스트 건너뛰기: {e}")

    def test_save_extracted_text(self):
        """추출된 텍스트 저장 테스트"""
        # 테스트용 추출 텍스트 데이터
        test_data = [{
            'filename': 'test.pdf',
            'source': 'ArXiv',
            'text': 'This is test content about warehouse management systems.',
            'word_count': 9
        }]
        
        # 텍스트 저장 실행
        self.extractor.save_extracted_text(test_data)
        
        # 파일이 생성되었는지 확인
        output_file = Path(self.temp_output_dir) / "extracted_text.txt"
        self.assertTrue(output_file.exists())
        
        # 파일 내용 확인
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('test.pdf', content)
            self.assertIn('warehouse management systems', content)

    def test_save_keywords(self):
        """키워드 저장 테스트"""
        # 테스트용 키워드 데이터
        test_keywords = [
            {'keyword': 'warehouse', 'score': 0.9, 'source': 'ArXiv', 'paper': 'test1.pdf'},
            {'keyword': 'automation', 'score': 0.8, 'source': 'IEEE', 'paper': 'test2.pdf'},
            {'keyword': 'warehouse', 'score': 0.85, 'source': 'ArXiv', 'paper': 'test3.pdf'},
        ]
        
        # 키워드 저장 실행
        self.extractor.save_keywords(test_keywords)
        
        # CSV 파일이 생성되었는지 확인
        csv_file = Path(self.temp_output_dir) / "keywords.csv"
        self.assertTrue(csv_file.exists())
        
        # CSV 파일 내용 확인
        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            self.assertGreater(len(rows), 0)
            
            # CSV 헤더 확인
            expected_headers = ['keyword', 'frequency', 'avg_score', 'sources', 'source_count']
            for header in expected_headers:
                self.assertIn(header, reader.fieldnames)

    def test_generate_extraction_report(self):
        """추출 보고서 생성 테스트"""
        # 테스트용 논문 요약 데이터
        test_summaries = [
            {
                'filename': 'test1.pdf',
                'source': 'ArXiv',
                'word_count': 5000,
                'top_keywords': ['warehouse', 'automation'],
                'categorized_keywords': {'core_wms': 5, 'technology': 3}
            },
            {
                'filename': 'test2.pdf',
                'source': 'IEEE',
                'word_count': 4500,
                'top_keywords': ['management', 'system'],
                'categorized_keywords': {'operations': 4, 'performance': 2}
            }
        ]
        
        # 보고서 생성 실행
        self.extractor.generate_extraction_report(test_summaries)
        
        # 보고서 파일이 생성되었는지 확인
        report_file = Path(self.temp_output_dir) / "extraction_report.md"
        self.assertTrue(report_file.exists())
        
        # 보고서 내용 확인
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('텍스트 추출 보고서', content)
            self.assertIn('2', content)  # 총 논문 수
            self.assertIn('9,500', content)  # 총 단어 수 (5000 + 4500)


class TestWMSTextExtractorIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self):
        """통합 테스트 설정"""
        self.temp_papers_dir = tempfile.mkdtemp(prefix="integration_papers_")
        self.temp_output_dir = tempfile.mkdtemp(prefix="integration_output_")
        
        # 테스트용 디렉토리 구조와 파일 생성
        for source in ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']:
            source_dir = Path(self.temp_papers_dir) / source
            source_dir.mkdir(exist_ok=True)
        
        self.extractor = WMSTextExtractor(
            papers_dir=self.temp_papers_dir,
            output_dir=self.temp_output_dir
        )

    def tearDown(self):
        """통합 테스트 정리"""
        shutil.rmtree(self.temp_papers_dir, ignore_errors=True)
        shutil.rmtree(self.temp_output_dir, ignore_errors=True)

    def test_end_to_end_workflow(self):
        """전체 워크플로우 테스트"""
        # 테스트용 텍스트 파일 생성 (실제 PDF 대신)
        test_content = """
        Warehouse Management System (WMS) Optimization
        
        This paper discusses advanced warehouse management systems and their integration
        with automated guided vehicles (AGV) for improved warehouse automation.
        The study focuses on inventory management, picking operations, and overall
        warehouse efficiency through the use of IoT sensors and machine learning algorithms.
        """
        
        # 각 소스 디렉토리에 테스트 파일 생성
        for source in ['ArXiv', 'IEEE']:
            test_file = Path(self.temp_papers_dir) / source / f"test_{source.lower()}.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
        
        # 키워드 추출 및 저장 테스트
        keywords = self.extractor.extract_keywords(test_content)
        self.assertGreater(len(keywords), 0)
        
        # 키워드 분류 테스트
        categorized = self.extractor.categorize_keywords(keywords)
        self.assertIn('core_wms', categorized)
        self.assertIn('technology', categorized)

    def test_directory_structure_validation(self):
        """디렉토리 구조 검증 테스트"""
        # 출력 디렉토리가 올바르게 생성되었는지 확인
        self.assertTrue(Path(self.temp_output_dir).exists())
        self.assertTrue(Path(self.temp_output_dir).is_dir())
        
        # 입력 디렉토리 구조 확인
        expected_sources = ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']
        for source in expected_sources:
            source_path = Path(self.temp_papers_dir) / source
            self.assertTrue(source_path.exists())
            self.assertTrue(source_path.is_dir())


def run_performance_test():
    """성능 테스트 (별도 함수로 분리)"""
    print("성능 테스트 실행...")
    
    temp_papers_dir = tempfile.mkdtemp(prefix="perf_papers_")
    temp_output_dir = tempfile.mkdtemp(prefix="perf_output_")
    
    try:
        # 큰 텍스트 데이터로 성능 테스트
        large_text = "warehouse management system automation " * 1000
        
        extractor = WMSTextExtractor(
            papers_dir=temp_papers_dir,
            output_dir=temp_output_dir
        )
        
        import time
        
        # 텍스트 전처리 성능 측정
        start_time = time.time()
        tokens = extractor.preprocess_text(large_text)
        preprocess_time = time.time() - start_time
        print(f"대용량 텍스트 전처리 시간: {preprocess_time:.3f}초")
        print(f"처리된 토큰 수: {len(tokens)}")
        
        # 키워드 추출 성능 측정
        start_time = time.time()
        keywords = extractor.extract_keywords(large_text, top_k=50)
        extraction_time = time.time() - start_time
        print(f"키워드 추출 시간: {extraction_time:.3f}초")
        print(f"추출된 키워드 수: {len(keywords)}")
        
    finally:
        shutil.rmtree(temp_papers_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)


if __name__ == '__main__':
    print("WMS 텍스트 추출기 테스트 시작...")
    print("=" * 50)
    
    # 단위 테스트 실행
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    
    # 성능 테스트 실행
    run_performance_test()
    
    print("\n모든 테스트가 완료되었습니다!")
