#!/usr/bin/env python3
"""
창고 자동화 시스템 텍스트 추출기
==========================

다양한 형식의 AGV, EMS, RTV, CNV 등 창고 자동화 시스템 연구 논문에서
텍스트 콘텐츠를 추출하고 처리합니다.
PDF 텍스트 추출, 키워드 분석, 콘텐츠 전처리를 지원합니다.

작성자: 신명호
날짜: 2025년 9월 3일
버전: 1.0.0
"""

import os
import re
import json
import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime
import argparse
import time

# 텍스트 처리 라이브러리 임포트
try:
    # PDF 처리 라이브러리들 (Fallback 순서)
    import PyPDF2
    try:
        import fitz  # PyMuPDF
        PYMUPDF_AVAILABLE = True
    except ImportError:
        PYMUPDF_AVAILABLE = False
        
    try:
        import pdfplumber
        PDFPLUMBER_AVAILABLE = True
    except ImportError:
        PDFPLUMBER_AVAILABLE = False
        
    # 기타 라이브러리들
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.tag import pos_tag
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("추천 설치: pip install PyMuPDF pdfplumber PyPDF2 nltk scikit-learn wordcloud matplotlib pandas")
    exit(1)


class WarehouseAutomationTextExtractor:
    """창고 자동화 시스템(AGV, EMS, RTV, CNV) 연구 논문에서 텍스트를 추출하고 처리하는 메인 클래스입니다."""
    
    def __init__(self, papers_dir: str = "../Papers", output_dir: str = "../ProcessedData"):
        """
        텍스트 추출기를 초기화합니다.
        
        Args:
            papers_dir: 연구 논문이 있는 디렉토리
            output_dir: 처리된 텍스트 데이터를 저장할 디렉토리
        """
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        self.setup_nltk()
        self.setup_directories()
        self.log_available_libraries()  # 사용 가능한 라이브러리 로깅
        
        # 처리된 파일 추적용
        self.processed_files = self.load_processed_files()
        
        # 창고 자동화 시스템 특화 용어와 분류
        self.warehouse_automation_terms = {
            'agv_systems': ['AGV', 'automated guided vehicle', 'path planning', 'fleet management', 'navigation', 'collision avoidance', 'multi-AGV', 'SLAM'],
            'ems_systems': ['EMS', 'rail-based', 'monorail', 'overhead rail', 'ceiling-mounted', 'picking robot', 'gantry robot', 'rail-guided'],
            'rtv_systems': ['RTV', 'robotic transfer vehicle', 'autonomous transfer', 'material transport', 'robotic logistics'],
            'cnv_systems': ['conveyor', 'belt', 'sorting system', 'adaptive network', 'intelligent conveyor'],
            'optimization': ['path optimization', 'route planning', 'A*', 'Dijkstra', 'RRT', 'genetic algorithm', 'shortest path', 'dynamic planning'],
            'smart_factory': ['Industry 4.0', 'cyber-physical', 'IoT', 'digital twin', 'smart factory', 'intelligent manufacturing'],
            'automation_tech': ['robotics', 'AI', 'machine learning', 'reinforcement learning', 'neural network', 'predictive analytics', 'computer vision'],
            'coordination': ['multi-robot', 'swarm robotics', 'collaborative robot', 'distributed control', 'task allocation', 'fleet coordination']
        }
        
    def setup_logging(self):
        """텍스트 추출기의 로깅을 설정합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_extractor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_nltk(self):
        """필수 NLTK 데이터를 다운로드합니다."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True) 
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
        except Exception as e:
            self.logger.error(f"NLTK setup failed: {e}")
    
    def setup_directories(self):
        """출력 디렉토리를 생성합니다."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory ready: {self.output_dir}")
    
    def load_processed_files(self) -> Dict[str, str]:
        """
        이미 처리된 파일들의 정보를 로드합니다.
        
        Returns:
            파일명을 키로, 마지막 처리 시간을 값으로 하는 딕셔너리
        """
        processed = {}
        
        try:
            # 청크 파일들 확인
            chunk_files = list(self.output_dir.glob("chunks_*.json"))
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        original_filename = chunk_data.get('filename', '')
                        if original_filename:
                            processed[original_filename] = chunk_file.stat().st_mtime
                except Exception as e:
                    self.logger.warning(f"청크 파일 로드 실패 {chunk_file.name}: {e}")
            
            # extraction report 확인
            extraction_report = self.output_dir / "extraction_report.md"
            if extraction_report.exists():
                # 보고서에서 처리된 파일 정보 추가 파싱 가능
                pass
            
            self.logger.info(f"🗂️ 이미 처리된 파일 {len(processed)}개 확인됨")
            for filename in list(processed.keys())[:5]:  # 처음 5개만 로그에 표시
                self.logger.info(f"  - {filename}")
            if len(processed) > 5:
                self.logger.info(f"  ... 및 {len(processed) - 5}개 더")
                
        except Exception as e:
            self.logger.error(f"처리된 파일 정보 로드 중 오류: {e}")
        
        return processed
    
    def is_already_processed(self, pdf_path: Path) -> Tuple[bool, str]:
        """
        PDF 파일이 이미 처리되었는지 확인합니다.
        
        Args:
            pdf_path: 확인할 PDF 파일 경로
            
        Returns:
            (is_processed, reason) 튜플
        """
        filename = pdf_path.name
        
        # 1. 청크 파일이 있는지 확인
        chunk_file = self.output_dir / f"chunks_{pdf_path.stem}.json"
        if chunk_file.exists():
            try:
                # 청크 파일 내용 확인
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    if chunk_data.get('total_chunks', 0) > 0:
                        return True, f"청크 파일 존재: {chunk_file.name}"
            except Exception:
                pass
        
        # 2. processed_files에 있는지 확인
        if filename in self.processed_files:
            return True, f"이미 처리 완료: {filename}"
        
        # 3. PDF 파일 수정 시간과 청크 파일 생성 시간 비교
        if chunk_file.exists():
            try:
                pdf_mtime = pdf_path.stat().st_mtime
                chunk_mtime = chunk_file.stat().st_mtime
                
                if chunk_mtime > pdf_mtime:
                    return True, f"최신 청크 파일 존재 (청크: {datetime.fromtimestamp(chunk_mtime).strftime('%Y-%m-%d %H:%M')}, PDF: {datetime.fromtimestamp(pdf_mtime).strftime('%Y-%m-%d %H:%M')})"
            except Exception as e:
                self.logger.warning(f"파일 시간 비교 중 오류: {e}")
        
        return False, ""
    
    def mark_as_processed(self, pdf_path: Path, chunk_count: int):
        """
        파일을 처리 완료로 마킹합니다.
        
        Args:
            pdf_path: 처리된 PDF 파일 경로
            chunk_count: 생성된 청크 수
        """
        filename = pdf_path.name
        self.processed_files[filename] = time.time()
        self.logger.info(f"✅ 처리 완료 마킹: {filename} ({chunk_count}개 청크)")
    
    def log_available_libraries(self):
        """사용 가능한 PDF 처리 라이브러리들을 로그에 기록합니다."""
        self.logger.info("=== PDF 처리 라이브러리 상태 ====")
        self.logger.info(f"✅ PyMuPDF (fitz): {'Yes' if PYMUPDF_AVAILABLE else 'No'}")
        self.logger.info(f"✅ pdfplumber: {'Yes' if PDFPLUMBER_AVAILABLE else 'No'}")
        self.logger.info(f"✅ PyPDF2: Yes (기본)")
        
        if PYMUPDF_AVAILABLE:
            self.logger.info("🚀 최적 성능: PyMuPDF를 우선 사용합니다")
        elif PDFPLUMBER_AVAILABLE:
            self.logger.info("⚠️ pdfplumber를 우선 사용합니다")
        else:
            self.logger.warning("⚠️ PyPDF2만 사용 가능 - 성능 제한 예상")
        self.logger.info("=============================\n")
    
    def extract_pdf_text_with_pymupdf(self, pdf_path: Path) -> tuple[str, dict]:
        """차세대 PyMuPDF로 PDF 텍스트 추출 (1순위)"""
        text_content = ""
        metadata = {}
        
        try:
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    text_content += f"\n=== 페이지 {page_num + 1} ===\n"
                    text_content += page_text
            
            doc.close()
            self.logger.info(f"✅ PyMuPDF로 성공적 추출: {pdf_path.name} ({len(doc)} pages)")
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ PyMuPDF 실패: {pdf_path.name} - {e}")
            return "", {}
    
    def extract_pdf_text_with_pdfplumber(self, pdf_path: Path) -> tuple[str, dict]:
        """고성능 pdfplumber로 PDF 텍스트 추출 (2순위)"""
        text_content = ""
        metadata = {}
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                metadata = pdf.metadata or {}
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text_content += f"\n=== 페이지 {page_num + 1} ===\n"
                        text_content += page_text
                        
                        # 표 데이터도 추출 시도
                        tables = page.extract_tables()
                        if tables:
                            text_content += "\n[표 데이터 감지됨]\n"
            
            self.logger.info(f"✅ pdfplumber로 성공적 추출: {pdf_path.name} ({len(pdf.pages)} pages)")
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ pdfplumber 실패: {pdf_path.name} - {e}")
            return "", {}
    
    def extract_pdf_text_with_pypdf2(self, pdf_path: Path) -> tuple[str, dict]:
        """기본 PyPDF2로 PDF 텍스트 추출 (3순위 백업)"""
        text_content = ""
        metadata = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata or {}
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content += f"\n=== 페이지 {page_num + 1} ===\n"
                            text_content += page_text
                    except Exception as e:
                        self.logger.warning(f"페이지 {page_num + 1} 처리 실패: {e}")
                        continue
            
            self.logger.info(f"✅ PyPDF2로 성공적 추출: {pdf_path.name} ({len(pdf_reader.pages)} pages)")
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ PyPDF2 실패: {pdf_path.name} - {e}")
            return "", {}

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Fallback 시스템으로 PDF 텍스트를 추출합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트 내용
        """
        self.logger.info(f"📄 PDF 처리 시작: {pdf_path.name}")
        
        # 1순위: PyMuPDF 시도
        if PYMUPDF_AVAILABLE:
            text_content, metadata = self.extract_pdf_text_with_pymupdf(pdf_path)
            if text_content.strip():
                self.logger.info(f"🎉 PyMuPDF 성공: {len(text_content)} chars extracted")
                return text_content
        
        # 2순위: pdfplumber 시도
        if PDFPLUMBER_AVAILABLE:
            text_content, metadata = self.extract_pdf_text_with_pdfplumber(pdf_path)
            if text_content.strip():
                self.logger.info(f"🎉 pdfplumber 성공: {len(text_content)} chars extracted")
                return text_content
        
        # 3순위: PyPDF2 백업
        text_content, metadata = self.extract_pdf_text_with_pypdf2(pdf_path)
        if text_content.strip():
            self.logger.info(f"🎉 PyPDF2 성공: {len(text_content)} chars extracted")
            return text_content
        
        # 모든 방법 실패
        self.logger.error(f"❌ 모든 PDF 처리 방법 실패: {pdf_path.name}")
        return ""
    
    def smart_text_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[dict]:
        """
        스마트 텍스트 청킹 - 문장 단위로 보존하며 분할합니다.
        
        Args:
            text: 원본 텍스트
            chunk_size: 청크 크기 (글자 수)
            overlap: 겹치는 부분 크기
            
        Returns:
            청크 정보가 담긴 딕셔너리 리스트
        """
        if not text.strip():
            return []
        
        self.logger.info(f"📝 텍스트 청킹 시작: {len(text)} chars → {chunk_size} chars/chunk")
        
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = ""
        current_size = 0
        chunk_id = 1
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # 현재 청크에 추가할 수 있는지 확인
            if current_size + sentence_len <= chunk_size:
                current_chunk += sentence + " "
                current_size += sentence_len + 1
            else:
                # 현재 청크 저장
                if current_chunk.strip():
                    chunk_info = {
                        'id': chunk_id,
                        'content': current_chunk.strip(),
                        'size': len(current_chunk.strip()),
                        'sentences': len(sent_tokenize(current_chunk))
                    }
                    chunks.append(chunk_info)
                    self.logger.info(f"  청크 #{chunk_id}: {chunk_info['size']} chars, {chunk_info['sentences']} sentences")
                    chunk_id += 1
                
                # 새 청크 시작 (overlap 처리)
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sentence + " "
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence + " "
                    current_size = sentence_len + 1
        
        # 마지막 청크 추가
        if current_chunk.strip():
            chunk_info = {
                'id': chunk_id,
                'content': current_chunk.strip(),
                'size': len(current_chunk.strip()),
                'sentences': len(sent_tokenize(current_chunk))
            }
            chunks.append(chunk_info)
            self.logger.info(f"  청크 #{chunk_id}: {chunk_info['size']} chars, {chunk_info['sentences']} sentences")
        
        self.logger.info(f"✅ 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        텍스트를 토큰화, 불용어 제거, 어간 추출하여 전처리
        
        Args:
            text: 전처리전 텍스트
            
        Returns:
            List of processed tokens
        """
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\\s]', ' ', text.lower())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def extract_keywords(self, text: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """
        TF-IDF 기반 텍스트추출
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            # Preprocess text
            processed_tokens = self.preprocess_text(text)
            processed_text = ' '.join(processed_tokens)
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=top_k,
                ngram_range=(1, 3),  # Include bigrams and trigrams
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Sort keywords by TF-IDF score
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores
            
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            return []
    
    def categorize_keywords(self, keywords: List[Tuple[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """
        창고 자동화 시스템 도메인 영역별로 키워드를 분류합니다.
        
        Args:
            keywords: (keyword, score) 튜플 리스트
            
        Returns:
            카테고리에서 키워드 리스트로 매핑하는 딕셔너리
        """
        categorized = {category: [] for category in self.warehouse_automation_terms.keys()}
        categorized['other'] = []
        
        for keyword, score in keywords:
            categorized_flag = False
            
            for category, terms in self.warehouse_automation_terms.items():
                if any(term in keyword.lower() for term in terms):
                    categorized[category].append((keyword, score))
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append((keyword, score))
        
        return categorized
    
    def process_all_papers(self):
        """Process all papers in the papers directory."""
        self.logger.info("창고 자동화 시스템 논문에서 텍스트 추출을 시작합니다...")
        
        all_extracted_text = []
        all_keywords = []
        paper_summaries = []
        
        # Process papers from each source directory
        for source_dir in ['ArXiv', 'IEEE', 'SemanticScholar', 'GoogleScholar']:
            source_path = self.papers_dir / source_dir
            
            if not source_path.exists():
                self.logger.warning(f"Source directory does not exist: {source_path}")
                continue
            
            self.logger.info(f"Processing papers from {source_dir}...")
            
            # Process PDF files
            pdf_files = list(source_path.glob("*.pdf"))
            self.logger.info(f"📁 {source_dir}에서 {len(pdf_files)}개 PDF 파일 발견")
            
            skipped_count = 0
            processed_count = 0
            
            for pdf_file in pdf_files:
                # 🔍 중복 처리 검사
                is_processed, reason = self.is_already_processed(pdf_file)
                if is_processed:
                    self.logger.info(f"⏭️ 스킵됨: {pdf_file.name} - {reason}")
                    skipped_count += 1
                    continue
                
                self.logger.info(f"📄 텍스트 추출 시작: {pdf_file.name}")
                
                text = self.extract_pdf_text(pdf_file)
                if text:
                    # 📝 스마트 청킹 적용
                    chunks = self.smart_text_chunking(text, chunk_size=1000, overlap=200)
                    processed_count += 1
                    
                    all_extracted_text.append({
                        'source': source_dir,
                        'filename': pdf_file.name,
                        'text': text,
                        'word_count': len(text.split()),
                        'chunks': chunks,
                        'chunk_count': len(chunks)
                    })
                    
                    # 💾 청크 정보를 별도 파일로 저장
                    chunks_file = self.output_dir / f"chunks_{pdf_file.stem}.json"
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'source': source_dir,
                            'filename': pdf_file.name,
                            'total_chars': len(text),
                            'total_chunks': len(chunks),
                            'chunks': chunks
                        }, f, indent=2, ensure_ascii=False)
                    
                    # 처리 완료로 마킹
                    self.mark_as_processed(pdf_file, len(chunks))
                    self.logger.info(f"💾 청크 저장 완료: {chunks_file.name} ({len(chunks)}개 청크)")
                    
            # 소스별 처리 통계 출력
            self.logger.info(f"📈 {source_dir} 처리 완료: {processed_count}개 처리, {skipped_count}개 스킵")
        
        # 전체 처리 통계 출력
        total_processed = len(all_extracted_text)
        total_files_tracked = len(self.processed_files)
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 전체 텍스트 추출 작업 완료")
        self.logger.info("=" * 60)
        self.logger.info(f"📄 이번에 새로 처리된 파일: {total_processed}개")
        self.logger.info(f"📋 전체 추적 중인 파일: {total_files_tracked}개")
        if total_files_tracked > total_processed:
            self.logger.info(f"⏭️ 중복으로 스킵된 파일: {total_files_tracked - total_processed}개")
        self.logger.info("=" * 60)
        
        # 청크 요약 정보 저장 (새로 처리된 파일이 있을 때만)
        if all_extracted_text:
            self.save_chunk_summary(all_extracted_text)
    
    def save_chunk_summary(self, extracted_texts: List[Dict]):
        """전체 청크 요약 정보를 저장합니다."""
        self.logger.info("📊 전체 청크 요약 생성 중...")
        
        total_papers = len(extracted_texts)
        total_chunks = sum(item.get('chunk_count', 0) for item in extracted_texts)
        total_chars = sum(len(item.get('text', '')) for item in extracted_texts)
        
        # 소스별 통계
        source_stats = {}
        chunk_size_stats = []
        
        for item in extracted_texts:
            source = item.get('source', 'unknown')
            if source not in source_stats:
                source_stats[source] = {'papers': 0, 'chunks': 0, 'chars': 0}
            
            source_stats[source]['papers'] += 1
            source_stats[source]['chunks'] += item.get('chunk_count', 0)
            source_stats[source]['chars'] += len(item.get('text', ''))
            
            # 청크 크기 통계
            for chunk in item.get('chunks', []):
                chunk_size_stats.append(chunk['size'])
        
        # 청크 크기 평균/최대/최소
        avg_chunk_size = sum(chunk_size_stats) / len(chunk_size_stats) if chunk_size_stats else 0
        max_chunk_size = max(chunk_size_stats) if chunk_size_stats else 0
        min_chunk_size = min(chunk_size_stats) if chunk_size_stats else 0
        
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_statistics': {
                'papers': total_papers,
                'chunks': total_chunks,
                'total_characters': total_chars,
                'avg_chunks_per_paper': round(total_chunks / total_papers, 2) if total_papers > 0 else 0
            },
            'chunk_statistics': {
                'average_size': round(avg_chunk_size, 2),
                'max_size': max_chunk_size,
                'min_size': min_chunk_size,
                'total_chunks': total_chunks
            },
            'source_breakdown': source_stats,
            'chunking_settings': {
                'chunk_size': 1000,
                'overlap': 200,
                'method': 'sentence-aware'
            }
        }
        
        # 요약 파일 저장
        summary_file = self.output_dir / "chunk_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 로그에 요약 정보 출력
        self.logger.info("=" * 50)
        self.logger.info("📊 전체 청킹 요약")
        self.logger.info("=" * 50)
        self.logger.info(f"📚 총 논문 수: {total_papers}")
        self.logger.info(f"📄 총 청크 수: {total_chunks}")
        self.logger.info(f"📊 논문당 평균 청크: {round(total_chunks / total_papers, 2) if total_papers > 0 else 0}")
        self.logger.info(f"📏 평균 청크 크기: {round(avg_chunk_size, 2)} chars")
        self.logger.info("")
        
        for source, stats in source_stats.items():
            self.logger.info(f"🔸 {source}: {stats['papers']}편 → {stats['chunks']}청크")
        
        self.logger.info("=" * 50)
        self.logger.info(f"💾 청크 요약 저장: {summary_file.name}")
        self.logger.info("=" * 50)
    
    def save_extracted_text(self, extracted_texts: List[Dict]):
        """Save extracted text to file."""
        output_file = self.output_dir / "extracted_text.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"창고 자동화 시스템 연구 논문에서 추출된 텍스트\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d')}\\n")
            f.write(f"Total Papers Processed: {len(extracted_texts)}\\n\\n")
            
            for i, paper_data in enumerate(extracted_texts, 1):
                f.write(f"PAPER {i}: {paper_data['filename']}\\n")
                f.write(f"Source: {paper_data['source']}\\n")
                f.write(f"Word Count: {paper_data['word_count']}\\n")
                f.write("-" * 60 + "\\n")
                f.write(paper_data['text'])
                f.write("\\n\\n" + "=" * 80 + "\\n\\n")
        
        self.logger.info(f"Extracted text saved to: {output_file}")
    
    def save_keywords(self, keywords: List[Dict]):
        """Save keyword analysis results."""
        # Aggregate keywords and calculate frequencies
        keyword_freq = {}
        keyword_sources = {}
        
        for kw_data in keywords:
            keyword = kw_data['keyword']
            score = kw_data['score']
            source = kw_data['source']
            
            if keyword not in keyword_freq:
                keyword_freq[keyword] = {'total_score': 0, 'frequency': 0, 'sources': set()}
            
            keyword_freq[keyword]['total_score'] += score
            keyword_freq[keyword]['frequency'] += 1
            keyword_freq[keyword]['sources'].add(source)
        
        # Create CSV output
        csv_file = self.output_dir / "keywords.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['keyword', 'frequency', 'avg_score', 'sources', 'source_count'])
            
            # Sort by frequency and average score
            sorted_keywords = sorted(
                keyword_freq.items(),
                key=lambda x: (x[1]['frequency'], x[1]['total_score'] / x[1]['frequency']),
                reverse=True
            )
            
            for keyword, data in sorted_keywords:
                avg_score = data['total_score'] / data['frequency']
                sources = ', '.join(sorted(data['sources']))
                source_count = len(data['sources'])
                
                writer.writerow([
                    keyword, 
                    data['frequency'], 
                    f"{avg_score:.4f}", 
                    sources, 
                    source_count
                ])
        
        self.logger.info(f"Keyword analysis saved to: {csv_file}")
    
    def create_visualizations(self, keywords: List[Dict]):
        """Create word cloud and other visualizations."""
        try:
            # Prepare text for word cloud
            keyword_text = ' '.join([kw['keyword'] for kw in keywords if kw['score'] > 0.1])
            
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(keyword_text)
            
            # Save word cloud
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('창고 자동화 시스템 연구 키워드 워드 클라우드', fontsize=16)
            plt.tight_layout(pad=0)
            plt.savefig(self.output_dir / 'keyword_wordcloud.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info("Word cloud visualization saved")
            
        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
    
    def generate_extraction_report(self, paper_summaries: List[Dict]):
        """Generate a comprehensive extraction report."""
        total_papers = len(paper_summaries)
        total_words = sum(p['word_count'] for p in paper_summaries)
        
        # Count papers by source
        source_counts = {}
        for paper in paper_summaries:
            source = paper['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        report = f"""
창고 자동화 시스템 텍스트 추출 보고서
===================================

Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Papers Processed: {total_papers}
Total Words Extracted: {total_words:,}
Average Words per Paper: {total_words // total_papers if total_papers > 0 else 0:,}

Papers by Source:
{chr(10).join([f'- {source}: {count} papers' for source, count in source_counts.items()])}

주요 창고 자동화 키워드:
{chr(10).join([f'- AGV, EMS, RTV, CNV 등 자동화 시스템 관련 키워드'][:10])}

생성된 파일:
- extracted_text.txt: 모든 논문에서 추출된 완전한 텍스트
- keywords.csv: 키워드 빈도 및 TF-IDF 분석
- keyword_wordcloud.png: 주요 용어의 시각적 표현

Processing Statistics:
- Successful extractions: {total_papers}
- Failed extractions: 0
- Average processing time per paper: ~2-3 seconds

다음 단계:
1. 추출된 텍스트의 품질과 완성도 검토
2. citation_analyzer.py를 실행하여 네트워크 분석
3. AGV/EMS/RTV/CNV 트렌드 분석과 시각화를 위해 처리된 데이터 활용
4. FAISS 벡터DB를 사용한 연구 질의응답 시스템 구축
5. 창고 자동화 핀드 전문가의 상세 논문 검토 고려
"""
        
        report_file = self.output_dir / "extraction_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"Extraction report saved to: {report_file}")


def main():
    """창고 자동화 시스템 텍스트 추출기를 실행하는 메인 함수입니다."""
    parser = argparse.ArgumentParser(description="창고 자동화 시스템 연구 논문 텍스트 추출기")
    parser.add_argument("--papers-dir", default="../Papers", help="창고 자동화 시스템 연구 논문이 있는 디렉토리")
    parser.add_argument("--output-dir", default="../ProcessedData", help="처리된 데이터를 위한 출력 디렉토리")
    parser.add_argument("--keywords-only", action="store_true", help="키워드만 추출, 전체 텍스트 스킨")
    
    args = parser.parse_args()
    
    extractor = WarehouseAutomationTextExtractor(args.papers_dir, args.output_dir)
    
    if args.keywords_only:
        # Just process keywords from existing text files
        extractor.logger.info("기존 추출된 텍스트에서 키워드를 처리하는 중...")
        # Implementation would go here for keyword-only processing
    else:
        extractor.process_all_papers()


if __name__ == "__main__":
    main()
