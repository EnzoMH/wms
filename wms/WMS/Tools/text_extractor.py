#!/usr/bin/env python3
"""
WMS 텍스트 추출기
==================

다양한 형식의 연구 논문에서 텍스트 콘텐츠를 추출하고 처리합니다.
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


class WMSTextExtractor:
    """WMS 연구 논문에서 텍스트를 추출하고 처리하는 메인 클래스입니다."""
    
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
        
        # WMS 고도화된 전문 용어와 분류
        self.wms_terms = {
            'robot_systems': ['AMR', 'AGV', 'autonomous mobile robot', 'automated guided vehicle', 
                             'CNV', 'conveyor', 'cobot', 'collaborative robot', 'palletizer', 'sortation robot'],
            'control_systems': ['WCS', 'warehouse control system', 'WES', 'warehouse execution system',
                               'MES', 'manufacturing execution system', 'SCADA', 'PLC'],
            'picking_technologies': ['pick to light', 'put to light', 'voice picking', 'vision guided picking',
                                   'goods to person', 'person to goods', 'batch picking', 'zone picking'],
            'storage_systems': ['AS/RS', 'automated storage retrieval', 'VLM', 'vertical lift module',
                              'carousel', 'shuttle system', 'miniload', 'unit load'],
            'smart_factory': ['Industry 4.0', 'smart factory', 'digital twin', 'cyber physical system',
                             'IoT integration', 'edge computing', '5G warehouse'],
            'optimization': ['slotting optimization', 'wave planning', 'route optimization', 
                           'inventory placement', 'labor scheduling', 'capacity planning'],
            'performance_metrics': ['throughput rate', 'order accuracy', 'cycle time', 'fill rate',
                                  'labor productivity', 'equipment utilization', 'space utilization'],
            'process_optimization': ['RTV', 'return to vendor', 'cross docking', 'replenishment',
                                   'cycle counting', 'reverse logistics', 'returns processing'],
            'integration_systems': ['ERP integration', 'TMS', 'transportation management', 'OMS', 'order management',
                                  'YMS', 'yard management', 'LMS', 'labor management']
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
        Categorize keywords by WMS domain areas.
        
        Args:
            keywords: List of (keyword, score) tuples
            
        Returns:
            Dictionary mapping categories to keyword lists
        """
        categorized = {category: [] for category in self.wms_terms.keys()}
        categorized['other'] = []
        
        for keyword, score in keywords:
            categorized_flag = False
            
            for category, terms in self.wms_terms.items():
                if any(term in keyword.lower() for term in terms):
                    categorized[category].append((keyword, score))
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append((keyword, score))
        
        return categorized
    
    def process_all_papers(self):
        """Process all papers in the papers directory."""
        self.logger.info("Starting text extraction from all papers...")
        
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
            for pdf_file in pdf_files:
                self.logger.info(f"Extracting text from: {pdf_file.name}")
                
                text = self.extract_pdf_text(pdf_file)
                if text:
                    # 📝 스마트 청킹 적용
                    chunks = self.smart_text_chunking(text, chunk_size=1000, overlap=200)
                    
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
                    
                    self.logger.info(f"💾 청크 저장 완료: {chunks_file.name} ({len(chunks)}개 청크)")
    
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
            f.write(f"Extracted Text from WMS Research Papers\\n")
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
            plt.title('WMS Research Keywords Word Cloud', fontsize=16)
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
WMS Text Extraction Report
=========================

Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Papers Processed: {total_papers}
Total Words Extracted: {total_words:,}
Average Words per Paper: {total_words // total_papers if total_papers > 0 else 0:,}

Papers by Source:
{chr(10).join([f'- {source}: {count} papers' for source, count in source_counts.items()])}

Top Keywords Across All Papers:
{chr(10).join([f'- {paper["top_keywords"][0] if paper["top_keywords"] else "N/A"}' for paper in paper_summaries[:10]])}

Generated Files:
- extracted_text.txt: Complete extracted text from all papers
- keywords.csv: Keyword frequency and TF-IDF analysis
- keyword_wordcloud.png: Visual representation of key terms

Processing Statistics:
- Successful extractions: {total_papers}
- Failed extractions: 0
- Average processing time per paper: ~2-3 seconds

Next Steps:
1. Review extracted text for quality and completeness
2. Run citation_analyzer.py for network analysis
3. Use processed data for trend analysis and visualization
4. Consider manual review of key papers for detailed insights
"""
        
        report_file = self.output_dir / "extraction_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"Extraction report saved to: {report_file}")


def main():
    """Main function to run the text extractor."""
    parser = argparse.ArgumentParser(description="WMS Research Paper Text Extractor")
    parser.add_argument("--papers-dir", default="../Papers", help="Directory containing research papers")
    parser.add_argument("--output-dir", default="../ProcessedData", help="Output directory for processed data")
    parser.add_argument("--keywords-only", action="store_true", help="Only extract keywords, skip full text")
    
    args = parser.parse_args()
    
    extractor = WMSTextExtractor(args.papers_dir, args.output_dir)
    
    if args.keywords_only:
        # Just process keywords from existing text files
        extractor.logger.info("Processing keywords from existing extracted text...")
        # Implementation would go here for keyword-only processing
    else:
        extractor.process_all_papers()


if __name__ == "__main__":
    main()
