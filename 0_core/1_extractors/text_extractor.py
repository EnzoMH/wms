#!/usr/bin/env python3
"""
    
==========================

  AGV, EMS, RTV, CNV      
   .
PDF  ,  ,   .

: 
: 2025 9 3
: 1.0.0
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

#    
try:
    # PDF   (Fallback )
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
        
    #  
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
    print(f"  . : {e}")
    print(" : pip install PyMuPDF pdfplumber PyPDF2 nltk scikit-learn wordcloud matplotlib pandas")
    exit(1)


class WarehouseAutomationTextExtractor:
    """  (AGV, EMS, RTV, CNV)       ."""
    
    def __init__(self, papers_dir: str = "../../1_data/0_crawled", output_dir: str = "../../1_data/1_chunks"):
        """
          .
        
        Args:
            papers_dir:    
            output_dir:     
        """
        self.papers_dir = Path(papers_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        self.setup_nltk()
        self.setup_directories()
        self.log_available_libraries()  #    
        
        #   
        self.processed_files = self.load_processed_files()
        
        #      
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
        """   ."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('text_extractor.log', encoding='utf-8', errors='ignore'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_nltk(self):
        """ NLTK  ."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True) 
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            
        except Exception as e:
            self.logger.error(f"NLTK setup failed: {e}")
    
    def setup_directories(self):
        """  ."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory ready: {self.output_dir}")
    
    def load_processed_files(self) -> Dict[str, str]:
        """
            .
        
        Returns:
             ,      
        """
        processed = {}
        
        try:
            #   
            chunk_files = list(self.output_dir.glob("chunks_*.json"))
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8', errors='ignore') as f:
                        chunk_data = json.load(f)
                        original_filename = chunk_data.get('filename', '')
                        if original_filename:
                            processed[original_filename] = chunk_file.stat().st_mtime
                except Exception as e:
                    self.logger.warning(f"    {chunk_file.name}: {e}")
            
            # extraction report 
            extraction_report = self.output_dir / "extraction_report.md"
            if extraction_report.exists():
                #       
                pass
            
            self.logger.info(f"[INFO]    {len(processed)} ")
            for filename in list(processed.keys())[:5]:  #  5  
                self.logger.info(f"  - {filename}")
            if len(processed) > 5:
                self.logger.info(f"  ...  {len(processed) - 5} ")
                
        except Exception as e:
            self.logger.error(f"     : {e}")
        
        return processed
    
    def is_already_processed(self, pdf_path: Path) -> Tuple[bool, str]:
        """
        PDF    .
        
        Args:
            pdf_path:  PDF  
            
        Returns:
            (is_processed, reason) 
        """
        filename = pdf_path.name
        
        # 1.    
        chunk_file = self.output_dir / f"chunks_{pdf_path.stem}.json"
        if chunk_file.exists():
            try:
                #    
                with open(chunk_file, 'r', encoding='utf-8', errors='ignore') as f:
                    chunk_data = json.load(f)
                    if chunk_data.get('total_chunks', 0) > 0:
                        return True, f"  : {chunk_file.name}"
            except Exception:
                pass
        
        # 2. processed_files  
        if filename in self.processed_files:
            return True, f"  : {filename}"
        
        # 3. PDF        
        if chunk_file.exists():
            try:
                pdf_mtime = pdf_path.stat().st_mtime
                chunk_mtime = chunk_file.stat().st_mtime
                
                if chunk_mtime > pdf_mtime:
                    return True, f"    (: {datetime.fromtimestamp(chunk_mtime).strftime('%Y-%m-%d %H:%M')}, PDF: {datetime.fromtimestamp(pdf_mtime).strftime('%Y-%m-%d %H:%M')})"
            except Exception as e:
                self.logger.warning(f"    : {e}")
        
        return False, ""
    
    def mark_as_processed(self, pdf_path: Path, chunk_count: int):
        """
           .
        
        Args:
            pdf_path:  PDF  
            chunk_count:   
        """
        filename = pdf_path.name
        self.processed_files[filename] = time.time()
        self.logger.info(f"  : {filename} ({chunk_count} )")
    
    def log_available_libraries(self):
        """  PDF    ."""
        self.logger.info("=== PDF    ====")
        self.logger.info(f"PyMuPDF (fitz): {'Yes' if PYMUPDF_AVAILABLE else 'No'}")
        self.logger.info(f"dfplumber: {'Yes' if PDFPLUMBER_AVAILABLE else 'No'}")
        self.logger.info(f"PyPDF2: Yes ()")
        
        if PYMUPDF_AVAILABLE:
            self.logger.info(" : PyMuPDF  ")
        elif PDFPLUMBER_AVAILABLE:
            self.logger.info("pdfplumber  ")
        else:
            self.logger.warning("PyPDF2   -   ")
        self.logger.info("=============================\n")
    
    def extract_pdf_text_with_pymupdf(self, pdf_path: Path) -> tuple[str, dict]:
        """ PyMuPDF PDF   (1)"""
        text_content = ""
        metadata = {}
        
        try:
            doc = fitz.open(str(pdf_path))
            metadata = doc.metadata
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    text_content += f"\n===  {page_num + 1} ===\n"
                    text_content += page_text
            
            doc.close()
            self.logger.info(f"PyMuPDF  : {pdf_path.name} ({len(doc)} pages)")
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"[WARN] PyMuPDF : {pdf_path.name} - {e}")
            return "", {}
    
    def extract_pdf_text_with_pdfplumber(self, pdf_path: Path) -> tuple[str, dict]:
        """ pdfplumber PDF   (2)"""
        text_content = ""
        metadata = {}
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                metadata = pdf.metadata or {}
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    
                    if page_text and page_text.strip():
                        text_content += f"\n===  {page_num + 1} ===\n"
                        text_content += page_text
                        
                        #    
                        tables = page.extract_tables()
                        if tables:
                            text_content += "\n[  ]\n"
            
            self.logger.info(f"pdfplumber  : {pdf_path.name} ({len(pdf.pages)} pages)")
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"[WARN] pdfplumber : {pdf_path.name} - {e}")
            return "", {}
    
    def extract_pdf_text_with_pypdf2(self, pdf_path: Path) -> tuple[str, dict]:
        """ PyPDF2 PDF   (3 )"""
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
                            text_content += f"\n===  {page_num + 1} ===\n"
                            text_content += page_text
                    except Exception as e:
                        self.logger.warning(f" {page_num + 1}  : {e}")
                        continue
            
            self.logger.info(f"PyPDF2  : {pdf_path.name} ({len(pdf_reader.pages)} pages)")
            return text_content, metadata
            
        except Exception as e:
            self.logger.warning(f"[WARN] PyPDF2 : {pdf_path.name} - {e}")
            return "", {}

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """
        Fallback  PDF  .
        
        Args:
            pdf_path: PDF  
            
        Returns:
              
        """
        self.logger.info(f"[FILE] PDF  : {pdf_path.name}")
        
        # 1: PyMuPDF 
        if PYMUPDF_AVAILABLE:
            text_content, metadata = self.extract_pdf_text_with_pymupdf(pdf_path)
            if text_content.strip():
                self.logger.info(f"PyMuPDF : {len(text_content)} chars extracted")
                return text_content
        
        # 2: pdfplumber 
        if PDFPLUMBER_AVAILABLE:
            text_content, metadata = self.extract_pdf_text_with_pdfplumber(pdf_path)
            if text_content.strip():
                self.logger.info(f"pdfplumber : {len(text_content)} chars extracted")
                return text_content
        
        # 3: PyPDF2 
        text_content, metadata = self.extract_pdf_text_with_pypdf2(pdf_path)
        if text_content.strip():
            self.logger.info(f"PyPDF2 : {len(text_content)} chars extracted")
            return text_content
        
        #   
        self.logger.error(f"[ERROR]  PDF   : {pdf_path.name}")
        return ""
    
    def smart_text_chunking(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[dict]:
        """
           -    .
        
        Args:
            text:  
            chunk_size:   ( )
            overlap:   
            
        Returns:
                
        """
        if not text.strip():
            return []
        
        self.logger.info(f"[NOTE]   : {len(text)} chars → {chunk_size} chars/chunk")
        
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = ""
        current_size = 0
        chunk_id = 1
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            #      
            if current_size + sentence_len <= chunk_size:
                current_chunk += sentence + " "
                current_size += sentence_len + 1
            else:
                #   
                if current_chunk.strip():
                    chunk_info = {
                        'id': chunk_id,
                        'content': current_chunk.strip(),
                        'size': len(current_chunk.strip()),
                        'sentences': len(sent_tokenize(current_chunk))
                    }
                    chunks.append(chunk_info)
                    self.logger.info(f"   #{chunk_id}: {chunk_info['size']} chars, {chunk_info['sentences']} sentences")
                    chunk_id += 1
                
                #    (overlap )
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sentence + " "
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence + " "
                    current_size = sentence_len + 1
        
        #   
        if current_chunk.strip():
            chunk_info = {
                'id': chunk_id,
                'content': current_chunk.strip(),
                'size': len(current_chunk.strip()),
                'sentences': len(sent_tokenize(current_chunk))
            }
            chunks.append(chunk_info)
            self.logger.info(f"   #{chunk_id}: {chunk_info['size']} chars, {chunk_info['sentences']} sentences")
        
        self.logger.info(f" : {len(chunks)}  ")
        return chunks
    
    def preprocess_text(self, text: str) -> List[str]:
        """
         ,  ,   
        
        Args:
            text:  
            
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
        TF-IDF  
        
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
              .
        
        Args:
            keywords: (keyword, score)  
            
        Returns:
                
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
        self.logger.info("      ...")
        
        all_extracted_text = []
        all_keywords = []
        paper_summaries = []
        
        # Process papers from ArXiv directory
        for source_dir in ['ArXiv']:
            source_path = self.papers_dir / source_dir
            
            if not source_path.exists():
                self.logger.warning(f"Source directory does not exist: {source_path}")
                continue
            
            self.logger.info(f"Processing papers from {source_dir}...")
            
            # Process PDF files
            pdf_files = list(source_path.glob("*.pdf"))
            self.logger.info(f"[DIR] {source_dir} {len(pdf_files)} PDF  ")
            
            skipped_count = 0
            processed_count = 0
            
            for pdf_file in pdf_files:
                # [SEARCH]   
                is_processed, reason = self.is_already_processed(pdf_file)
                if is_processed:
                    self.logger.info(f"[SKIP] : {pdf_file.name} - {reason}")
                    skipped_count += 1
                    continue
                
                self.logger.info(f"[FILE]   : {pdf_file.name}")
                
                text = self.extract_pdf_text(pdf_file)
                if text:
                    # [NOTE]   
                    chunks = self.smart_text_chunking(text, chunk_size=500, overlap=100)
                    processed_count += 1
                    
                    # 파일명에서 도메인 추출 (형식: 숫자_도메인_제목.pdf)
                    domain = 'General'
                    parts = pdf_file.stem.split('_', 2)
                    if len(parts) >= 2:
                        domain = parts[1]
                    
                    all_extracted_text.append({
                        'source': source_dir,
                        'filename': pdf_file.name,
                        'domain': domain,
                        'text': text,
                        'word_count': len(text.split()),
                        'chunks': chunks,
                        'chunk_count': len(chunks)
                    })
                    
                    # [SAVE]     
                    chunks_file = self.output_dir / f"chunks_{pdf_file.stem}.json"
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'source': source_dir,
                            'filename': pdf_file.name,
                            'domain': domain,
                            'total_chars': len(text),
                            'total_chunks': len(chunks),
                            'chunks': chunks
                        }, f, indent=2, ensure_ascii=False)
                    
                    #   
                    self.mark_as_processed(pdf_file, len(chunks))
                    self.logger.info(f"[SAVE]   : {chunks_file.name} ({len(chunks)} )")
                    
            #    
            self.logger.info(f"[PROGRESS] {source_dir}  : {processed_count} , {skipped_count} ")
        
        #    
        total_processed = len(all_extracted_text)
        total_files_tracked = len(self.processed_files)
        
        self.logger.info("=" * 60)
        self.logger.info("    ")
        self.logger.info("=" * 60)
        self.logger.info(f"[FILE]    : {total_processed}")
        self.logger.info(f"    : {total_files_tracked}")
        if total_files_tracked > total_processed:
            self.logger.info(f"[SKIP]   : {total_files_tracked - total_processed}")
        self.logger.info("=" * 60)
        
        #     (    )
        if all_extracted_text:
            self.save_chunk_summary(all_extracted_text)
    
    def save_chunk_summary(self, extracted_texts: List[Dict]):
        """    ."""
        self.logger.info("[STATS]     ...")
        
        total_papers = len(extracted_texts)
        total_chunks = sum(item.get('chunk_count', 0) for item in extracted_texts)
        total_chars = sum(len(item.get('text', '')) for item in extracted_texts)
        
        #  
        source_stats = {}
        domain_stats = {}
        chunk_size_stats = []
        
        for item in extracted_texts:
            source = item.get('source', 'unknown')
            domain = item.get('domain', 'General')
            
            if source not in source_stats:
                source_stats[source] = {'papers': 0, 'chunks': 0, 'chars': 0}
            
            source_stats[source]['papers'] += 1
            source_stats[source]['chunks'] += item.get('chunk_count', 0)
            source_stats[source]['chars'] += len(item.get('text', ''))
            
            # 도메인별 통계
            if domain not in domain_stats:
                domain_stats[domain] = {'papers': 0, 'chunks': 0}
            domain_stats[domain]['papers'] += 1
            domain_stats[domain]['chunks'] += item.get('chunk_count', 0)
            
            #   
            for chunk in item.get('chunks', []):
                chunk_size_stats.append(chunk['size'])
        
        #   //
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
            'domain_breakdown': domain_stats,
            'chunking_settings': {
                'chunk_size': 500,
                'overlap': 100,
                'method': 'sentence-aware'
            }
        }
        
        #   
        summary_file = self.output_dir / "chunk_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        #    
        self.logger.info("=" * 50)
        self.logger.info("[STATS]   ")
        self.logger.info("=" * 50)
        self.logger.info(f"[INFO]   : {total_papers}")
        self.logger.info(f"[FILE]   : {total_chunks}")
        self.logger.info(f"[STATS]   : {round(total_chunks / total_papers, 2) if total_papers > 0 else 0}")
        self.logger.info(f"   : {round(avg_chunk_size, 2)} chars")
        self.logger.info("")
        
        self.logger.info(" ")
        for source, stats in source_stats.items():
            self.logger.info(f" {source}: {stats['papers']} → {stats['chunks']}")
        
        self.logger.info("")
        self.logger.info(" ")
        for domain, stats in sorted(domain_stats.items(), key=lambda x: x[1]['papers'], reverse=True):
            self.logger.info(f" {domain}: {stats['papers']} → {stats['chunks']}")
        
        self.logger.info("=" * 50)
        self.logger.info(f"[SAVE]   : {summary_file.name}")
        self.logger.info("=" * 50)
    
    def save_extracted_text(self, extracted_texts: List[Dict]):
        """Save extracted text to file."""
        output_file = self.output_dir / "extracted_text.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"      \\n")
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
            plt.title('      ', fontsize=16)
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
     
===================================

Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Papers Processed: {total_papers}
Total Words Extracted: {total_words:,}
Average Words per Paper: {total_words // total_papers if total_papers > 0 else 0:,}

Papers by Source:
{chr(10).join([f'- {source}: {count} papers' for source, count in source_counts.items()])}

   :
{chr(10).join([f'- AGV, EMS, RTV, CNV     '][:10])}

 :
- extracted_text.txt:     
- keywords.csv:    TF-IDF 
- keyword_wordcloud.png:    

Processing Statistics:
- Successful extractions: {total_papers}
- Failed extractions: 0
- Average processing time per paper: ~2-3 seconds

 :
1.     
2. citation_analyzer.py   
3. AGV/EMS/RTV/CNV       
4. FAISS DB     
5.        
"""
        
        report_file = self.output_dir / "extraction_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"Extraction report saved to: {report_file}")


def main():
    """       ."""
    parser = argparse.ArgumentParser(description="      ")
    parser.add_argument("--papers-dir", default="../../1_data/0_crawled", help="      ")
    parser.add_argument("--output-dir", default="../../1_data/1_chunks", help="    ")
    parser.add_argument("--keywords-only", action="store_true", help=" ,   ")
    
    # Advanced RAG 옵션 (main.py 호환용)
    parser.add_argument("--enhanced-metadata", action="store_true", help="메타데이터 강화 (Advanced RAG)")
    parser.add_argument("--chunk-optimization", action="store_true", help="청크 최적화 (Advanced RAG)")
    parser.add_argument("--pre-retrieval-mode", action="store_true", help="Pre-Retrieval 모드 (Advanced RAG)")
    
    args = parser.parse_args()
    
    extractor = WarehouseAutomationTextExtractor(args.papers_dir, args.output_dir)
    
    if args.keywords_only:
        # Just process keywords from existing text files
        extractor.logger.info("     ...")
        # Implementation would go here for keyword-only processing
    else:
        extractor.process_all_papers()


if __name__ == "__main__":
    main()
