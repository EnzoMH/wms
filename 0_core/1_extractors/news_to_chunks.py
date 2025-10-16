#!/usr/bin/env python3
"""
뉴스 기사를 벡터DB용 청크로 변환
================================

1_data/0_news/*.txt 파일을 읽어서
1_data/1_chunks/News/ 디렉토리에 청크 저장
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

class NewsChunker:
    """뉴스 기사를 청크로 변환"""
    
    def __init__(self, news_dir: str = "../../1_data/0_news", output_dir: str = "../../1_data/1_chunks"):
        self.news_dir = Path(news_dir)
        self.output_dir = Path(output_dir) / "news"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.output_dir / "news_chunking.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8', errors='ignore'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def parse_txt_article(self, content: str) -> List[Dict]:
        """
        뉴스 TXT 파일을 파싱하여 개별 기사로 분리
        
        형식:
        ================================================================================
        제목: ...
        출처: ...
        링크: ...
        날짜: ...
        
        내용:
        ...
        ================================================================================
        """
        articles = []
        
        # '=' 구분자로 기사 분리
        raw_articles = content.split('=' * 80)
        
        for raw in raw_articles:
            raw = raw.strip()
            if not raw:
                continue
            
            article = {}
            
            # 제목 추출
            title_match = re.search(r'제목:\s*(.+)', raw)
            if title_match:
                article['title'] = title_match.group(1).strip()
            
            # 출처 추출
            source_match = re.search(r'출처:\s*(.+)', raw)
            if source_match:
                article['source'] = source_match.group(1).strip()
            
            # 링크 추출
            link_match = re.search(r'링크:\s*(.+)', raw)
            if link_match:
                article['link'] = link_match.group(1).strip()
            
            # 날짜 추출
            date_match = re.search(r'날짜:\s*(.+)', raw)
            if date_match:
                article['published'] = date_match.group(1).strip()
            
            # 내용 추출
            content_match = re.search(r'내용:\s*(.+)', raw, re.DOTALL)
            if content_match:
                article['content'] = content_match.group(1).strip()
            
            if article.get('title') and article.get('content'):
                articles.append(article)
        
        return articles
    
    def clean_html(self, text: str) -> str:
        """HTML 태그 제거 및 텍스트 정리"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # &nbsp; 등 HTML 엔티티 제거
        text = re.sub(r'&[a-z]+;', ' ', text)
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def create_chunks(self, article: Dict) -> List[Dict]:
        """기사를 청크로 분할 (최대 1000자)"""
        content = self.clean_html(article['content'])
        
        # 짧은 기사는 하나의 청크로
        if len(content) <= 1000:
            return [{
                'content': content,
                'metadata': {
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published', ''),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'document_type': 'news_article'
                }
            }]
        
        # 긴 기사는 문단 단위로 분할
        paragraphs = [p.strip() for p in content.split('. ') if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= 1000:
                current_chunk += para + ". "
            else:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            'title': article.get('title', ''),
                            'source': article.get('source', ''),
                            'link': article.get('link', ''),
                            'published': article.get('published', ''),
                            'chunk_index': chunk_index,
                            'document_type': 'news_article'
                        }
                    })
                    chunk_index += 1
                current_chunk = para + ". "
        
        # 마지막 청크
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published', ''),
                    'chunk_index': chunk_index,
                    'document_type': 'news_article'
                }
            })
        
        # total_chunks 업데이트
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = len(chunks)
        
        return chunks
    
    def process_news_file(self, txt_file: Path):
        """TXT 뉴스 파일을 처리하여 청크 생성"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"처리 중: {txt_file.name}")
        self.logger.info(f"{'='*60}")
        
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            articles = self.parse_txt_article(content)
            self.logger.info(f"   {len(articles)}개 기사 파싱 완료")
            
            all_chunks = []
            for article in articles:
                chunks = self.create_chunks(article)
                all_chunks.extend(chunks)
                self.logger.info(f" [ V ] '{article['title'][:50]}...' -> {len(chunks)}개 청크")
            
            # 청크 저장
            output_file = self.output_dir / f"{txt_file.stem}_chunks.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"   저장: {output_file}")
            self.logger.info(f"   총 {len(all_chunks)}개 청크 생성")
            
            return len(all_chunks)
            
        except Exception as e:
            self.logger.error(f"   오류: {e}")
            return 0
    
    def run(self):
        """모든 .txt 뉴스 파일 처리"""
        self.logger.info("\n" + "="*60)
        self.logger.info("뉴스 기사 청킹 시작")
        self.logger.info("="*60)
        
        txt_files = list(self.news_dir.glob("*.txt"))
        
        if not txt_files:
            self.logger.warning(f" [ ! ] {self.news_dir}에 .txt 파일이 없습니다")
            return 0
        
        self.logger.info(f"발견된 뉴스 파일: {len(txt_files)}개\n")
        
        total_chunks = 0
        for txt_file in txt_files:
            chunks = self.process_news_file(txt_file)
            total_chunks += chunks
        
        # 요약
        self.logger.info(f"\n{'='*60}")
        self.logger.info("청킹 완료")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"총 생성 청크: {total_chunks}개")
        self.logger.info(f"저장 위치: {self.output_dir}")
        self.logger.info(f"\n다음 단계:")
        self.logger.info(f"   python 0_core/vectorDB/faiss_builder.py --action update")
        
        return total_chunks


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="뉴스 기사를 청크로 변환")
    parser.add_argument(
        "--news-dir",
        type=str,
        default="../../1_data/0_news",
        help="뉴스 TXT 파일 디렉토리"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../1_data/1_chunks",
        help="청크 저장 디렉토리"
    )
    
    args = parser.parse_args()
    
    chunker = NewsChunker(news_dir=args.news_dir, output_dir=args.output_dir)
    total = chunker.run()
    
    print(f"\n [ V ] 완료! 총 {total}개 청크 생성")

