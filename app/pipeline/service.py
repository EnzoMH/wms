#!/usr/bin/env python3
"""
파이프라인 통계 서비스
"""

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PipelineService:
    """파이프라인 통계 서비스"""
    
    @staticmethod
    def get_scraper_stats() -> Dict[str, Any]:
        """크롤링 통계 수집"""
        try:
            data_dir = Path("1_data/0_crawled")
            
            # ArXiv 논문 수
            arxiv_dir = data_dir / "ArXiv"
            arxiv_count = len(list(arxiv_dir.glob("*.pdf"))) if arxiv_dir.exists() else 0
            
            # 뉴스 파일 수
            news_dir = Path("1_data/0_news")
            news_count = len(list(news_dir.glob("*.json"))) if news_dir.exists() else 0
            
            # 마지막 크롤링 날짜
            last_crawl_date = None
            if arxiv_dir.exists():
                files = list(arxiv_dir.glob("*.pdf"))
                if files:
                    last_file = max(files, key=lambda f: f.stat().st_mtime)
                    from datetime import datetime
                    last_crawl_date = datetime.fromtimestamp(last_file.stat().st_mtime).strftime("%Y-%m-%d")
            
            return {
                "total_papers": arxiv_count,
                "total_news": news_count,
                "total_documents": arxiv_count + news_count,
                "last_crawl_date": last_crawl_date,
                "arxiv_count": arxiv_count,
                "news_sources": news_count
            }
        except Exception as e:
            logger.error(f"크롤링 통계 수집 실패: {e}")
            return {
                "total_papers": 0,
                "total_news": 0,
                "total_documents": 0,
                "last_crawl_date": None,
                "arxiv_count": 0,
                "news_sources": 0
            }
    
    @staticmethod
    def get_chunk_stats() -> Dict[str, Any]:
        """청크 통계 수집"""
        try:
            chunks_dir = Path("1_data/1_chunks")
            optimized_dir = Path("1_data/1_chunks_optimized")
            filtered_dir = Path("1_data/1_chunks_filtered")
            
            total_chunks = len(list(chunks_dir.glob("*.json"))) if chunks_dir.exists() else 0
            optimized_chunks = len(list(optimized_dir.glob("*.json"))) if optimized_dir.exists() else 0
            filtered_chunks = len(list(filtered_dir.glob("*.json"))) if filtered_dir.exists() else 0
            
            # 평균 청크 크기 계산 (샘플링)
            avg_chunk_size = 0
            total_chars = 0
            
            if chunks_dir.exists():
                sample_files = list(chunks_dir.glob("*.json"))[:10]
                if sample_files:
                    total_sample_chars = 0
                    for file in sample_files:
                        try:
                            with open(file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for chunk in data:
                                        if 'text' in chunk:
                                            total_sample_chars += len(chunk['text'])
                        except:
                            continue
                    
                    avg_chunk_size = total_sample_chars / len(sample_files) if sample_files else 0
                    total_chars = int(avg_chunk_size * total_chunks)
            
            filter_rate = 0
            if total_chunks > 0:
                filter_rate = ((total_chunks - filtered_chunks) / total_chunks) * 100
            
            return {
                "total_chunks": total_chunks,
                "optimized_chunks": optimized_chunks,
                "filtered_chunks": filtered_chunks,
                "avg_chunk_size": round(avg_chunk_size, 2),
                "total_chars": total_chars,
                "filter_rate": round(filter_rate, 2)
            }
        except Exception as e:
            logger.error(f"청크 통계 수집 실패: {e}")
            return {
                "total_chunks": 0,
                "optimized_chunks": 0,
                "filtered_chunks": 0,
                "avg_chunk_size": 0,
                "total_chars": 0,
                "filter_rate": 0
            }
    
    @staticmethod
    def get_vectordb_stats() -> Dict[str, Any]:
        """벡터DB 통계 수집"""
        try:
            vectordb_dir = Path("2_vecdb/faiss_storage")
            
            # config.json 읽기
            config_file = vectordb_dir / "config.json"
            config = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            
            # documents.json에서 벡터 수 계산
            documents_file = vectordb_dir / "documents.json"
            total_vectors = 0
            if documents_file.exists():
                with open(documents_file, 'r', encoding='utf-8') as f:
                    documents = json.load(f)
                    total_vectors = len(documents) if isinstance(documents, list) else 0
            
            # 인덱스 파일 크기
            index_file = vectordb_dir / "warehouse_automation_knowledge.index"
            index_size_mb = 0
            last_update = None
            if index_file.exists():
                index_size_mb = index_file.stat().st_size / (1024 * 1024)
                from datetime import datetime
                last_update = datetime.fromtimestamp(index_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                "total_vectors": total_vectors,
                "index_type": config.get("index_type", "HNSW"),
                "embedding_dim": config.get("embedding_dim", 768),
                "embedding_model": config.get("embedding_model", "jhgan/ko-sroberta-multitask"),
                "index_size_mb": round(index_size_mb, 2),
                "last_update": last_update
            }
        except Exception as e:
            logger.error(f"벡터DB 통계 수집 실패: {e}")
            return {
                "total_vectors": 0,
                "index_type": "HNSW",
                "embedding_dim": 768,
                "embedding_model": "unknown",
                "index_size_mb": 0,
                "last_update": None
            }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """전체 파이프라인 상태"""
        from datetime import datetime
        return {
            "scrapers": self.get_scraper_stats(),
            "chunks": self.get_chunk_stats(),
            "vectordb": self.get_vectordb_stats(),
            "status": "healthy",
            "last_updated": datetime.now()
        }

