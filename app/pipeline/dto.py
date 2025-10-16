#!/usr/bin/env python3
"""
파이프라인 DTO
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ScraperStats(BaseModel):
    """크롤링 통계"""
    total_papers: int = Field(0, description="총 논문 수")
    total_news: int = Field(0, description="총 뉴스 수")
    total_documents: int = Field(0, description="총 문서 수")
    last_crawl_date: Optional[str] = Field(None, description="마지막 크롤링 날짜")
    arxiv_count: int = Field(0, description="ArXiv 논문 수")
    news_sources: int = Field(0, description="뉴스 소스 수")


class ChunkStats(BaseModel):
    """청크 통계"""
    total_chunks: int = Field(0, description="총 청크 수")
    optimized_chunks: int = Field(0, description="최적화된 청크 수")
    filtered_chunks: int = Field(0, description="필터링된 청크 수")
    avg_chunk_size: float = Field(0, description="평균 청크 크기 (문자)")
    total_chars: int = Field(0, description="총 문자 수")
    filter_rate: float = Field(0, description="필터링 비율 (%)")


class VectorDBStats(BaseModel):
    """벡터DB 통계"""
    total_vectors: int = Field(0, description="총 벡터 수")
    index_type: str = Field("HNSW", description="인덱스 타입")
    embedding_dim: int = Field(768, description="임베딩 차원")
    embedding_model: str = Field("unknown", description="임베딩 모델")
    index_size_mb: float = Field(0, description="인덱스 크기 (MB)")
    last_update: Optional[str] = Field(None, description="마지막 업데이트")


class PipelineStatus(BaseModel):
    """전체 파이프라인 현황"""
    scrapers: ScraperStats = Field(default_factory=ScraperStats)
    chunks: ChunkStats = Field(default_factory=ChunkStats)
    vectordb: VectorDBStats = Field(default_factory=VectorDBStats)
    status: str = Field("unknown", description="시스템 상태")
    last_updated: datetime = Field(default_factory=datetime.now)

