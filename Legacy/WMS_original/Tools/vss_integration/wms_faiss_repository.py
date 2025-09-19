#!/usr/bin/env python3
"""
WMS Faiss Repository
===================

VSS-AI-API-dev에서 사용할 WMS Faiss 벡터 데이터베이스 접근 클래스
"""

import faiss
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class WMSFaissRepository:
    """WMS Faiss 벡터 데이터베이스 접근 클래스"""
    
    def __init__(self, faiss_storage_path: str):
        """
        WMS Faiss Repository 초기화
        
        Args:
            faiss_storage_path: Faiss 저장소 경로 (faiss_storage 폴더)
        """
        self.storage_path = Path(faiss_storage_path)
        self.index = None
        self.documents = []
        self.metadatas = []
        self.embedding_model = None
        
        self._load_index()
        self._load_documents()
        self._setup_embedding_model()
        
        logger.info(f"✅ WMS Faiss Repository 초기화 완료: {len(self.documents)} 문서")
    
    def _load_index(self):
        """Faiss 인덱스 로드"""
        index_path = self.storage_path / "wms_knowledge.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Faiss 인덱스를 찾을 수 없습니다: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        logger.info(f"📊 Faiss 인덱스 로드: {self.index.ntotal} 벡터, {self.index.d} 차원")
    
    def _load_documents(self):
        """문서와 메타데이터 로드"""
        # 문서 로드
        documents_path = self.storage_path / "documents.json"
        with open(documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        # 메타데이터 로드
        metadata_path = self.storage_path / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadatas = json.load(f)
        
        logger.info(f"📚 문서 로드: {len(self.documents)} 개")
    
    def _setup_embedding_model(self):
        """임베딩 모델 초기화"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"🤖 임베딩 모델 초기화 완료 (디바이스: {device})")
            
        except Exception as e:
            logger.error(f"❌ 임베딩 모델 초기화 실패: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        벡터 검색 수행
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 쿼리 임베딩
            query_embedding = self.embedding_model.embed_query(query)
            query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            # Faiss 검색
            scores, indices = self.index.search(query_vector, top_k)
            
            # 결과 구성
            results = []
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.documents):
                    similarity = float(1 / (1 + score))  # 거리를 유사도로 변환
                    
                    results.append({
                        'rank': i + 1,
                        'document': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(score),
                        'similarity': similarity
                    })
            
            logger.info(f"🔍 검색 완료: '{query}' -> {len(results)} 결과")
            return results
            
        except Exception as e:
            logger.error(f"❌ 검색 실패: {e}")
            return []
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """문서 ID로 특정 문서 조회"""
        for i, metadata in enumerate(self.metadatas):
            if metadata.get('id') == doc_id:
                return {
                    'document': self.documents[i],
                    'metadata': metadata
                }
        return None
    
    def get_papers_by_source(self, source: str) -> List[str]:
        """소스별 논문 목록 조회"""
        papers = set()
        for metadata in self.metadatas:
            if metadata.get('paper_source', '').lower() == source.lower():
                papers.add(metadata.get('paper_filename', ''))
        return list(papers)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보"""
        # 논문별 통계
        papers = {}
        sources = {}
        
        for metadata in self.metadatas:
            # 논문별 청크 수
            filename = metadata.get('paper_filename', 'unknown')
            if filename not in papers:
                papers[filename] = 0
            papers[filename] += 1
            
            # 소스별 논문 수
            source = metadata.get('paper_source', 'unknown')
            if source not in sources:
                sources[source] = set()
            sources[source].add(filename)
        
        # 소스별 논문 수 계산
        source_stats = {source: len(papers) for source, papers in sources.items()}
        
        return {
            'total_documents': len(self.documents),
            'total_papers': len(papers),
            'vector_dimension': self.index.d,
            'embedding_model': 'jhgan/ko-sroberta-multitask',
            'papers_by_source': source_stats,
            'top_papers': sorted(papers.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def search_by_paper(self, paper_filename: str) -> List[Dict[str, Any]]:
        """특정 논문의 모든 청크 조회"""
        results = []
        for i, metadata in enumerate(self.metadatas):
            if metadata.get('paper_filename') == paper_filename:
                results.append({
                    'chunk_id': metadata.get('chunk_id'),
                    'document': self.documents[i],
                    'metadata': metadata
                })
        
        # 청크 ID 순으로 정렬
        results.sort(key=lambda x: x['chunk_id'])
        return results
    
    def get_similar_papers(self, paper_filename: str, top_k: int = 5) -> List[str]:
        """특정 논문과 유사한 논문들 찾기"""
        # 해당 논문의 첫 번째 청크로 검색
        paper_chunks = self.search_by_paper(paper_filename)
        if not paper_chunks:
            return []
        
        # 첫 번째 청크 내용으로 검색
        first_chunk = paper_chunks[0]['document']
        search_results = self.search(first_chunk, top_k * 2)  # 여유있게 검색
        
        # 다른 논문들만 추출
        similar_papers = []
        for result in search_results:
            result_paper = result['metadata']['paper_filename']
            if result_paper != paper_filename and result_paper not in similar_papers:
                similar_papers.append(result_paper)
                if len(similar_papers) >= top_k:
                    break
        
        return similar_papers


