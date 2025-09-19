#!/usr/bin/env python3
"""
창고 자동화 시스템 Faiss 구축기
==============================

AGV, EMS, RTV, CNV 등 창고 자동화 시스템 연구 논문의 
JSON 청크 파일들을 Faiss 벡터 데이터베이스로 변환하고
고도화된 RAG 시스템을 위한 고성능 검색 기능을 제공합니다.

작성자: 창고 자동화 시스템 연구팀
날짜: 2024년 1월 15일
버전: 1.0.0
"""

import os
import json
import glob
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import argparse

# 벡터 데이터베이스와 임베딩 라이브러리
try:
    import faiss
    import pandas as pd
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # CUDA 사용 가능한지 확인
    CUDA_AVAILABLE = torch.cuda.is_available()
    
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("권장 설치: pip install faiss-cpu langchain-huggingface torch pandas")
    exit(1)


class WarehouseAutomationFaissBuilder:
    """창고 자동화 시스템 연구 논문의 JSON 청크 파일들을 Faiss 벡터 데이터베이스로 변환하는 메인 클래스입니다."""
    
    def __init__(self, 
                 processed_data_dir: str = "../ProcessedData", 
                 vector_db_dir: str = "../VectorDB",
                 embedding_model: str = "korean_specialized"):
        """
        Faiss 빌더를 초기화합니다.
        
        Args:
            processed_data_dir: 청크 JSON 파일들이 있는 디렉토리
            vector_db_dir: Faiss 데이터베이스를 저장할 디렉토리
            embedding_model: 사용할 임베딩 모델
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.embedding_model_name = embedding_model
        
        # Faiss 관련 속성
        self.index = None
        self.documents = []
        self.metadatas = []
        self.embeddings_cache = []
        self.dimension = 768  # ko-sroberta-multitask 임베딩 차원
        
        self.setup_logging()
        self.setup_directories()
        self.setup_embedding_model()
        
    def setup_logging(self):
        """로깅을 설정합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('faiss_builder.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """필요한 디렉토리를 생성합니다."""
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_storage_dir = self.vector_db_dir / "faiss_storage"
        self.faiss_storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"창고 자동화 Faiss DB 디렉토리 준비: {self.faiss_storage_dir}")
        
    def setup_embedding_model(self):
        """한국어 특화 임베딩 모델을 설정합니다."""
        self.logger.info(f"임베딩 모델 초기화 중: {self.embedding_model_name}")
        
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        self.logger.info(f"사용 디바이스: {device}")
        
        try:
            # 한국어 특화 임베딩 모델 초기화
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # 모델 차원 확인을 위한 테스트
            test_embedding = self.embedding_model.embed_query("테스트")
            self.dimension = len(test_embedding)
            
            self.logger.info(f"✅ 한국어 특화 임베딩 모델 로드 완료")
            self.logger.info(f"   모델: jhgan/ko-sroberta-multitask")
            self.logger.info(f"   디바이스: {device}")
            self.logger.info(f"   차원: {self.dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ 임베딩 모델 로드 실패: {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """텍스트를 벡터로 임베딩합니다."""
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """텍스트 리스트를 배치로 임베딩합니다."""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"배치 임베딩 생성 실패: {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def load_chunk_files(self) -> List[Dict]:
        """청크 JSON 파일들을 로드합니다."""
        self.logger.info("청크 파일들 로드 중...")
        
        chunk_files = list(self.processed_data_dir.glob("chunks_*.json"))
        self.logger.info(f"발견된 청크 파일 수: {len(chunk_files)}")
        
        all_chunks = []
        
        for chunk_file in chunk_files:
            self.logger.info(f"📂 로드 중: {chunk_file.name}")
            
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    paper_data = json.load(f)
                
                paper_info = {
                    'source': paper_data.get('source', 'unknown'),
                    'filename': paper_data.get('filename', chunk_file.name),
                    'total_chars': paper_data.get('total_chars', 0),
                    'total_chunks': paper_data.get('total_chunks', 0)
                }
                
                for chunk in paper_data.get('chunks', []):
                    chunk_with_paper_info = {
                        **chunk,
                        'paper_source': paper_info['source'],
                        'paper_filename': paper_info['filename'],
                        'paper_total_chunks': paper_info['total_chunks']
                    }
                    all_chunks.append(chunk_with_paper_info)
                
                self.logger.info(f"  ✅ {len(paper_data.get('chunks', []))} 청크 로드됨")
                
            except Exception as e:
                self.logger.error(f"❌ {chunk_file.name} 로드 실패: {e}")
                
        self.logger.info(f"🎉 총 {len(all_chunks)} 청크 로드 완료")
        return all_chunks
    
    def build_vector_database(self):
        """Faiss 벡터 데이터베이스를 구축합니다."""
        self.logger.info("🚀 창고 자동화 시스템 Faiss 벡터 데이터베이스 구축 시작...")
        
        # 기존 데이터 확인
        index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
        if index_path.exists():
            response = input("기존 Faiss 인덱스가 있습니다. 새로 구축하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                self.logger.info("기존 인덱스를 로드합니다...")
                return self.load_existing_index()
        
        # 청크 데이터 로드
        chunks = self.load_chunk_files()
        
        if not chunks:
            self.logger.error("❌ 로드할 청크가 없습니다!")
            return
        
        # Faiss 인덱스 초기화 (HNSW 인덱스 사용 - 높은 성능)
        self.logger.info(f"📊 Faiss HNSW 인덱스 초기화 (차원: {self.dimension})")
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32는 연결 수
        self.index.hnsw.efConstruction = 200  # 구축 시 품질
        
        # 데이터 준비
        self.documents = []
        self.metadatas = []
        
        # 배치 단위로 처리 (메모리 효율성)
        batch_size = 50  # Faiss는 작은 배치로 처리
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        self.logger.info(f"📊 배치 처리 시작: {total_batches} 배치, 배치당 {batch_size} 청크")
        
        all_embeddings = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            self.logger.info(f"🔄 배치 {batch_idx + 1}/{total_batches} 처리 중... ({len(batch_chunks)} 청크)")
            
            # 텍스트와 메타데이터 준비
            batch_texts = []
            batch_metadatas = []
            
            for chunk in batch_chunks:
                chunk_id = f"paper_{chunk['paper_filename']}_{chunk['id']:03d}"
                batch_texts.append(chunk['content'])
                
                metadata = {
                    'id': chunk_id,
                    'paper_filename': chunk['paper_filename'],
                    'paper_source': chunk['paper_source'],
                    'chunk_id': chunk['id'],
                    'chunk_size': chunk['size'],
                    'sentences': chunk['sentences'],
                    'paper_total_chunks': chunk['paper_total_chunks'],
                    'content': chunk['content']  # 검색 시 사용
                }
                batch_metadatas.append(metadata)
            
            # 배치 임베딩 생성
            try:
                batch_embeddings = self.get_embeddings_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
                
                # 메타데이터와 문서 저장
                self.documents.extend(batch_texts)
                self.metadatas.extend(batch_metadatas)
                
                self.logger.info(f"  ✅ 배치 {batch_idx + 1} 임베딩 생성 완료")
                
            except Exception as e:
                self.logger.error(f"❌ 배치 {batch_idx + 1} 처리 실패: {e}")
        
        # 모든 임베딩을 하나로 합치기
        if all_embeddings:
            embeddings_matrix = np.vstack(all_embeddings)
            self.logger.info(f"📊 임베딩 매트릭스 형태: {embeddings_matrix.shape}")
            
            # Faiss 인덱스에 추가
            self.index.add(embeddings_matrix)
            
            # 인덱스 저장
            self.save_index()
            
            # 최종 통계
            self.logger.info("=" * 60)
            self.logger.info("🎉 창고 자동화 시스템 Faiss 벡터 데이터베이스 구축 완료!")
            self.logger.info(f"📊 총 저장된 청크 수: {len(self.documents)}")
            self.logger.info(f"💾 저장 위치: {self.faiss_storage_dir}")
            self.logger.info("=" * 60)
            
            return len(self.documents)
        else:
            self.logger.error("❌ 임베딩 생성에 실패했습니다!")
            return 0
    
    def save_index(self):
        """Faiss 인덱스와 메타데이터를 저장합니다."""
        self.logger.info("💾 Faiss 인덱스 저장 중...")
        
        # Faiss 인덱스 저장
        index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
        faiss.write_index(self.index, str(index_path))
        
        # 문서 내용 저장
        documents_path = self.faiss_storage_dir / "documents.json"
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # 메타데이터 저장
        metadata_path = self.faiss_storage_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        
        # 설정 정보 저장
        config = {
            'dimension': self.dimension,
            'total_documents': len(self.documents),
            'embedding_model': 'jhgan/ko-sroberta-multitask',
            'index_type': 'HNSW',
            'created_at': datetime.now().isoformat()
        }
        
        config_path = self.faiss_storage_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ 인덱스 저장 완료: {index_path}")
        self.logger.info(f"✅ 문서 저장 완료: {documents_path}")
        self.logger.info(f"✅ 메타데이터 저장 완료: {metadata_path}")
        self.logger.info(f"✅ 설정 저장 완료: {config_path}")
    
    def load_existing_index(self):
        """기존 Faiss 인덱스를 로드합니다."""
        try:
            self.logger.info("📂 기존 Faiss 인덱스 로드 중...")
            
            # 인덱스 로드
            index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
            self.index = faiss.read_index(str(index_path))
            
            # 문서 로드
            documents_path = self.faiss_storage_dir / "documents.json"
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            # 메타데이터 로드
            metadata_path = self.faiss_storage_dir / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
            
            # 설정 로드
            config_path = self.faiss_storage_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.logger.info(f"📊 인덱스 정보: {config}")
            
            self.logger.info(f"✅ 기존 인덱스 로드 완료 ({len(self.documents)} 문서)")
            return len(self.documents)
            
        except Exception as e:
            self.logger.error(f"❌ 기존 인덱스 로드 실패: {e}")
            return 0
    
    def test_search(self, query: str = "AGV 경로 계획", top_k: int = 5):
        """벡터 검색 테스트를 수행합니다."""
        if self.index is None:
            self.logger.error("❌ 인덱스가 로드되지 않았습니다!")
            return
        
        self.logger.info(f"🔍 검색 테스트: '{query}'")
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)  # Faiss 형태로 변환
            
            # 검색 수행
            scores, indices = self.index.search(query_embedding, top_k)
            
            self.logger.info(f"📋 검색 결과 ({top_k}개):")
            
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.metadatas):
                    metadata = self.metadatas[idx]
                    document = self.documents[idx]
                    
                    # 유사도 점수 계산 (L2 거리를 유사도로 변환)
                    similarity = 1 / (1 + score)
                    
                    self.logger.info(f"\n{i+1}. 📄 {metadata['paper_filename']}")
                    self.logger.info(f"   🎯 유사도: {similarity:.3f}")
                    self.logger.info(f"   📝 청크 #{metadata['chunk_id']}")
                    self.logger.info(f"   📊 크기: {metadata['chunk_size']} chars, {metadata['sentences']} sentences")
                    self.logger.info(f"   📃 내용: {document[:200]}...")
            
            return {
                'query': query,
                'results': [
                    {
                        'document': self.documents[idx],
                        'metadata': self.metadatas[idx],
                        'score': float(score),
                        'similarity': float(1 / (1 + score))
                    }
                    for idx, score in zip(indices[0], scores[0])
                    if idx < len(self.metadatas)
                ]
            }
            
        except Exception as e:
            self.logger.error(f"❌ 검색 실패: {e}")
            return None
    
    def get_database_stats(self):
        """데이터베이스 통계를 출력합니다."""
        if not self.documents:
            if not self.load_existing_index():
                self.logger.error("❌ 로드할 데이터베이스가 없습니다!")
                return
        
        count = len(self.documents)
        self.logger.info("📊 창고 자동화 시스템 Faiss 데이터베이스 통계:")
        self.logger.info(f" 총 청크 수: {count}")
        self.logger.info(f" 벡터 차원: {self.dimension}")
        self.logger.info(f" 인덱스 타입: HNSW")
        
        if count > 0:
            # 논문별 통계
            papers = {}
            for metadata in self.metadatas:
                filename = metadata['paper_filename']
                if filename not in papers:
                    papers[filename] = 0
                papers[filename] += 1
            
            self.logger.info(f"   📚 논문 수: {len(papers)}")
            self.logger.info(f"   📊 논문당 평균 청크: {count / len(papers):.1f}")
            
            # 상위 5개 논문
            top_papers = sorted(papers.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info("   🏆 상위 논문들:")
            for filename, chunk_count in top_papers:
                display_name = filename[:50] + "..." if len(filename) > 50 else filename
                self.logger.info(f"      - {display_name}: {chunk_count} 청크")


def main():
    """창고 자동화 시스템 Faiss 구축기 메인 실행 함수"""
    parser = argparse.ArgumentParser(description="창고 자동화 시스템 Faiss 벡터 데이터베이스 구축기")
    parser.add_argument("--processed-data", default="../ProcessedData", 
                       help="창고 자동화 청크 JSON 파일들이 있는 디렉토리")
    parser.add_argument("--vector-db", default="../VectorDB", 
                       help="창고 자동화 시스템 Faiss DB를 저장할 디렉토리")
    parser.add_argument("--action", choices=['build', 'test', 'stats'],
                       default='build', help="수행할 작업")
    parser.add_argument("--test-query", default="AGV 경로 계획",
                       help="테스트용 창고 자동화 검색 쿼리")
    
    args = parser.parse_args()
    
    # 창고 자동화 시스템 Faiss 빌더 초기화
    builder = WarehouseAutomationFaissBuilder(
        processed_data_dir=args.processed_data,
        vector_db_dir=args.vector_db,
        embedding_model="korean_specialized"
    )
    
    if args.action == 'build':
        # 벡터 데이터베이스 구축
        builder.build_vector_database()
        
    elif args.action == 'test':
        # 기존 인덱스 로드 후 검색 테스트
        builder.load_existing_index()
        builder.test_search(query=args.test_query)
        
    elif args.action == 'stats':
        # 통계 조회
        builder.get_database_stats()


if __name__ == "__main__":
    main()
