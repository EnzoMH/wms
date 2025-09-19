#!/usr/bin/env python3
"""
WMS ChromaDB 구축기
=================

JSON 청크 파일들을 ChromaDB 벡터 데이터베이스로 변환하고
RAG 시스템을 위한 검색 기능을 제공합니다.

작성자: WMS 연구팀
날짜: 2024년 1월 15일
버전: 1.0.0
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import argparse

# 벡터 데이터베이스와 임베딩 라이브러리
try:
    import chromadb
    from chromadb.config import Settings
    import pandas as pd
    # 임베딩 모델 - 여러 옵션 제공
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("권장 설치: pip install chromadb sentence-transformers pandas openai")
    exit(1)


class WMSChromaDBBuilder:
    """JSON 청크 파일들을 ChromaDB로 변환하는 메인 클래스입니다."""
    
    def __init__(self, 
                 processed_data_dir: str = "../ProcessedData", 
                 vector_db_dir: str = "../VectorDB",
                 embedding_model: str = "sentence-transformers"):
        """
        ChromaDB 빌더를 초기화합니다.
        
        Args:
            processed_data_dir: 청크 JSON 파일들이 있는 디렉토리
            vector_db_dir: ChromaDB를 저장할 디렉토리
            embedding_model: 사용할 임베딩 모델 ('sentence-transformers' or 'openai')
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.embedding_model_type = embedding_model
        
        self.setup_logging()
        self.setup_directories()
        self.setup_embedding_model()
        self.setup_chromadb()
        
    def setup_logging(self):
        """로깅을 설정합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chromadb_builder.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """필요한 디렉토리를 생성합니다."""
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"벡터 DB 디렉토리 준비: {self.vector_db_dir}")
        
    def setup_embedding_model(self):
        """임베딩 모델을 설정합니다."""
        self.logger.info(f"임베딩 모델 초기화 중: {self.embedding_model_type}")
        
        if self.embedding_model_type == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers가 설치되지 않았습니다.")
            
            # 한국어도 지원하는 멀티링구얼 모델 사용
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 빠르고 좋은 성능
            # model_name = "sentence-transformers/all-mpnet-base-v2"  # 더 좋은 성능, 느림
            
            self.embedding_model = SentenceTransformer(model_name)
            self.logger.info(f"✅ SentenceTransformer 로드 완료: {model_name}")
            
        elif self.embedding_model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai가 설치되지 않았습니다.")
            
            # OpenAI API 키 확인
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
            self.embedding_model = openai.Client()
            self.logger.info("✅ OpenAI 임베딩 API 준비 완료")
        else:
            raise ValueError(f"지원하지 않는 임베딩 모델: {self.embedding_model_type}")
    
    def setup_chromadb(self):
        """ChromaDB 클라이언트를 초기화합니다."""
        self.logger.info("ChromaDB 초기화 중...")
        
        # Persistent 클라이언트 생성 (데이터 영구 저장)
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.vector_db_dir / "chroma_storage")
        )
        
        # 컬렉션 생성 또는 가져오기
        collection_name = "wms_research_papers"
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "WMS 연구 논문 청크 벡터 DB"}
            )
            self.logger.info(f"✅ 새 컬렉션 생성: {collection_name}")
        except Exception:
            self.collection = self.chroma_client.get_collection(collection_name)
            self.logger.info(f"✅ 기존 컬렉션 로드: {collection_name}")
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트를 벡터로 임베딩합니다."""
        if self.embedding_model_type == "sentence-transformers":
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        elif self.embedding_model_type == "openai":
            response = self.embedding_model.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
    
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
        """벡터 데이터베이스를 구축합니다."""
        self.logger.info("🚀 벡터 데이터베이스 구축 시작...")
        
        # 기존 컬렉션 데이터 확인
        existing_count = self.collection.count()
        if existing_count > 0:
            self.logger.info(f"⚠️ 기존 데이터 {existing_count}개 발견")
            response = input("기존 데이터를 삭제하고 새로 구축하시겠습니까? (y/N): ")
            if response.lower() == 'y':
                self.chroma_client.delete_collection("wms_research_papers")
                self.collection = self.chroma_client.create_collection(
                    name="wms_research_papers",
                    metadata={"description": "WMS 연구 논문 청크 벡터 DB"}
                )
                self.logger.info("🗑️ 기존 데이터 삭제 완료")
            else:
                self.logger.info("기존 데이터 유지, 새 데이터만 추가합니다")
        
        # 청크 데이터 로드
        chunks = self.load_chunk_files()
        
        if not chunks:
            self.logger.error("❌ 로드할 청크가 없습니다!")
            return
        
        # 배치 단위로 처리 (메모리 효율성)
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        self.logger.info(f"📊 배치 처리 시작: {total_batches} 배치, 배치당 {batch_size} 청크")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            self.logger.info(f"🔄 배치 {batch_idx + 1}/{total_batches} 처리 중... ({len(batch_chunks)} 청크)")
            
            # 임베딩 생성
            documents = []
            metadatas = []
            ids = []
            
            for chunk in batch_chunks:
                chunk_id = f"paper_{chunk['paper_filename']}_{chunk['id']:03d}"
                documents.append(chunk['content'])
                
                metadata = {
                    'paper_filename': chunk['paper_filename'],
                    'paper_source': chunk['paper_source'],
                    'chunk_id': chunk['id'],
                    'chunk_size': chunk['size'],
                    'sentences': chunk['sentences'],
                    'paper_total_chunks': chunk['paper_total_chunks']
                }
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            # ChromaDB에 추가 (자동으로 임베딩 생성됨)
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                self.logger.info(f"  ✅ 배치 {batch_idx + 1} 저장 완료")
            except Exception as e:
                self.logger.error(f"❌ 배치 {batch_idx + 1} 저장 실패: {e}")
        
        # 최종 통계
        final_count = self.collection.count()
        self.logger.info("=" * 60)
        self.logger.info("🎉 벡터 데이터베이스 구축 완료!")
        self.logger.info(f"📊 총 저장된 청크 수: {final_count}")
        self.logger.info(f"💾 저장 위치: {self.vector_db_dir}")
        self.logger.info("=" * 60)
        
        return final_count
    
    def test_search(self, query: str = "warehouse automation", top_k: int = 5):
        """벡터 검색 테스트를 수행합니다."""
        self.logger.info(f"🔍 검색 테스트: '{query}'")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        self.logger.info(f"📋 검색 결과 ({top_k}개):")
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            self.logger.info(f"\n{i+1}. 📄 {metadata['paper_filename']}")
            self.logger.info(f"   🎯 유사도: {1-distance:.3f}")
            self.logger.info(f"   📝 청크 #{metadata['chunk_id']}")
            self.logger.info(f"   📊 크기: {metadata['chunk_size']} chars, {metadata['sentences']} sentences")
            self.logger.info(f"   📃 내용: {doc[:200]}...")
        
        return results
    
    def get_database_stats(self):
        """데이터베이스 통계를 출력합니다."""
        count = self.collection.count()
        self.logger.info("📊 데이터베이스 통계:")
        self.logger.info(f"   💾 총 청크 수: {count}")
        
        if count > 0:
            # 샘플 데이터로 통계 확인
            sample = self.collection.get(limit=min(100, count), include=['metadatas'])
            
            # 논문별 통계
            papers = {}
            for metadata in sample['metadatas']:
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
                self.logger.info(f"      - {filename[:50]}...: {chunk_count} 청크")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="WMS ChromaDB 구축기")
    parser.add_argument("--processed-data", default="../ProcessedData", 
                       help="청크 JSON 파일들이 있는 디렉토리")
    parser.add_argument("--vector-db", default="../VectorDB", 
                       help="ChromaDB를 저장할 디렉토리")
    parser.add_argument("--embedding-model", choices=['sentence-transformers', 'openai'],
                       default='sentence-transformers', help="사용할 임베딩 모델")
    parser.add_argument("--action", choices=['build', 'test', 'stats'],
                       default='build', help="수행할 작업")
    parser.add_argument("--test-query", default="warehouse automation",
                       help="테스트용 검색 쿼리")
    
    args = parser.parse_args()
    
    # ChromaDB 빌더 초기화
    builder = WMSChromaDBBuilder(
        processed_data_dir=args.processed_data,
        vector_db_dir=args.vector_db,
        embedding_model=args.embedding_model
    )
    
    if args.action == 'build':
        # 벡터 데이터베이스 구축
        builder.build_vector_database()
        
    elif args.action == 'test':
        # 검색 테스트
        builder.test_search(query=args.test_query)
        
    elif args.action == 'stats':
        # 통계 조회
        builder.get_database_stats()


if __name__ == "__main__":
    main()
