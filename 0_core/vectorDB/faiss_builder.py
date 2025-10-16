#!/usr/bin/env python3
"""
창고 자동화 시스템 Faiss 벡터DB 구축기
==============================

AGV, EMS, RTV, CNV 등의 연구논문 데이터를
JSON 청크에서 읽어 Faiss 벡터DB로 구축
고성능 RAG 시스템의 핵심 구성요소.

작성자: 신명호
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

# torch.load 보안 경고 우회
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

#    
try:
    import faiss
    import pandas as pd
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # CUDA   
    CUDA_AVAILABLE = torch.cuda.is_available()
    
except ImportError as e:
    print(f"  . : {e}")
    print(" : pip install faiss-cpu langchain-huggingface torch pandas")
    exit(1)


class WarehouseAutomationFaissBuilder:
    """     JSON   Faiss     ."""
    
    def __init__(self, 
                 processed_data_dir: str = "../../1_data/1_chunks_filtered", 
                 vector_db_dir: str = "../../2_vecdb",
                 embedding_model: str = "korean_specialized",
                 use_reranked: bool = True):
        """
        Faiss  .
        
        Args:
            processed_data_dir:  JSON    (  )
            vector_db_dir: Faiss   
            embedding_model:   
            use_reranked:    
        """
        self.use_reranked = use_reranked
        self.vector_db_dir = Path(vector_db_dir)
        self.embedding_model_name = embedding_model
        
        # Faiss  
        self.index = None
        self.documents = []
        self.metadatas = []
        self.embeddings_cache = []
        self.dimension = 768  # ko-sroberta-multitask  
        
        self.setup_logging()
        
        #    
        if use_reranked:
            reranked_dir = Path(processed_data_dir)
            if reranked_dir.exists() and list(reranked_dir.glob("chunks_*.json")):
                self.processed_data_dir = reranked_dir
                self.logger.info(f"[SUCCESS]   : {reranked_dir}")
            else:
                #     
                original_dir = Path("../../1_data/1_chunks")
                self.processed_data_dir = original_dir
                self.logger.warning(f"[WARN]   .   : {original_dir}")
        else:
            self.processed_data_dir = Path(processed_data_dir)
        
        self.setup_directories()
        self.setup_embedding_model()
        
    def setup_logging(self):
        """ ."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('faiss_builder.log', encoding='utf-8', errors='ignore'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """  ."""
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_storage_dir = self.vector_db_dir / "faiss_storage"
        self.faiss_storage_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  Faiss DB  : {self.faiss_storage_dir}")
        
    def setup_embedding_model(self):
        """    ."""
        self.logger.info(f"   : {self.embedding_model_name}")
        
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        self.logger.info(f" : {device}")
        
        try:
            # 한국어 임베딩 모델 로드
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sroberta-multitask",
                model_kwargs={
                    'device': device,
                    'trust_remote_code': True
                },
                encode_kwargs={'normalize_embeddings': True}
            )
            
            #     
            test_embedding = self.embedding_model.embed_query("")
            self.dimension = len(test_embedding)
            
            self.logger.info(f"[SUCCESS]      ")
            self.logger.info(f"   : jhgan/ko-sroberta-multitask")
            self.logger.info(f"   : {device}")
            self.logger.info(f"   : {self.dimension}")
            
        except Exception as e:
            self.logger.error(f"[ERROR]    : {e}")
            raise
    
    def get_embedding(self, text: str) -> np.ndarray:
        """  ."""
        try:
            embedding = self.embedding_model.embed_query(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"  : {e}")
            return np.zeros(self.dimension, dtype=np.float32)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """   ."""
        try:
            embeddings = self.embedding_model.embed_documents(texts)
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"   : {e}")
            return np.zeros((len(texts), self.dimension), dtype=np.float32)
    
    def load_chunk_files(self) -> List[Dict]:
        """ JSON  ."""
        self.logger.info("   ...")
        
        # 논문 청크 (기존)
        chunk_files = list(self.processed_data_dir.glob("chunks_*.json"))
        self.logger.info(f"   : {len(chunk_files)}")
        
        # 뉴스 청크 (신규)
        news_dir = self.processed_data_dir / "news"
        news_files = list(news_dir.glob("*_chunks.json")) if news_dir.exists() else []
        self.logger.info(f"   : {len(news_files)}")
        
        all_chunks = []
        
        # 논문 청크 처리
        for chunk_file in chunk_files:
            self.logger.info(f"  : {chunk_file.name}")
            
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
                
                self.logger.info(f"  [SUCCESS] {len(paper_data.get('chunks', []))}  ")
                
            except Exception as e:
                self.logger.error(f"[ERROR] {chunk_file.name}  : {e}")
        
        # 뉴스 청크 처리 (다른 구조)
        for news_file in news_files:
            self.logger.info(f"  : news/{news_file.name}")
            
            try:
                with open(news_file, 'r', encoding='utf-8') as f:
                    news_chunks = json.load(f)
                
                for chunk in news_chunks:
                    # 뉴스 청크를 논문 청크 형식으로 변환
                    chunk_with_info = {
                        'id': chunk['metadata'].get('chunk_index', 0),
                        'content': chunk['content'],
                        'paper_source': 'news',
                        'paper_filename': chunk['metadata'].get('title', news_file.name),
                        'paper_total_chunks': chunk['metadata'].get('total_chunks', 1),
                        'metadata': chunk['metadata']
                    }
                    all_chunks.append(chunk_with_info)
                
                self.logger.info(f"  [SUCCESS] {len(news_chunks)}  ")
                
            except Exception as e:
                self.logger.error(f"[ERROR] news/{news_file.name}  : {e}")
                
        self.logger.info(f"[DONE]  {len(all_chunks)}   ")
        return all_chunks
    
    def build_vector_database(self):
        """Faiss   ."""
        self.logger.info("[START]    Faiss    ...")
        
        #   
        index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
        if index_path.exists():
            response = input(" Faiss  .  ? (y/N): ")
            if response.lower() != 'y':
                self.logger.info("  ...")
                return self.load_existing_index()
        
        #   
        chunks = self.load_chunk_files()
        
        if not chunks:
            self.logger.error("[ERROR]   !")
            return
        
        # Faiss   (HNSW   -  )
        self.logger.info(f"[STATS] Faiss HNSW   (: {self.dimension})")
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32  
        self.index.hnsw.efConstruction = 200  #   
        
        #  
        self.documents = []
        self.metadatas = []
        
        #    ( )
        batch_size = 50  # Faiss   
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        self.logger.info(f"[STATS]   : {total_batches} ,  {batch_size} ")
        
        all_embeddings = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            self.logger.info(f"[PROCESS]  {batch_idx + 1}/{total_batches}  ... ({len(batch_chunks)} )")
            
            #   
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
                    'chunk_size': chunk.get('size', len(chunk['content'])),  # 뉴스 청크 대응
                    'sentences': chunk.get('sentences', 1),  # 뉴스 청크 대응
                    'paper_total_chunks': chunk['paper_total_chunks'],
                    'content': chunk['content'],  #   
                    # 뉴스 청크 추가 메타데이터
                    'document_type': chunk.get('metadata', {}).get('document_type', 'paper'),
                    'news_source': chunk.get('metadata', {}).get('source', ''),
                    'news_link': chunk.get('metadata', {}).get('link', ''),
                    'published': chunk.get('metadata', {}).get('published', '')
                }
                batch_metadatas.append(metadata)
            
            #   
            try:
                batch_embeddings = self.get_embeddings_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
                
                #   
                self.documents.extend(batch_texts)
                self.metadatas.extend(batch_metadatas)
                
                self.logger.info(f"  [SUCCESS]  {batch_idx + 1}   ")
                
            except Exception as e:
                self.logger.error(f"[ERROR]  {batch_idx + 1}  : {e}")
        
        #    
        if all_embeddings:
            embeddings_matrix = np.vstack(all_embeddings)
            self.logger.info(f"[STATS]   : {embeddings_matrix.shape}")
            
            # Faiss  
            self.index.add(embeddings_matrix)
            
            #  
            self.save_index()
            
            #  
            self.logger.info("=" * 60)
            self.logger.info("[DONE]    Faiss    !")
            self.logger.info(f"[STATS]    : {len(self.documents)}")
            self.logger.info(f"[SAVE]  : {self.faiss_storage_dir}")
            self.logger.info("=" * 60)
            
            return len(self.documents)
        else:
            self.logger.error("[ERROR]   !")
            return 0
    
    def save_index(self):
        """Faiss   ."""
        self.logger.info("[SAVE] Faiss   ...")
        
        # Faiss  
        index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
        faiss.write_index(self.index, str(index_path))
        
        #   
        documents_path = self.faiss_storage_dir / "documents.json"
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        #  
        metadata_path = self.faiss_storage_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        
        #   
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
        
        self.logger.info(f"[SUCCESS]   : {index_path}")
        self.logger.info(f"[SUCCESS]   : {documents_path}")
        self.logger.info(f"[SUCCESS]   : {metadata_path}")
        self.logger.info(f"[SUCCESS]   : {config_path}")
    
    def load_existing_index(self):
        """ Faiss  ."""
        try:
            self.logger.info("  Faiss   ...")
            
            #  
            index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
            self.index = faiss.read_index(str(index_path))
            
            #  
            documents_path = self.faiss_storage_dir / "documents.json"
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            #  
            metadata_path = self.faiss_storage_dir / "metadata.json"
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
            
            #  
            config_path = self.faiss_storage_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.logger.info(f"[STATS]  : {config}")
            
            self.logger.info(f"[SUCCESS]     ({len(self.documents)} )")
            return len(self.documents)
            
        except Exception as e:
            self.logger.error(f"[ERROR]    : {e}")
            return 0
    
    def update_vector_database(self):
        """기존 벡터DB에 새로운 청크만 추가 (증분 업데이트)"""
        self.logger.info("[START] 벡터DB 증분 업데이트...")
        
        # 1. 기존 인덱스 로드
        index_path = self.faiss_storage_dir / "warehouse_automation_knowledge.index"
        if not index_path.exists():
            self.logger.error("[ERROR] 기존 벡터DB가 없습니다. build를 먼저 실행하세요.")
            return 0
        
        self.load_existing_index()
        
        # 2. 기존 메타데이터에서 이미 추가된 파일 확인
        existing_ids = set(m['id'] for m in self.metadatas)
        self.logger.info(f"[STATS] 기존 청크: {len(existing_ids)}개")
        
        # 3. 모든 청크 로드
        all_chunks = self.load_chunk_files()
        
        # 4. 새로운 청크만 필터링
        new_chunks = []
        for chunk in all_chunks:
            chunk_id = f"paper_{chunk['paper_filename']}_{chunk['id']:03d}"
            if chunk_id not in existing_ids:
                new_chunks.append(chunk)
        
        self.logger.info(f"[STATS] 새로운 청크: {len(new_chunks)}개")
        
        if not new_chunks:
            self.logger.info("[INFO] 추가할 새로운 청크가 없습니다.")
            return len(self.documents)
        
        # 5. 새로운 청크 임베딩 및 추가
        batch_size = 50
        total_batches = (len(new_chunks) + batch_size - 1) // batch_size
        
        self.logger.info(f"[STATS] {total_batches}개 배치로 처리...")
        
        all_new_embeddings = []
        new_documents = []
        new_metadatas = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(new_chunks))
            batch_chunks = new_chunks[start_idx:end_idx]
            
            self.logger.info(f"[PROCESS] 배치 {batch_idx + 1}/{total_batches} 처리 중... ({len(batch_chunks)}개)")
            
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
                    'chunk_size': chunk.get('size', len(chunk['content'])),
                    'sentences': chunk.get('sentences', 1),
                    'paper_total_chunks': chunk['paper_total_chunks'],
                    'content': chunk['content'],
                    'document_type': chunk.get('metadata', {}).get('document_type', 'paper'),
                    'news_source': chunk.get('metadata', {}).get('source', ''),
                    'news_link': chunk.get('metadata', {}).get('link', ''),
                    'published': chunk.get('metadata', {}).get('published', '')
                }
                batch_metadatas.append(metadata)
            
            try:
                batch_embeddings = self.get_embeddings_batch(batch_texts)
                all_new_embeddings.append(batch_embeddings)
                
                new_documents.extend(batch_texts)
                new_metadatas.extend(batch_metadatas)
                
                self.logger.info(f"  [SUCCESS] 배치 {batch_idx + 1} 임베딩 완료")
                
            except Exception as e:
                self.logger.error(f"[ERROR] 배치 {batch_idx + 1} 처리 실패: {e}")
        
        # 6. Faiss 인덱스에 추가
        if all_new_embeddings:
            embeddings_matrix = np.vstack(all_new_embeddings)
            self.logger.info(f"[STATS] 새 임베딩 행렬: {embeddings_matrix.shape}")
            
            self.index.add(embeddings_matrix)
            
            # 문서 및 메타데이터 병합
            self.documents.extend(new_documents)
            self.metadatas.extend(new_metadatas)
            
            # 7. 저장
            self.save_index()
            
            self.logger.info("=" * 60)
            self.logger.info("[DONE] 벡터DB 증분 업데이트 완료!")
            self.logger.info(f"[STATS] 기존: {len(existing_ids)}개 → 현재: {len(self.documents)}개")
            self.logger.info(f"[STATS] 추가됨: {len(new_documents)}개")
            self.logger.info(f"[SAVE] 저장 위치: {self.faiss_storage_dir}")
            self.logger.info("=" * 60)
            
            return len(self.documents)
        else:
            self.logger.error("[ERROR] 임베딩 실패!")
            return 0
    
    def test_search(self, query: str = "AGV  ", top_k: int = 5):
        """   ."""
        if self.index is None:
            self.logger.error("[ERROR]   !")
            return
        
        self.logger.info(f"[SEARCH]  : '{query}'")
        
        try:
            #   
            query_embedding = self.get_embedding(query)
            query_embedding = query_embedding.reshape(1, -1)  # Faiss  
            
            #  
            scores, indices = self.index.search(query_embedding, top_k)
            
            self.logger.info(f"   ({top_k}):")
            
            for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
                if idx < len(self.metadatas):
                    metadata = self.metadatas[idx]
                    document = self.documents[idx]
                    
                    #    (L2   )
                    similarity = 1 / (1 + score)
                    
                    self.logger.info(f"\n{i+1}. [FILE] {metadata['paper_filename']}")
                    self.logger.info(f"    : {similarity:.3f}")
                    self.logger.info(f"   [NOTE]  #{metadata['chunk_id']}")
                    self.logger.info(f"   [STATS] : {metadata['chunk_size']} chars, {metadata['sentences']} sentences")
                    self.logger.info(f"    : {document[:200]}...")
            
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
            self.logger.error(f"[ERROR]  : {e}")
            return None
    
    def get_database_stats(self):
        """  ."""
        if not self.documents:
            if not self.load_existing_index():
                self.logger.error("[ERROR]   !")
                return
        
        count = len(self.documents)
        self.logger.info("[STATS]    Faiss  :")
        self.logger.info(f"   : {count}")
        self.logger.info(f"  : {self.dimension}")
        self.logger.info(f"  : HNSW")
        
        if count > 0:
            #  
            papers = {}
            for metadata in self.metadatas:
                filename = metadata['paper_filename']
                if filename not in papers:
                    papers[filename] = 0
                papers[filename] += 1
            
            self.logger.info(f"   [INFO]  : {len(papers)}")
            self.logger.info(f"   [STATS]   : {count / len(papers):.1f}")
            
            #  5 
            top_papers = sorted(papers.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info("   [TOP]  :")
            for filename, chunk_count in top_papers:
                display_name = filename[:50] + "..." if len(filename) > 50 else filename
                self.logger.info(f"      - {display_name}: {chunk_count} ")


def main():
    """   Faiss    """
    # 프로젝트 루트 찾기 (main.py가 있는 위치)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # 0_core/vectorDB/ -> 0_core/ -> vss/
    
    parser = argparse.ArgumentParser(description="   Faiss   ")
    parser.add_argument("--processed-data", 
                       default=str(project_root / "1_data/1_chunks_filtered"), 
                       help="   JSON    (  )")
    parser.add_argument("--vector-db", 
                       default=str(project_root / "2_vecdb"), 
                       help="   Faiss DB  ")
    parser.add_argument("--use-reranked", action="store_true", default=True,
                       help="   (: True)")
    parser.add_argument("--action", choices=['build', 'update', 'test', 'stats'],
                       default='build', help="build=새로 구축, update=증분 업데이트, test=검색 테스트, stats=통계")
    parser.add_argument("--test-query", default="AGV  ",
                       help="    ")
    
    # Advanced RAG / Modular RAG 옵션 (main.py 호환용)
    parser.add_argument("--modular-rag-mode", action="store_true", 
                       help="Modular RAG 모드 활성화 (Advanced RAG)")
    parser.add_argument("--multi-search-support", action="store_true",
                       help="다중 검색 모듈 지원 (Modular RAG)")
    
    args = parser.parse_args()
    
    #    Faiss  
    builder = WarehouseAutomationFaissBuilder(
        processed_data_dir=args.processed_data,
        vector_db_dir=args.vector_db,
        embedding_model="korean_specialized",
        use_reranked=args.use_reranked
    )
    
    if args.action == 'build':
        #   
        builder.build_vector_database()
    
    elif args.action == 'update':
        # 증분 업데이트 (새로운 청크만 추가)
        builder.update_vector_database()
        
    elif args.action == 'test':
        #      
        builder.load_existing_index()
        builder.test_search(query=args.test_query)
        
    elif args.action == 'stats':
        #  
        builder.get_database_stats()


if __name__ == "__main__":
    main()
