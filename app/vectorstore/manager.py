#!/usr/bin/env python3
"""
벡터스토어 관리자
=================

FAISS 벡터스토어 로드 및 관리
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class VectorStoreConfig:
    """벡터스토어 설정"""
    
    def __init__(self, config_path: str = "2_vecdb/faiss_storage/config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """config.json 로드"""
        if not self.config_path.exists():
            logger.warning(f"설정 파일 없음: {self.config_path}")
            return self._default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # index_name이 없으면 추가
            if "index_name" not in config:
                config["index_name"] = "warehouse_automation_knowledge"
                self._save_config(config)
            
            return config
            
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            return self._default_config()
    
    def _save_config(self, config: Dict[str, Any]):
        """config.json 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"설정 저장 완료: {self.config_path}")
        except Exception as e:
            logger.error(f"설정 저장 실패: {e}")
    
    def _default_config(self) -> Dict[str, Any]:
        """기본 설정"""
        return {
            "dimension": 768,
            "total_documents": 0,
            "embedding_model": "jhgan/ko-sroberta-multitask",
            "index_type": "HNSW",
            "index_name": "warehouse_automation_knowledge",
            "created_at": ""
        }
    
    @property
    def index_name(self) -> str:
        """인덱스 이름"""
        return self.config.get("index_name", "warehouse_automation_knowledge")
    
    @property
    def embedding_model(self) -> str:
        """임베딩 모델"""
        return self.config.get("embedding_model", "jhgan/ko-sroberta-multitask")
    
    @property
    def dimension(self) -> int:
        """임베딩 차원"""
        return self.config.get("dimension", 768)
    
    @property
    def index_type(self) -> str:
        """인덱스 타입"""
        return self.config.get("index_type", "HNSW")


class VectorStoreManager:
    """벡터스토어 관리자"""
    
    def __init__(self, base_path: str = "2_vecdb/faiss_storage"):
        self.base_path = Path(base_path)
        self.config = VectorStoreConfig(str(self.base_path / "config.json"))
        self.vectorstore = None
        self.retriever = None
    
    def load_vectorstore(self):
        """FAISS 벡터스토어 로드 (순수 FAISS 형식)"""
        try:
            import faiss
            from langchain_community.vectorstores import FAISS
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain.docstore.in_memory import InMemoryDocstore
            from langchain.schema import Document
            
            if not self.base_path.exists():
                raise FileNotFoundError(f"벡터스토어 경로 없음: {self.base_path}")
            
            # 파일 경로
            index_path = self.base_path / f"{self.config.index_name}.index"
            documents_path = self.base_path / "documents.json"
            
            if not index_path.exists() or not documents_path.exists():
                raise FileNotFoundError(
                    f"필수 파일 없음: {index_path} 또는 {documents_path}"
                )
            
            # 임베딩 모델 로드
            logger.info(f"임베딩 모델 로드: {self.config.embedding_model}")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            
            # FAISS 인덱스 로드
            logger.info(f"FAISS 인덱스 로드: {index_path}")
            index = faiss.read_index(str(index_path))
            
            # 문서 로드
            logger.info(f"문서 로드: {documents_path}")
            with open(documents_path, 'r', encoding='utf-8') as f:
                documents_data = json.load(f)
            
            # Langchain Document 형식으로 변환
            documents = [Document(page_content=doc) for doc in documents_data]
            
            # InMemoryDocstore 생성
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
            index_to_docstore_id = {i: str(i) for i in range(len(documents))}
            
            # Langchain FAISS 객체 생성
            self.vectorstore = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            
            logger.info(f"[OK] 벡터스토어 로드 완료")
            logger.info(f"     인덱스 이름: {self.config.index_name}")
            logger.info(f"     문서 수: {len(documents)}")
            logger.info(f"     임베딩 모델: {self.config.embedding_model}")
            logger.info(f"     인덱스 타입: {self.config.index_type}")
            logger.info(f"     차원: {self.config.dimension}")
            
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"벡터스토어 로드 실패: {e}")
            return None
    
    def get_retriever(self, search_type: str = "similarity", k: int = 5):
        """Retriever 생성"""
        if self.vectorstore is None:
            self.load_vectorstore()
        
        if self.vectorstore is None:
            raise RuntimeError("벡터스토어가 로드되지 않았습니다")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )
        
        logger.info(f"Retriever 생성 완료 (k={k})")
        return self.retriever
    
    def search(self, query: str, k: int = 5):
        """문서 검색"""
        if self.retriever is None:
            self.get_retriever(k=k)
        
        docs = self.retriever.get_relevant_documents(query)
        return [doc.page_content for doc in docs]
    
    def get_info(self) -> Dict[str, Any]:
        """벡터스토어 정보"""
        return {
            "index_name": self.config.index_name,
            "embedding_model": self.config.embedding_model,
            "index_type": self.config.index_type,
            "dimension": self.config.dimension,
            "path": str(self.base_path),
            "loaded": self.vectorstore is not None
        }


# 전역 인스턴스 (싱글톤)
_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """벡터스토어 관리자 가져오기 (싱글톤)"""
    global _vector_store_manager
    
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    
    return _vector_store_manager


if __name__ == "__main__":
    # 테스트
    manager = VectorStoreManager()
    
    print("=== 벡터스토어 정보 ===")
    info = manager.get_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n=== 벡터스토어 로드 ===")
    vectorstore = manager.load_vectorstore()
    
    if vectorstore:
        print("\n=== 검색 테스트 ===")
        results = manager.search("AGV와 AMR의 차이", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc[:200]}...")

