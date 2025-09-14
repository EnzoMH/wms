#!/usr/bin/env python3
"""
Langchain Faiss 통합 코드
========================

마이그레이션된 WMS 전문지식 Faiss 인덱스를 
본 프로젝트의 Langchain Application에 통합하기 위한 코드

작성자: WMS 연구팀
날짜: 2024년 1월 15일
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Langchain 관련 임포트
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_community.llms import Ollama
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Langchain 라이브러리 설치 필요: {e}")
    print("설치: pip install langchain langchain-community sentence-transformers")
    exit(1)


class WMSFaissVectorStore:
    """WMS 전문지식 Faiss 벡터스토어 래퍼"""
    
    def __init__(self, 
                 faiss_index_path: str,
                 documents_path: str,
                 metadata_path: str,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        WMS Faiss 벡터스토어 초기화
        
        Args:
            faiss_index_path: Faiss 인덱스 파일 경로
            documents_path: 문서 JSON 파일 경로
            metadata_path: 메타데이터 JSON 파일 경로
            embedding_model_name: 임베딩 모델명
        """
        self.faiss_index_path = Path(faiss_index_path)
        self.documents_path = Path(documents_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model_name = embedding_model_name
        
        self.setup_logging()
        self.load_data()
        self.setup_embeddings()
        self.create_langchain_vectorstore()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """Faiss 인덱스와 관련 데이터 로드"""
        self.logger.info("WMS Faiss 데이터 로드 중...")
        
        # Faiss 인덱스 로드
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"Faiss 인덱스를 찾을 수 없습니다: {self.faiss_index_path}")
        
        self.index = faiss.read_index(str(self.faiss_index_path))
        self.logger.info(f"✅ Faiss 인덱스 로드: {self.index.ntotal}개 벡터")
        
        # 문서 텍스트 로드
        with open(self.documents_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        self.logger.info(f"✅ 문서 로드: {len(self.documents)}개")
        
        # 메타데이터 로드
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.logger.info(f"✅ 메타데이터 로드: {len(self.metadata)}개")
        
    def setup_embeddings(self):
        """임베딩 모델 설정"""
        self.logger.info(f"임베딩 모델 설정: {self.embedding_model_name}")
        
        # HuggingFace 임베딩 (Langchain 호환)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # SentenceTransformer (직접 사용)
        self.sentence_model = SentenceTransformer(self.embedding_model_name)
        
    def create_langchain_vectorstore(self):
        """Langchain FAISS 벡터스토어 생성"""
        self.logger.info("Langchain FAISS 벡터스토어 생성 중...")
        
        # Document 객체 생성
        docs = []
        for i, (doc_text, meta) in enumerate(zip(self.documents, self.metadata)):
            doc = Document(
                page_content=doc_text,
                metadata={
                    'paper_filename': meta.get('paper_filename', 'unknown'),
                    'paper_source': meta.get('paper_source', 'unknown'),
                    'chunk_id': meta.get('chunk_id', i),
                    'chunk_size': meta.get('chunk_size', len(doc_text)),
                    'sentences': meta.get('sentences', 0)
                }
            )
            docs.append(doc)
        
        # FAISS 벡터스토어 생성 (기존 인덱스 사용)
        self.vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=self.embeddings
        )
        
        # 기존 Faiss 인덱스로 교체
        self.vectorstore.index = self.index
        
        self.logger.info("✅ Langchain FAISS 벡터스토어 생성 완료")
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """유사도 검색"""
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """점수와 함께 유사도 검색"""
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: Dict = None):
        """Langchain Retriever 반환"""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)


class WMSRAGChain:
    """WMS 전문지식 기반 RAG 체인"""
    
    def __init__(self, 
                 vectorstore: WMSFaissVectorStore,
                 llm_model: str = "llama3.1",
                 llm_base_url: str = "http://localhost:11434"):
        """
        WMS RAG 체인 초기화
        
        Args:
            vectorstore: WMS Faiss 벡터스토어
            llm_model: LLM 모델명
            llm_base_url: LLM 서버 URL
        """
        self.vectorstore = vectorstore
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url
        
        self.setup_llm()
        self.setup_prompt()
        self.create_rag_chain()
        
    def setup_llm(self):
        """LLM 설정"""
        self.llm = Ollama(
            model=self.llm_model,
            base_url=self.llm_base_url,
            temperature=0.1
        )
        
    def setup_prompt(self):
        """WMS 전문 프롬프트 템플릿 설정"""
        template = """
당신은 WMS(창고관리시스템) 및 물류 전문가입니다.
아래 제공된 WMS 연구 논문 내용을 바탕으로 질문에 정확하고 전문적으로 답변해주세요.

참고 자료:
{context}

질문: {question}

답변 가이드라인:
1. 제공된 연구 자료만을 기반으로 답변하세요
2. 구체적인 논문명과 기술적 세부사항을 포함하세요
3. WMS/물류 전문 용어를 정확히 사용하세요
4. 실무 적용 가능한 구체적인 방안을 제시하세요
5. 확실하지 않은 내용은 "제공된 자료에서 확인되지 않습니다"라고 하세요

답변:
"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
    def create_rag_chain(self):
        """RAG 체인 생성"""
        retriever = self.vectorstore.get_retriever(
            search_kwargs={"k": 5}
        )
        
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
    def ask(self, question: str) -> Dict:
        """질문하기"""
        result = self.rag_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": [
                {
                    "content": doc.page_content[:300] + "...",
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }


def create_wms_rag_system(faiss_data_dir: str) -> WMSRAGChain:
    """WMS RAG 시스템 생성 (원스톱 함수)"""
    
    # 파일 경로 설정
    faiss_data_path = Path(faiss_data_dir)
    index_path = faiss_data_path / "wms_knowledge.index"
    documents_path = faiss_data_path / "documents.json"
    metadata_path = faiss_data_path / "metadata.json"
    
    # 벡터스토어 생성
    vectorstore = WMSFaissVectorStore(
        faiss_index_path=str(index_path),
        documents_path=str(documents_path),
        metadata_path=str(metadata_path)
    )
    
    # RAG 체인 생성
    rag_chain = WMSRAGChain(vectorstore)
    
    return rag_chain


# 사용 예시
def main():
    """사용 예시"""
    print("🚀 WMS Langchain Faiss 통합 테스트")
    
    # WMS RAG 시스템 생성
    try:
        rag_system = create_wms_rag_system("./faiss_output")
        
        # 테스트 질문들
        test_questions = [
            "WMS에서 가장 효율적인 창고 자동화 방법은 무엇인가요?",
            "AGV와 로봇을 활용한 물류 최적화 전략을 설명해주세요.",
            "창고 관리에서 IoT 기술의 ROI는 어떻게 측정하나요?",
            "Amazon의 창고 로봇 시스템과 비교했을 때 우리가 집중해야 할 기술은?"
        ]
        
        for question in test_questions:
            print(f"\n💬 질문: {question}")
            print("-" * 80)
            
            result = rag_system.ask(question)
            
            print(f"🤖 답변:\n{result['answer']}")
            print(f"\n📚 참고 논문:")
            for i, source in enumerate(result['source_documents'], 1):
                print(f"{i}. {source['metadata']['paper_filename']}")
                print(f"   청크 #{source['metadata']['chunk_id']}")
                print(f"   내용: {source['content']}")
            print("=" * 80)
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        print("먼저 chroma_to_faiss_migrator.py를 실행하여 Faiss 데이터를 생성하세요.")


if __name__ == "__main__":
    main()
