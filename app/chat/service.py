#!/usr/bin/env python3
"""
채팅 서비스 - Langchain RAG Chain 통합
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChatService:
    """채팅 서비스 (Langchain RAG Chain 기반)"""
    
    def __init__(self, exaone_rag=None):
        """
        Args:
            exaone_rag: EXAONE RAG wrapper 인스턴스
        """
        self.exaone_rag = exaone_rag
        self.faiss_retriever = None
        self.rag_chain = None
        
        # Langchain RAG Chain 초기화
        self._init_rag_chain()
    
    def _init_rag_chain(self):
        """Langchain RAG Chain 초기화 (VectorStoreManager 사용)"""
        try:
            from app.vectorstore.manager import get_vector_store_manager
            
            # VectorStoreManager 사용
            manager = get_vector_store_manager()
            
            # 벡터스토어 정보 출력
            info = manager.get_info()
            logger.info(f"벡터스토어 설정:")
            logger.info(f"  인덱스 이름: {info['index_name']}")
            logger.info(f"  임베딩 모델: {info['embedding_model']}")
            logger.info(f"  인덱스 타입: {info['index_type']}")
            
            # Retriever 생성
            self.faiss_retriever = manager.get_retriever(k=5)
            
            logger.info("[OK] Langchain RAG Chain 초기화 완료")
            
        except Exception as e:
            logger.error(f"RAG Chain 초기화 실패: {e}")
            logger.info("더미 컨텍스트 모드로 계속 진행합니다")
            self.faiss_retriever = None
    
    def query_rag(self, query: str, context_count: int = 5, temperature: float = 0.1) -> Dict[str, Any]:
        """
        RAG 쿼리 실행 (Langchain RAG Chain 사용)
        
        Args:
            query: 사용자 질문
            context_count: 컨텍스트 수
            temperature: 생성 온도
            
        Returns:
            응답 딕셔너리
        """
        if self.exaone_rag is None:
            raise RuntimeError("RAG 시스템이 초기화되지 않았습니다")
        
        # 1. 문서 검색 (FAISS Retriever 사용)
        if self.faiss_retriever:
            try:
                # Langchain Retriever로 관련 문서 검색
                docs = self.faiss_retriever.get_relevant_documents(query)
                contexts = [doc.page_content for doc in docs[:context_count]]
                
                logger.info(f"FAISS에서 {len(contexts)}개 문서 검색 완료")
                
            except Exception as e:
                logger.error(f"FAISS 검색 실패: {e}")
                contexts = self._get_dummy_contexts()
        else:
            # 폴백: 더미 컨텍스트
            logger.warning("FAISS 검색기 없음, 더미 컨텍스트 사용")
            contexts = self._get_dummy_contexts()
        
        # 2. EXAONE으로 답변 생성
        result = self.exaone_rag.query_rag(
            query=query,
            contexts=contexts[:context_count],
            temperature=temperature,
            show_metrics=False
        )
        
        # 3. 소스 문서 정보 추가
        result['sources'] = contexts[:context_count]
        
        return result
    
    def _get_dummy_contexts(self) -> List[str]:
        """더미 컨텍스트 (FAISS 없을 때 폴백)"""
        return [
            "AGV는 정해진 경로를 따라 이동하는 자동화 운반차량입니다.",
            "AMR은 자율주행이 가능한 모바일 로봇으로, 환경을 인식하고 경로를 스스로 계획합니다.",
            "WMS는 창고 관리 시스템으로 재고, 입출고, 피킹 등을 통합 관리합니다."
        ]
    
    def get_chat_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        채팅 히스토리 조회
        
        Args:
            limit: 조회 개수
            
        Returns:
            히스토리 리스트
        """
        try:
            log_file = Path("_0_1_core/rag/performance_log.json")
            
            if not log_file.exists():
                return []
            
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            # 최근 N개만 반환 (역순)
            recent_logs = logs[-limit:][::-1]
            
            # 히스토리 형식으로 변환
            history = []
            for log in recent_logs:
                history.append({
                    "query": log.get("query", "질문 없음"),
                    "timestamp": log.get("timestamp", ""),
                    "inference_time_ms": log.get("inference_time_ms", 0),
                    "tokens_per_second": log.get("tokens_per_second", 0),
                    "success": True
                })
            
            return history
            
        except Exception as e:
            logger.error(f"히스토리 조회 오류: {e}")
            return []


class ConversationalChatService(ChatService):
    """대화형 채팅 서비스 (Memory 포함)"""
    
    def __init__(self, exaone_rag=None):
        super().__init__(exaone_rag)
        self.sessions = {}  # session_id -> memory
    
    def _get_or_create_memory(self, session_id: str):
        """세션별 메모리 가져오기 또는 생성"""
        try:
            from langchain.memory import ConversationBufferMemory
            
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            
            return self.sessions[session_id]
            
        except ImportError:
            logger.warning("langchain memory 모듈 없음")
            return None
    
    def query_with_memory(self, query: str, session_id: str = "default", 
                          context_count: int = 5, temperature: float = 0.1) -> Dict[str, Any]:
        """
        대화 기록을 포함한 RAG 쿼리
        
        Args:
            query: 질문
            session_id: 세션 ID
            context_count: 컨텍스트 수
            temperature: 생성 온도
            
        Returns:
            응답 딕셔너리
        """
        memory = self._get_or_create_memory(session_id)
        
        # 이전 대화 컨텍스트 추가
        chat_history = ""
        if memory:
            history = memory.load_memory_variables({})
            if history.get("chat_history"):
                chat_history = "\n".join([
                    f"Q: {msg.content}" if msg.type == "human" else f"A: {msg.content}"
                    for msg in history["chat_history"][-3:]  # 최근 3개 대화
                ])
        
        # RAG 쿼리 실행
        result = self.query_rag(query, context_count, temperature)
        
        # 메모리에 저장
        if memory:
            memory.save_context({"input": query}, {"output": result.get("response", "")})
        
        return result
