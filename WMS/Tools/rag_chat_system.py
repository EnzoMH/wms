#!/usr/bin/env python3
"""
WMS RAG 채팅 시스템
==================

ChromaDB와 LLM을 연결한 질의응답 시스템입니다.
ChatOllama, OpenAI GPT, Claude 등 다양한 LLM 지원.

작성자: WMS 연구팀
날짜: 2024년 1월 15일
버전: 1.0.0
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import argparse

try:
    import chromadb
    from chromadb.config import Settings
    # LLM 라이브러리들
    try:
        from langchain_community.llms import Ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
        
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        
    try:
        import anthropic
        ANTHROPIC_AVAILABLE = True
    except ImportError:
        ANTHROPIC_AVAILABLE = False
        
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("권장 설치: pip install chromadb langchain-community openai anthropic")
    exit(1)


class WMSRAGChatSystem:
    """WMS 연구 논문 RAG 질의응답 시스템입니다."""
    
    def __init__(self, 
                 vector_db_dir: str = "../VectorDB",
                 llm_provider: str = "ollama",
                 model_name: str = "llama3.1"):
        """
        RAG 채팅 시스템을 초기화합니다.
        
        Args:
            vector_db_dir: ChromaDB가 저장된 디렉토리
            llm_provider: LLM 제공자 ('ollama', 'openai', 'claude')
            model_name: 사용할 모델명
        """
        self.vector_db_dir = Path(vector_db_dir)
        self.llm_provider = llm_provider
        self.model_name = model_name
        
        self.setup_logging()
        self.setup_chromadb()
        self.setup_llm()
        
    def setup_logging(self):
        """로깅을 설정합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_chat.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_chromadb(self):
        """ChromaDB 클라이언트를 초기화합니다."""
        self.logger.info("ChromaDB 연결 중...")
        
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_dir / "chroma_storage")
            )
            self.collection = self.chroma_client.get_collection("wms_research_papers")
            
            # 데이터베이스 상태 확인
            count = self.collection.count()
            self.logger.info(f"✅ ChromaDB 연결 완료: {count}개 청크 로드됨")
            
            if count == 0:
                raise ValueError("ChromaDB에 데이터가 없습니다. chromadb_builder.py를 먼저 실행하세요.")
                
        except Exception as e:
            self.logger.error(f"❌ ChromaDB 연결 실패: {e}")
            raise
    
    def setup_llm(self):
        """LLM을 설정합니다."""
        self.logger.info(f"LLM 초기화 중: {self.llm_provider} - {self.model_name}")
        
        if self.llm_provider == "ollama":
            if not OLLAMA_AVAILABLE:
                raise ImportError("langchain-community가 설치되지 않았습니다.")
            
            try:
                self.llm = Ollama(model=self.model_name, base_url="http://localhost:11434")
                # 연결 테스트
                response = self.llm.invoke("Hello")
                self.logger.info(f"✅ ChatOllama 연결 완료: {self.model_name}")
            except Exception as e:
                self.logger.error(f"❌ ChatOllama 연결 실패: {e}")
                self.logger.info("💡 Ollama가 실행 중인지 확인하세요: ollama serve")
                raise
                
        elif self.llm_provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai가 설치되지 않았습니다.")
            
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            
            self.llm = openai.Client()
            self.logger.info("✅ OpenAI API 연결 완료")
            
        elif self.llm_provider == "claude":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("anthropic이 설치되지 않았습니다.")
            
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다.")
            
            self.llm = anthropic.Anthropic()
            self.logger.info("✅ Claude API 연결 완료")
            
        else:
            raise ValueError(f"지원하지 않는 LLM 제공자: {self.llm_provider}")
    
    def search_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """질문과 관련된 청크들을 검색합니다."""
        self.logger.info(f"🔍 관련 문서 검색 중: '{query[:50]}...' (상위 {top_k}개)")
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        chunks = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            chunks.append({
                'content': doc,
                'metadata': metadata,
                'similarity': 1 - distance,
                'rank': i + 1
            })
        
        self.logger.info(f"✅ {len(chunks)}개 관련 문서 발견")
        return chunks
    
    def generate_answer_ollama(self, query: str, context_chunks: List[Dict]) -> str:
        """ChatOllama를 사용하여 답변을 생성합니다."""
        # 컨텍스트 준비
        context = "\n\n".join([
            f"📄 {chunk['metadata']['paper_filename']} (유사도: {chunk['similarity']:.3f})\n{chunk['content']}"
            for chunk in context_chunks
        ])
        
        # 프롬프트 생성
        prompt = f"""
당신은 WMS(창고관리시스템) 전문 연구 어시스턴트입니다.
제공된 연구 논문 내용을 바탕으로 질문에 답변해주세요.

질문: {query}

참고 자료:
{context}

답변 지침:
1. 제공된 연구 자료만을 바탕으로 답변하세요
2. 구체적인 논문명과 페이지를 인용하세요  
3. 한국어로 명확하고 자세하게 설명하세요
4. 확실하지 않은 내용은 "자료에서 확인되지 않았습니다"라고 하세요

답변:
"""
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            self.logger.error(f"❌ Ollama 답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}"
    
    def generate_answer_openai(self, query: str, context_chunks: List[Dict]) -> str:
        """OpenAI GPT를 사용하여 답변을 생성합니다."""
        context = "\n\n".join([
            f"논문: {chunk['metadata']['paper_filename']}\n내용: {chunk['content']}"
            for chunk in context_chunks
        ])
        
        messages = [
            {"role": "system", "content": """
당신은 WMS(창고관리시스템) 전문 연구 어시스턴트입니다.
제공된 연구 논문 내용만을 바탕으로 정확하고 상세하게 답변하세요.
논문명을 명시하며 한국어로 답변하세요.
"""},
            {"role": "user", "content": f"질문: {query}\n\n참고자료:\n{context}"}
        ]
        
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"❌ OpenAI 답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}"
    
    def generate_answer_claude(self, query: str, context_chunks: List[Dict]) -> str:
        """Claude를 사용하여 답변을 생성합니다."""
        context = "\n\n".join([
            f"논문: {chunk['metadata']['paper_filename']}\n내용: {chunk['content']}"
            for chunk in context_chunks
        ])
        
        prompt = f"""
WMS(창고관리시스템) 연구 전문가로서 답변해주세요.

질문: {query}

참고 논문 자료:
{context}

위 자료를 바탕으로 한국어로 정확하고 상세하게 답변하세요.
논문명을 반드시 인용하세요.
"""
        
        try:
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"❌ Claude 답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {e}"
    
    def ask_question(self, query: str, top_k: int = 5) -> Dict:
        """질문에 대한 답변을 생성합니다."""
        self.logger.info(f"💬 질문 처리 중: {query}")
        
        # 1단계: 관련 문서 검색
        relevant_chunks = self.search_relevant_chunks(query, top_k)
        
        # 2단계: LLM으로 답변 생성
        if self.llm_provider == "ollama":
            answer = self.generate_answer_ollama(query, relevant_chunks)
        elif self.llm_provider == "openai":
            answer = self.generate_answer_openai(query, relevant_chunks)
        elif self.llm_provider == "claude":
            answer = self.generate_answer_claude(query, relevant_chunks)
        else:
            answer = "지원하지 않는 LLM 제공자입니다."
        
        # 결과 패키징
        result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'paper': chunk['metadata']['paper_filename'],
                    'chunk_id': chunk['metadata']['chunk_id'],
                    'similarity': chunk['similarity']
                }
                for chunk in relevant_chunks
            ],
            'timestamp': datetime.now().isoformat(),
            'llm_provider': self.llm_provider
        }
        
        self.logger.info("✅ 답변 생성 완료")
        return result
    
    def interactive_chat(self):
        """대화형 채팅 인터페이스를 시작합니다."""
        print("🤖 WMS 연구 어시스턴트에 오신 것을 환영합니다!")
        print(f"📊 {self.collection.count()}개 논문 청크 로드됨")
        print(f"🧠 LLM: {self.llm_provider} - {self.model_name}")
        print("\n💡 예시 질문:")
        print("- WMS에서 가장 효율적인 물체 인식 방법은?")
        print("- Amazon 로봇과 비교했을 때 우리가 집중해야 할 기술은?")
        print("- 창고 자동화에서 ROI가 가장 높은 기술은?")
        print("- 종료하려면 'quit' 또는 'exit' 입력\n")
        
        while True:
            try:
                query = input("🙋 질문: ").strip()
                
                if query.lower() in ['quit', 'exit', '종료', 'q']:
                    print("👋 WMS 연구 어시스턴트를 종료합니다.")
                    break
                
                if not query:
                    continue
                
                print("🔍 검색 중...")
                result = self.ask_question(query)
                
                print(f"\n🤖 답변:\n{result['answer']}")
                print(f"\n📚 참고 논문:")
                for source in result['sources']:
                    print(f"- {source['paper']} (유사도: {source['similarity']:.3f})")
                print("-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 WMS 연구 어시스턴트를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="WMS RAG 채팅 시스템")
    parser.add_argument("--vector-db", default="../VectorDB", 
                       help="ChromaDB가 저장된 디렉토리")
    parser.add_argument("--llm-provider", choices=['ollama', 'openai', 'claude'],
                       default='ollama', help="사용할 LLM 제공자")
    parser.add_argument("--model-name", default="llama3.1",
                       help="사용할 모델명")
    parser.add_argument("--query", help="단일 질문 (대화형 모드 대신)")
    
    args = parser.parse_args()
    
    # RAG 시스템 초기화
    rag_system = WMSRAGChatSystem(
        vector_db_dir=args.vector_db,
        llm_provider=args.llm_provider,
        model_name=args.model_name
    )
    
    if args.query:
        # 단일 질문 모드
        result = rag_system.ask_question(args.query)
        print(f"질문: {result['query']}")
        print(f"답변: {result['answer']}")
        print(f"참고 논문: {len(result['sources'])}개")
    else:
        # 대화형 모드
        rag_system.interactive_chat()


if __name__ == "__main__":
    main()
