#!/usr/bin/env python3
"""
WMS ChatOllama RAG 시스템
========================

ChromaDB + ChatOllama를 사용한 간단한 RAG 시스템
EXAONE, gpt-oss 등 로컬 모델 지원

작성자: WMS 연구팀  
날짜: 2024년 1월 15일
"""

import os
import sys
from pathlib import Path

# ChromaDB와 LangChain 임포트
try:
    import chromadb
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    print("✅ 필요한 라이브러리 로드 완료")
except ImportError as e:
    print(f"❌ 라이브러리 임포트 오류: {e}")
    print("설치: pip install chromadb langchain-community")
    exit(1)


class WMSChatOllamaRAG:
    """ChromaDB + ChatOllama RAG 시스템"""
    
    def __init__(self, model_name="hf.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:BF16"):
        """
        RAG 시스템 초기화
        
        Args:
            model_name: 사용할 Ollama 모델명
        """
        self.model_name = model_name
        print(f"🤖 모델 초기화 중: {model_name}")
        
        # ChromaDB 연결
        self.setup_chromadb()
        
        # ChatOllama LLM 설정
        self.setup_llm()
        
        # 프롬프트 템플릿 설정
        self.setup_prompt()
        
    def setup_chromadb(self):
        """ChromaDB 연결"""
        print("📊 ChromaDB 연결 중...")
        
        # ChromaDB 경로 (WMS/VectorDB에서 찾기)
        vector_db_path = Path("../WMS/VectorDB/chroma_storage")
        
        if not vector_db_path.exists():
            print("❌ ChromaDB를 찾을 수 없습니다!")
            print("먼저 WMS/Tools/chromadb_builder.py를 실행하세요.")
            exit(1)
        
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
            self.collection = self.chroma_client.get_collection("wms_research_papers")
            
            count = self.collection.count()
            print(f"✅ ChromaDB 연결 완료: {count:,}개 청크 로드됨")
            
        except Exception as e:
            print(f"❌ ChromaDB 연결 실패: {e}")
            exit(1)
    
    def setup_llm(self):
        """ChatOllama LLM 설정"""
        print("🧠 ChatOllama 설정 중...")
        
        try:
            self.llm = Ollama(
                model=self.model_name,
                base_url="http://localhost:11434",
                temperature=0.1,  # 정확한 답변을 위해 낮게 설정
                num_predict=2048   # 긴 답변 허용
            )
            
            # 연결 테스트
            test_response = self.llm.invoke("안녕하세요")
            print(f"✅ ChatOllama 연결 성공!")
            
        except Exception as e:
            print(f"❌ ChatOllama 연결 실패: {e}")
            print("💡 Ollama 서비스가 실행 중인지 확인하세요: ollama serve")
            exit(1)
    
    def setup_prompt(self):
        """프롬프트 템플릿 설정"""
        template = """
당신은 WMS(창고관리시스템) 전문 연구 어시스턴트입니다.
아래 제공된 연구 논문 내용을 바탕으로 질문에 정확하고 상세하게 답변해주세요.

질문: {question}

관련 연구 자료:
{context}

답변 가이드라인:
1. 제공된 연구 자료만을 기반으로 답변하세요
2. 구체적인 논문명을 인용하세요
3. 기술적 세부사항을 포함하여 전문적으로 답변하세요  
4. 한국어로 명확하게 설명하세요
5. 확신이 없는 내용은 "제공된 자료에서 확인되지 않습니다"라고 하세요

답변:
"""
        
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=template
        )
        
        # LangChain 체인 생성
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def search_relevant_docs(self, question, top_k=5):
        """질문과 관련된 문서 검색"""
        print(f"🔍 관련 문서 검색 중... (상위 {top_k}개)")
        
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=top_k
            )
            
            # 검색 결과 정리
            docs = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                docs.append({
                    'content': doc,
                    'paper': metadata['paper_filename'],
                    'chunk_id': metadata['chunk_id'],
                    'similarity': 1 - distance
                })
            
            print(f"✅ {len(docs)}개 관련 문서 발견")
            return docs
            
        except Exception as e:
            print(f"❌ 문서 검색 실패: {e}")
            return []
    
    def ask(self, question):
        """질문하기"""
        print(f"\n💬 질문: {question}")
        print("=" * 80)
        
        # 1단계: 관련 문서 검색
        relevant_docs = self.search_relevant_docs(question, top_k=5)
        
        if not relevant_docs:
            return "❌ 관련 문서를 찾을 수 없습니다."
        
        # 2단계: 컨텍스트 생성
        context = "\n\n".join([
            f"📄 논문: {doc['paper']}\n"
            f"유사도: {doc['similarity']:.3f}\n"
            f"내용: {doc['content'][:500]}..."
            for doc in relevant_docs
        ])
        
        print("🤖 답변 생성 중...")
        
        # 3단계: LLM으로 답변 생성
        try:
            response = self.chain.run(question=question, context=context)
            
            print(f"💡 답변:\n{response}")
            
            print(f"\n📚 참고 논문:")
            for doc in relevant_docs:
                print(f"- {doc['paper']} (유사도: {doc['similarity']:.3f})")
            
            return response
            
        except Exception as e:
            error_msg = f"❌ 답변 생성 실패: {e}"
            print(error_msg)
            return error_msg
    
    def chat_interactive(self):
        """대화형 채팅"""
        print("\n🚀 WMS 연구 어시스턴트 시작!")
        print(f"📊 로드된 데이터: {self.collection.count():,}개 청크")
        print(f"🤖 사용 모델: {self.model_name}")
        print("\n💡 예시 질문:")
        print("- WMS에서 가장 효율적인 창고 자동화 방법은?")
        print("- Amazon 로봇 시스템과 비교한 우리의 경쟁 우위는?")
        print("- 2024년 WMS 기술 트렌드는?")
        print("- 창고 로봇의 ROI 분석 결과는?")
        print("\n종료: 'quit', 'exit', 'q' 입력")
        print("=" * 80)
        
        while True:
            try:
                question = input("\n🙋‍♂️ 질문: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', '종료', '나가기']:
                    print("👋 WMS 연구 어시스턴트를 종료합니다. 감사합니다!")
                    break
                
                if not question:
                    continue
                
                # 질문 처리
                self.ask(question)
                print("=" * 80)
                
            except KeyboardInterrupt:
                print("\n👋 WMS 연구 어시스턴트를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
    
    def quick_test(self):
        """빠른 테스트"""
        test_questions = [
            "WMS 시스템에서 가장 중요한 기술은 무엇인가요?",
            "창고 자동화의 최신 트렌드는?",
            "로봇을 활용한 물류 최적화 방법은?"
        ]
        
        print("🧪 빠른 테스트 시작...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- 테스트 {i}/3 ---")
            self.ask(question)
            
        print("\n✅ 테스트 완료!")


def main():
    """메인 함수"""
    
    # 사용 가능한 모델들
    available_models = [
        "hf.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:BF16",  # 추천
        "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:BF16",
        "gpt-oss:20b"
    ]
    
    print("🎯 사용 가능한 모델들:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    
    try:
        choice = input(f"\n모델 선택 (1-{len(available_models)}, 기본값: 1): ").strip()
        if not choice:
            choice = "1"
        
        model_index = int(choice) - 1
        selected_model = available_models[model_index]
        
    except (ValueError, IndexError):
        print("잘못된 선택입니다. 기본 모델을 사용합니다.")
        selected_model = available_models[0]
    
    # RAG 시스템 초기화
    rag = WMSChatOllamaRAG(model_name=selected_model)
    
    # 실행 모드 선택
    mode = input("\n실행 모드 선택 (1: 대화형, 2: 테스트, 기본값: 1): ").strip()
    
    if mode == "2":
        rag.quick_test()
    else:
        rag.chat_interactive()


if __name__ == "__main__":
    main()
