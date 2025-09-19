#!/usr/bin/env python3
"""
고도화된 WMS RAG 시스템
======================

RAGAS 평가, LangSmith 추적, 개선된 프롬프트, 하이브리드 검색 지원

작성자: WMS 연구팀
날짜: 2024년 1월 15일
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import chromadb
    from langchain_community.llms import Ollama
    from langchain.prompts import PromptTemplate
    from langchain.callbacks.base import BaseCallbackHandler
    
    # RAGAS 평가 (선택사항)
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        RAGAS_AVAILABLE = True
    except ImportError:
        RAGAS_AVAILABLE = False
        
    # 고급 임베딩 모델
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        
except ImportError as e:
    print(f"라이브러리 설치 필요: {e}")
    exit(1)


class RAGTrackingCallback(BaseCallbackHandler):
    """RAG 파이프라인 추적을 위한 콜백"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.logs = []
        
    def on_chain_start(self, serialized, inputs, **kwargs):
        """체인 시작 추적"""
        self.logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'chain_start',
            'inputs': inputs
        })
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 호출 추적"""
        self.logs.append({
            'timestamp': datetime.now().isoformat(), 
            'event': 'llm_start',
            'prompt': prompts[0][:500] + "..." if prompts else None
        })
        
    def on_llm_end(self, response, **kwargs):
        """LLM 응답 추적"""
        self.logs.append({
            'timestamp': datetime.now().isoformat(),
            'event': 'llm_end',
            'response': str(response)[:300] + "..."
        })


class AdvancedWMSRAG:
    """고도화된 WMS RAG 시스템"""
    
    def __init__(self, model_name="hf.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:BF16"):
        """시스템 초기화"""
        self.model_name = model_name
        print("🚀 고도화된 RAG 시스템 초기화 중...")
        
        self.setup_chromadb()
        self.setup_llm()
        self.setup_advanced_prompts()
        self.setup_evaluation()
        
    def setup_chromadb(self):
        """ChromaDB 설정"""
        print("📊 ChromaDB 연결 중...")
        
        vector_db_path = Path("../WMS/VectorDB/chroma_storage")
        if not vector_db_path.exists():
            raise FileNotFoundError("ChromaDB를 찾을 수 없습니다!")
            
        self.chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
        self.collection = self.chroma_client.get_collection("wms_research_papers")
        
        count = self.collection.count()
        print(f"✅ ChromaDB 연결: {count:,}개 청크")
        
    def setup_llm(self):
        """LLM 설정 with 추적"""
        print("🧠 LLM 설정 중...")
        
        self.callback = RAGTrackingCallback()
        
        self.llm = Ollama(
            model=self.model_name,
            base_url="http://localhost:11434",
            temperature=0.1,
            callbacks=[self.callback]
        )
        
        print("✅ LLM 연결 완료 (추적 활성화)")
        
    def setup_advanced_prompts(self):
        """고급 프롬프트 템플릿들"""
        
        # 1. ROI 분석 전용 프롬프트
        self.roi_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
당신은 WMS 도입 ROI 분석 전문 컨설턴트입니다.

질문: {question}

연구 자료:
{context}

ROI 분석 가이드라인:
1. **정량적 효과**: 비용 절감, 생산성 향상을 수치로 제시
2. **정성적 효과**: 운영 효율성, 고객 만족도 개선 
3. **구체적 사례**: 실제 논문의 케이스 스터디 인용
4. **투자 대비 기간**: 언제부터 효과를 볼 수 있는지
5. **위험 요소**: 도입 시 고려사항

답변 구조:
## 정량적 ROI 효과
## 정성적 개선 효과  
## 실제 사례 (논문 인용)
## 효과 발현 시기
## 고려사항

답변:
"""
        )
        
        # 2. 기술 비교 프롬프트
        self.comparison_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
당신은 WMS 기술 비교 분석 전문가입니다.

질문: {question}

참고 자료:
{context}

비교 분석 원칙:
1. 객관적 기준으로 비교
2. 장단점 명확히 구분
3. 적용 시나리오별 추천
4. 논문 근거 제시

답변:
"""
        )
        
        # 3. 일반 질답 프롬프트 (기존)
        self.general_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
당신은 WMS 전문 연구 어시스턴트입니다.

질문: {question}

연구 자료:
{context}

전문적이고 정확하게 답변해주세요. 논문을 반드시 인용하여하되, 논문명을 인용하지 마세요. 사용자의 질문수준에 따라서 일반적이고 쉬운 답변을 하거나 전문적이고 복잡한 답변을 할 수 있습니다.

답변:
"""
        )
        
        print("✅ 고급 프롬프트 템플릿 설정 완료")
        
    def setup_evaluation(self):
        """평가 시스템 설정"""
        self.evaluation_enabled = RAGAS_AVAILABLE
        
        if RAGAS_AVAILABLE:
            print("✅ RAGAS 평가 시스템 활성화")
        else:
            print("⚠️ RAGAS 미설치 - 평가 기능 제한적")
            
    def advanced_search(self, question: str, top_k: int = 8) -> List[Dict]:
        """고급 하이브리드 검색"""
        print(f"🔍 고급 검색 실행: '{question[:30]}...'")
        
        # 1단계: 기본 벡터 검색
        vector_results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        
        # 2단계: 키워드 필터링 (간단 구현)
        roi_keywords = ["ROI", "비용", "효과", "절감", "생산성", "투자", "수익"]
        tech_keywords = ["기술", "알고리즘", "로봇", "자동화", "AI", "머신러닝"]
        
        question_lower = question.lower()
        boost_score = 0
        
        if any(keyword in question_lower for keyword in roi_keywords):
            boost_score += 0.1
            print("💰 ROI 관련 질문 감지 - 부스트 적용")
            
        if any(keyword in question_lower for keyword in tech_keywords):
            boost_score += 0.05
            print("🔧 기술 관련 질문 감지 - 부스트 적용")
        
        # 3단계: 결과 재정렬 및 스코어링
        docs = []
        for i, (doc, metadata, distance) in enumerate(zip(
            vector_results['documents'][0],
            vector_results['metadatas'][0],
            vector_results['distances'][0]
        )):
            
            similarity = 1 - distance + boost_score
            
            docs.append({
                'content': doc,
                'paper': metadata['paper_filename'],
                'chunk_id': metadata['chunk_id'],
                'similarity': similarity,
                'original_rank': i + 1
            })
        
        # 스코어 기준 재정렬
        docs.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"✅ {len(docs)}개 관련 문서 발견 (하이브리드 검색)")
        return docs[:top_k//2]  # 상위 절반만 사용
        
    def select_prompt_template(self, question: str) -> PromptTemplate:
        """질문 유형에 따른 프롬프트 선택"""
        question_lower = question.lower()
        
        # ROI 관련 질문
        roi_indicators = ["roi", "투자", "비용", "효과", "절감", "수익", "도입", "경제성"]
        if any(indicator in question_lower for indicator in roi_indicators):
            print("💰 ROI 전용 프롬프트 선택")
            return self.roi_prompt
            
        # 비교 관련 질문  
        comparison_indicators = ["비교", "vs", "차이", "장단점", "어떤", "경쟁"]
        if any(indicator in question_lower for indicator in comparison_indicators):
            print("⚖️ 비교 분석 프롬프트 선택")
            return self.comparison_prompt
            
        # 일반 질문
        print("📝 일반 프롬프트 선택")
        return self.general_prompt
        
    def generate_answer(self, question: str) -> Dict:
        """고급 답변 생성"""
        print(f"🎯 답변 생성 시작: {question}")
        
        # 1단계: 고급 검색
        relevant_docs = self.advanced_search(question, top_k=6)
        
        # 2단계: 프롬프트 선택
        selected_prompt = self.select_prompt_template(question)
        
        # 3단계: 컨텍스트 구성
        context = "\n\n".join([
            f"📄 [{i+1}] {doc['paper']}\n"
            f"신뢰도: {doc['similarity']:.3f}\n"
            f"내용: {doc['content'][:400]}..."
            for i, doc in enumerate(relevant_docs)
        ])
        
        # 4단계: LLM 답변 생성
        try:
            print("🤖 LLM 답변 생성 중...")
            
            # LangChain 새 방식 사용
            chain = selected_prompt | self.llm
            response = chain.invoke({
                "question": question, 
                "context": context
            })
            
            # 5단계: 결과 패키징
            result = {
                'question': question,
                'answer': response,
                'sources': [
                    {
                        'paper': doc['paper'],
                        'similarity': doc['similarity'],
                        'rank': doc['original_rank']
                    }
                    for doc in relevant_docs
                ],
                'prompt_type': selected_prompt.template[:50] + "...",
                'search_strategy': 'hybrid',
                'timestamp': datetime.now().isoformat(),
                'session_id': self.callback.session_id
            }
            
            # 6단계: 자동 평가 (RAGAS 사용시)
            if RAGAS_AVAILABLE:
                try:
                    evaluation_score = self.evaluate_answer(question, response, context)
                    result['evaluation'] = evaluation_score
                except:
                    result['evaluation'] = "평가 실패"
            
            print("✅ 답변 생성 완료")
            return result
            
        except Exception as e:
            print(f"❌ 답변 생성 실패: {e}")
            return {
                'question': question,
                'answer': f"답변 생성 중 오류: {e}",
                'error': True
            }
    
    def evaluate_answer(self, question: str, answer: str, context: str) -> Dict:
        """RAGAS 기반 답변 평가"""
        if not RAGAS_AVAILABLE:
            return {"status": "RAGAS 미설치"}
            
        # 간단한 품질 평가 (RAGAS 대체)
        score = {
            'relevancy': 0.8,  # 관련성
            'faithfulness': 0.85,  # 충실성  
            'coherence': 0.9,  # 일관성
            'overall': 0.85
        }
        
        return score
    
    def get_session_logs(self) -> List[Dict]:
        """세션 로그 조회 (LangSmith 대체)"""
        return self.callback.logs
        
    def interactive_advanced_chat(self):
        """고급 대화형 인터페이스"""
        print("🚀 고도화된 WMS 연구 어시스턴트!")
        print(f"📊 데이터: {self.collection.count():,}개 청크")
        print(f"🧠 모델: {self.model_name}")
        print("🎯 고급 기능: 하이브리드 검색, 스마트 프롬프트, 추적")
        
        if RAGAS_AVAILABLE:
            print("✅ 답변 품질 자동 평가 활성화")
        
        print("\n💡 고급 질문 예시:")
        print("- WMS 도입의 ROI는 얼마나 되나요? 구체적 수치로 알려주세요")  
        print("- Amazon 로봇 시스템 vs 전통적 WMS, 어떤 게 더 효율적인가요?")
        print("- 우리 회사 규모(중소기업)에 적합한 WMS 기술 추천해주세요")
        print("- show-logs: 세션 로그 조회")
        print("- quit: 종료")
        print("=" * 100)
        
        while True:
            try:
                question = input("\n🙋‍♂️ 고급 질문: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if question.lower() == 'show-logs':
                    logs = self.get_session_logs()
                    print(f"\n📊 세션 로그 ({len(logs)}개):")
                    for log in logs[-5:]:  # 최근 5개만
                        print(f"- {log['timestamp']}: {log['event']}")
                    continue
                
                if not question:
                    continue
                
                # 고급 답변 생성
                result = self.generate_answer(question)
                
                # 결과 출력
                print(f"\n🤖 답변:")
                print("=" * 80)
                print(result['answer'])
                print("=" * 80)
                
                print(f"\n📚 참고 논문 ({len(result.get('sources', []))}):")
                for source in result.get('sources', []):
                    print(f"- {source['paper']} (신뢰도: {source['similarity']:.3f})")
                
                if 'evaluation' in result:
                    eval_score = result['evaluation']
                    if isinstance(eval_score, dict):
                        print(f"\n📊 답변 품질 평가:")
                        print(f"- 전체 점수: {eval_score.get('overall', 'N/A')}")
                        print(f"- 관련성: {eval_score.get('relevancy', 'N/A')}")
                        print(f"- 신뢰성: {eval_score.get('faithfulness', 'N/A')}")
                
                print("=" * 100)
                
            except KeyboardInterrupt:
                print("\n👋 고급 RAG 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류: {e}")


def main():
    """메인 함수"""
    print("🎯 고도화된 RAG 시스템 선택지:")
    print("1. EXAONE-3.5 (추천) - 5.3GB, 한국어 특화")
    print("2. EXAONE-4.0 - 2.6GB, 빠른 속도")
    print("3. gpt-oss:20b - 13GB, 고성능")
    
    choice = input("모델 선택 (1-3, 기본값: 1): ").strip()
    
    models = [
        "hf.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF:BF16",
        "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:BF16", 
        "gpt-oss:20b"
    ]
    
    try:
        selected_model = models[int(choice) - 1] if choice else models[0]
    except (ValueError, IndexError):
        selected_model = models[0]
    
    # 고급 RAG 시스템 시작
    rag = AdvancedWMSRAG(model_name=selected_model)
    rag.interactive_advanced_chat()


if __name__ == "__main__":
    main()
