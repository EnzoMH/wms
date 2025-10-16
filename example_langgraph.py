#!/usr/bin/env python3
"""
LangGraph 예시 코드
==================

Multi-Agent RAG 워크플로우 예시
"""

from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator


# 1. State 정의
class AgentState(TypedDict):
    """에이전트 상태"""
    query: str
    analyzed_query: str
    retrieved_docs: List[str]
    reranked_docs: List[str]
    final_answer: str
    metadata: dict


# 2. 각 노드 함수 정의
def analyze_query(state: AgentState) -> AgentState:
    """질문 분석 노드"""
    query = state["query"]
    
    # 질문 의도 분석 (간단한 예시)
    if "차이" in query or "비교" in query:
        analyzed = f"[비교 쿼리] {query}"
    elif "방법" in query or "어떻게" in query:
        analyzed = f"[방법 쿼리] {query}"
    else:
        analyzed = f"[일반 쿼리] {query}"
    
    print(f"[Analyzer] {analyzed}")
    
    return {
        **state,
        "analyzed_query": analyzed,
        "metadata": {**state.get("metadata", {}), "query_type": "comparison" if "차이" in query else "general"}
    }


def retrieve_documents(state: AgentState) -> AgentState:
    """문서 검색 노드"""
    query = state["analyzed_query"]
    
    # FAISS에서 문서 검색 (실제 구현 필요)
    # 여기서는 더미 문서 사용
    docs = [
        f"검색된 문서 1: {query}에 대한 내용...",
        f"검색된 문서 2: 관련 정보...",
        f"검색된 문서 3: 추가 세부사항..."
    ]
    
    print(f"[Retriever] {len(docs)}개 문서 검색")
    
    return {
        **state,
        "retrieved_docs": docs
    }


def rerank_documents(state: AgentState) -> AgentState:
    """문서 재순위화 노드"""
    docs = state["retrieved_docs"]
    query = state["query"]
    
    # Cross-Encoder로 재순위화 (실제 구현 필요)
    # 여기서는 간단히 역순으로
    reranked = list(reversed(docs))
    
    print(f"[Reranker] 문서 재순위화 완료")
    
    return {
        **state,
        "reranked_docs": reranked
    }


def generate_answer(state: AgentState) -> AgentState:
    """답변 생성 노드"""
    query = state["query"]
    docs = state["reranked_docs"]
    
    # EXAONE으로 답변 생성 (실제 구현 필요)
    context = "\n".join(docs[:3])  # 상위 3개 문서
    answer = f"질문: {query}\n\n컨텍스트 기반 답변:\n{context[:200]}..."
    
    print(f"[Generator] 답변 생성 완료")
    
    return {
        **state,
        "final_answer": answer
    }


# 3. 조건부 라우팅 함수
def should_rerank(state: AgentState) -> str:
    """재순위화 필요 여부 판단"""
    # 문서가 3개 이상이면 재순위화
    if len(state.get("retrieved_docs", [])) >= 3:
        return "rerank"
    else:
        return "generate"


# 4. 워크플로우 구성
def create_rag_workflow():
    """Multi-Agent RAG 워크플로우 생성"""
    
    # StateGraph 생성
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("analyzer", analyze_query)
    workflow.add_node("retriever", retrieve_documents)
    workflow.add_node("reranker", rerank_documents)
    workflow.add_node("generator", generate_answer)
    
    # 엣지 추가 (워크플로우 정의)
    workflow.set_entry_point("analyzer")
    workflow.add_edge("analyzer", "retriever")
    
    # 조건부 엣지 (재순위화 여부)
    workflow.add_conditional_edges(
        "retriever",
        should_rerank,
        {
            "rerank": "reranker",
            "generate": "generator"
        }
    )
    
    workflow.add_edge("reranker", "generator")
    workflow.add_edge("generator", END)
    
    # 컴파일
    app = workflow.compile()
    
    return app


# 5. 사용 예시
def example_usage():
    """LangGraph 사용 예시"""
    
    print("=" * 60)
    print("LangGraph Multi-Agent RAG 워크플로우 예시")
    print("=" * 60)
    
    # 워크플로우 생성
    app = create_rag_workflow()
    
    # 테스트 쿼리
    test_queries = [
        "AGV와 AMR의 차이점은?",
        "창고 자동화 방법은?",
        "WMS 시스템이란?"
    ]
    
    for query in test_queries:
        print(f"\n질문: {query}")
        print("-" * 60)
        
        # 초기 상태
        initial_state = {
            "query": query,
            "analyzed_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_answer": "",
            "metadata": {}
        }
        
        # 워크플로우 실행
        result = app.invoke(initial_state)
        
        print(f"\n최종 답변:")
        print(result["final_answer"][:200] + "...")
        print("=" * 60)


# 6. FastAPI 통합 예시
def fastapi_integration_example():
    """
    FastAPI에 통합하는 방법
    
    app/chat/router.py에 추가:
    
    from langgraph_workflow import create_rag_workflow
    
    # 전역 변수로 워크플로우 생성
    rag_workflow = create_rag_workflow()
    
    @router.post("/chat/langgraph")
    async def chat_with_langgraph(request: ChatRequest):
        initial_state = {
            "query": request.query,
            "analyzed_query": "",
            "retrieved_docs": [],
            "reranked_docs": [],
            "final_answer": "",
            "metadata": {}
        }
        
        result = rag_workflow.invoke(initial_state)
        
        return {
            "query": request.query,
            "response": result["final_answer"],
            "metadata": result["metadata"]
        }
    """
    pass


# 7. 고급 워크플로우 예시
def create_advanced_workflow():
    """
    고급 워크플로우 예시
    
    특징:
    - 질문 타입에 따라 다른 처리
    - 자가 검증 (Self-Verification)
    - 반복 개선 (Iterative Refinement)
    """
    
    workflow = StateGraph(AgentState)
    
    # 노드 추가
    workflow.add_node("classifier", classify_question)  # 질문 분류
    workflow.add_node("simple_rag", simple_rag_chain)   # 간단한 RAG
    workflow.add_node("complex_rag", complex_rag_chain) # 복잡한 RAG
    workflow.add_node("verifier", verify_answer)        # 답변 검증
    workflow.add_node("refiner", refine_answer)         # 답변 개선
    
    # 조건부 라우팅
    workflow.set_entry_point("classifier")
    workflow.add_conditional_edges(
        "classifier",
        route_by_complexity,
        {
            "simple": "simple_rag",
            "complex": "complex_rag"
        }
    )
    
    workflow.add_edge("simple_rag", "verifier")
    workflow.add_edge("complex_rag", "verifier")
    
    workflow.add_conditional_edges(
        "verifier",
        check_quality,
        {
            "good": END,
            "refine": "refiner"
        }
    )
    
    workflow.add_edge("refiner", "verifier")
    
    return workflow.compile()


# 스텁 함수들 (실제 구현 필요)
def classify_question(state): return state
def simple_rag_chain(state): return state
def complex_rag_chain(state): return state
def verify_answer(state): return state
def refine_answer(state): return state
def route_by_complexity(state): return "simple"
def check_quality(state): return "good"


if __name__ == "__main__":
    # 예시 실행
    example_usage()
    
    print("\n\nFastAPI 통합 방법은 코드 내 주석 참조")

