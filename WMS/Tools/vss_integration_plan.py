#!/usr/bin/env python3
"""
WMS Faiss 시스템을 VSS-AI-API-dev에 통합하는 계획
=====================================================

VSS-AI-API-dev의 구조를 분석하여 WMS Faiss 시스템을 통합하는 방안을 제시합니다.
"""

# ============================================================================
# 방안 1: 새로운 WMS 전용 라우터 추가 (추천)
# ============================================================================

"""
1. 디렉토리 구조:
VSS-AI-API-dev/
├── app/
│   └── wms_research/           # 새로 추가
│       └── v1/
│           ├── model/
│           │   └── dto/
│           │       ├── request_dto.py
│           │       └── response_dto.py
│           ├── repository/
│           │   └── wms_faiss_repository.py
│           ├── router/
│           │   └── wms_research_router.py
│           └── service/
│               └── wms_research_service.py
├── resource/
│   └── wms_knowledge/          # 새로 추가
│       ├── faiss_storage/      # 우리가 만든 Faiss 시스템
│       │   ├── wms_knowledge.index
│       │   ├── documents.json
│       │   ├── metadata.json
│       │   └── config.json
│       └── processed_data/     # 원본 청크 데이터 (옵션)
└── core/
    └── wms_vector_store.py     # WMS 벡터 스토어 관리자
"""

# ============================================================================
# 구현 코드 예시
# ============================================================================

# 1. WMS Faiss Repository
class WMSFaissRepository:
    """
    WMS Faiss 벡터 데이터베이스 접근 클래스
    """
    def __init__(self, faiss_storage_path: str):
        import faiss
        import json
        from pathlib import Path
        
        self.storage_path = Path(faiss_storage_path)
        self.index = faiss.read_index(str(self.storage_path / "wms_knowledge.index"))
        
        # 문서와 메타데이터 로드
        with open(self.storage_path / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(self.storage_path / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadatas = json.load(f)
            
        # 임베딩 모델 초기화
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': 'cuda' if self._cuda_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def search(self, query: str, top_k: int = 5):
        """벡터 검색 수행"""
        import numpy as np
        
        # 쿼리 임베딩
        query_embedding = self.embedding_model.embed_query(query)
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Faiss 검색
        scores, indices = self.index.search(query_vector, top_k)
        
        # 결과 구성
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.documents):
                results.append({
                    'rank': i + 1,
                    'document': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'score': float(score),
                    'similarity': float(1 / (1 + score))
                })
        
        return results

# 2. WMS Research Service
class WMSResearchService:
    """
    WMS 연구 논문 검색 및 분석 서비스
    """
    def __init__(self, repository: WMSFaissRepository):
        self.repository = repository
        
        # LLM 초기화 (기존 VSS 시스템과 동일)
        from core.langchain_manager import model_korean_normal
        self.llm = model_korean_normal
    
    def search_papers(self, query: str, top_k: int = 5):
        """논문 검색"""
        return self.repository.search(query, top_k)
    
    def ask_question(self, question: str, top_k: int = 3):
        """RAG 기반 질의응답"""
        # 1. 관련 문서 검색
        search_results = self.repository.search(question, top_k)
        
        # 2. 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {i+1}] {result['metadata']['paper_filename']}\n{result['document'][:500]}..."
            for i, result in enumerate(search_results)
        ])
        
        # 3. 프롬프트 구성
        prompt = f"""
다음은 창고 관리 시스템(WMS) 관련 연구 논문들의 내용입니다.

=== 관련 문서들 ===
{context}

=== 질문 ===
{question}

=== 지시사항 ===
위 문서들을 참고하여 질문에 대해 정확하고 전문적으로 답변해주세요.
답변 시 참고한 논문의 제목도 함께 언급해주세요.
"""
        
        # 4. LLM 응답 생성
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content,
            "sources": [r['metadata']['paper_filename'] for r in search_results],
            "search_results": search_results
        }
    
    def get_research_trends(self, topic: str = None):
        """연구 동향 분석"""
        if topic:
            results = self.repository.search(topic, 10)
        else:
            # 전체 데이터베이스 통계
            results = []
        
        # 논문별 통계 계산
        papers = {}
        for result in results:
            filename = result['metadata']['paper_filename']
            if filename not in papers:
                papers[filename] = {
                    'title': filename,
                    'source': result['metadata']['paper_source'],
                    'chunks': 0,
                    'relevance': 0
                }
            papers[filename]['chunks'] += 1
            papers[filename]['relevance'] += result['similarity']
        
        # 평균 관련도 계산
        for paper in papers.values():
            paper['avg_relevance'] = paper['relevance'] / paper['chunks']
        
        return {
            "topic": topic or "전체",
            "total_papers": len(papers),
            "papers": list(papers.values())
        }

# 3. WMS Research Router
"""
from fastapi import APIRouter, Body, Depends
from app.wms_research.v1.service.wms_research_service import WMSResearchService
from app.wms_research.v1.model.dto.request_dto import *
from app.wms_research.v1.model.dto.response_dto import *

router = APIRouter(
    prefix="/wms-research",
    tags=["wms-research"],
    responses={404: {"description": "Not found"}},
)

def get_wms_research_service():
    from core.wms_vector_store import wms_faiss_repository
    return WMSResearchService(wms_faiss_repository)

@router.post("/search", description="WMS 연구 논문 검색")
async def search_papers(
    request: SearchRequestDto = Body(...),
    service: WMSResearchService = Depends(get_wms_research_service)
) -> SearchResponseDto:
    results = service.search_papers(request.query, request.top_k or 5)
    return SearchResponseDto(
        query=request.query,
        results=results,
        total_found=len(results)
    )

@router.post("/ask", description="WMS 연구 질의응답 (RAG)")
async def ask_question(
    request: QuestionRequestDto = Body(...),
    service: WMSResearchService = Depends(get_wms_research_service)
) -> QuestionResponseDto:
    result = service.ask_question(request.question, request.top_k or 3)
    return QuestionResponseDto(**result)

@router.get("/trends", description="WMS 연구 동향 분석")
async def get_research_trends(
    topic: str = None,
    service: WMSResearchService = Depends(get_wms_research_service)
) -> TrendsResponseDto:
    trends = service.get_research_trends(topic)
    return TrendsResponseDto(**trends)

@router.get("/stats", description="데이터베이스 통계")
async def get_database_stats(
    service: WMSResearchService = Depends(get_wms_research_service)
) -> dict:
    # 데이터베이스 통계 반환
    return {
        "total_documents": len(service.repository.documents),
        "total_papers": len(set(m['paper_filename'] for m in service.repository.metadatas)),
        "embedding_model": "jhgan/ko-sroberta-multitask",
        "vector_dimension": 768
    }
"""

# ============================================================================
# 방안 2: 기존 VSS 봇에 WMS 기능 추가
# ============================================================================

"""
기존 vss_bot_service.py에 WMS 관련 메서드 추가:

class VssBotService:
    def __init__(self, vector_store: FAISS, wms_repository: WMSFaissRepository = None):
        self.vector_store = vector_store
        self.wms_repository = wms_repository
    
    def query_wms_research(self, query: str, top_k: int = 5):
        '''WMS 연구 논문 검색'''
        if not self.wms_repository:
            return {"error": "WMS research database not available"}
        
        return self.wms_repository.search(query, top_k)
    
    def wms_rag_query(self, question: str):
        '''WMS RAG 질의응답'''
        # WMS 검색 + LLM 응답 생성
        pass
"""

# ============================================================================
# 방안 3: 하이브리드 벡터 스토어 (고급)
# ============================================================================

"""
VSS 함수 검색 + WMS 연구 검색을 통합한 하이브리드 시스템:

class HybridVectorService:
    def __init__(self):
        self.vss_store = vss_function_vector_store  # 기존
        self.wms_store = wms_faiss_repository       # 새로 추가
    
    def unified_search(self, query: str, search_type: str = "auto"):
        '''통합 검색'''
        if search_type == "auto":
            # 쿼리 분석하여 자동 판단
            if self._is_function_query(query):
                return self.vss_store.similarity_search(query)
            else:
                return self.wms_store.search(query)
        elif search_type == "function":
            return self.vss_store.similarity_search(query)
        elif search_type == "research":
            return self.wms_store.search(query)
        else:
            # 둘 다 검색하여 결합
            vss_results = self.vss_store.similarity_search(query, k=3)
            wms_results = self.wms_store.search(query, 3)
            return self._merge_results(vss_results, wms_results)
"""

print("🎯 WMS Faiss 시스템 VSS 통합 계획 완료!")
print("추천: 방안 1 - 새로운 WMS 전용 라우터 추가")
print("장점: 기존 시스템 영향 없음, 확장성 좋음, 유지보수 용이")


