#!/usr/bin/env python3
"""
WMS Research Service
===================

WMS 연구 논문 검색 및 분석 서비스
"""

from typing import List, Dict, Any, Optional
import logging
from .wms_faiss_repository import WMSFaissRepository

logger = logging.getLogger(__name__)

class WMSResearchService:
    """WMS 연구 논문 검색 및 분석 서비스"""
    
    def __init__(self, repository: WMSFaissRepository, llm=None):
        """
        WMS Research Service 초기화
        
        Args:
            repository: WMS Faiss Repository
            llm: LLM 모델 (옵션)
        """
        self.repository = repository
        self.llm = llm
        
        logger.info("✅ WMS Research Service 초기화 완료")
    
    def search_papers(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        논문 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            
        Returns:
            검색 결과
        """
        results = self.repository.search(query, top_k)
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
    
    def ask_question(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        RAG 기반 질의응답
        
        Args:
            question: 질문
            top_k: 참고할 문서 수
            
        Returns:
            답변 및 참고 문서
        """
        if not self.llm:
            return {"error": "LLM이 설정되지 않았습니다"}
        
        try:
            # 1. 관련 문서 검색
            search_results = self.repository.search(question, top_k)
            
            if not search_results:
                return {
                    "answer": "관련된 연구 자료를 찾을 수 없습니다.",
                    "sources": [],
                    "search_results": []
                }
            
            # 2. 컨텍스트 구성
            context_parts = []
            sources = []
            
            for i, result in enumerate(search_results):
                paper_name = result['metadata']['paper_filename'].replace('.pdf', '')
                content = result['document'][:800]  # 적절한 길이로 제한
                
                context_parts.append(f"[문서 {i+1}] {paper_name}\n{content}")
                sources.append(paper_name)
            
            context = "\n\n".join(context_parts)
            
            # 3. 프롬프트 구성
            prompt = f"""다음은 창고 관리 시스템(WMS) 관련 연구 논문들의 내용입니다.

=== 관련 문서들 ===
{context}

=== 질문 ===
{question}

=== 지시사항 ===
위 문서들을 참고하여 질문에 대해 정확하고 전문적으로 답변해주세요.
- 기술적 내용은 구체적으로 설명해주세요
- 답변 마지막에 참고한 논문들을 명시해주세요
- 한국어로 답변해주세요
"""
            
            # 4. LLM 응답 생성
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "sources": sources,
                "search_results": search_results[:3]  # 상위 3개만 반환
            }
            
        except Exception as e:
            logger.error(f"❌ 질의응답 실패: {e}")
            return {
                "error": f"질의응답 처리 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "search_results": []
            }
    
    def get_research_trends(self, topic: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """
        연구 동향 분석
        
        Args:
            topic: 분석할 주제 (None이면 전체)
            top_k: 분석할 문서 수
            
        Returns:
            연구 동향 분석 결과
        """
        try:
            if topic:
                # 특정 주제 검색
                search_results = self.repository.search(topic, top_k)
                papers = {}
                
                for result in search_results:
                    filename = result['metadata']['paper_filename']
                    if filename not in papers:
                        papers[filename] = {
                            'title': filename.replace('.pdf', ''),
                            'source': result['metadata']['paper_source'],
                            'chunks': 0,
                            'total_relevance': 0,
                            'avg_relevance': 0
                        }
                    papers[filename]['chunks'] += 1
                    papers[filename]['total_relevance'] += result['similarity']
                
                # 평균 관련도 계산
                for paper in papers.values():
                    paper['avg_relevance'] = paper['total_relevance'] / paper['chunks']
                    del paper['total_relevance']  # 불필요한 필드 제거
                
                return {
                    "topic": topic,
                    "analysis_type": "topic_specific",
                    "total_papers": len(papers),
                    "papers": sorted(papers.values(), key=lambda x: x['avg_relevance'], reverse=True)
                }
            
            else:
                # 전체 데이터베이스 통계
                stats = self.repository.get_database_stats()
                
                return {
                    "topic": "전체 데이터베이스",
                    "analysis_type": "database_overview",
                    "total_documents": stats['total_documents'],
                    "total_papers": stats['total_papers'],
                    "papers_by_source": stats['papers_by_source'],
                    "top_papers": [{"title": title.replace('.pdf', ''), "chunks": count} 
                                 for title, count in stats['top_papers']]
                }
                
        except Exception as e:
            logger.error(f"❌ 연구 동향 분석 실패: {e}")
            return {"error": f"연구 동향 분석 중 오류가 발생했습니다: {str(e)}"}
    
    def get_paper_details(self, paper_filename: str) -> Dict[str, Any]:
        """
        특정 논문의 상세 정보 조회
        
        Args:
            paper_filename: 논문 파일명
            
        Returns:
            논문 상세 정보
        """
        try:
            chunks = self.repository.search_by_paper(paper_filename)
            
            if not chunks:
                return {"error": f"논문을 찾을 수 없습니다: {paper_filename}"}
            
            # 기본 정보
            first_chunk = chunks[0]
            metadata = first_chunk['metadata']
            
            # 유사한 논문들
            similar_papers = self.repository.get_similar_papers(paper_filename, 5)
            
            return {
                "paper_filename": paper_filename,
                "title": paper_filename.replace('.pdf', ''),
                "source": metadata['paper_source'],
                "total_chunks": len(chunks),
                "total_characters": sum(len(chunk['document']) for chunk in chunks),
                "similar_papers": similar_papers,
                "first_chunk_preview": chunks[0]['document'][:500] + "..." if chunks[0]['document'] else ""
            }
            
        except Exception as e:
            logger.error(f"❌ 논문 상세 정보 조회 실패: {e}")
            return {"error": f"논문 정보 조회 중 오류가 발생했습니다: {str(e)}"}
    
    def compare_papers(self, paper1: str, paper2: str) -> Dict[str, Any]:
        """
        두 논문 비교 분석
        
        Args:
            paper1: 첫 번째 논문 파일명
            paper2: 두 번째 논문 파일명
            
        Returns:
            비교 분석 결과
        """
        try:
            # 각 논문의 첫 번째 청크로 유사도 검색
            chunks1 = self.repository.search_by_paper(paper1)
            chunks2 = self.repository.search_by_paper(paper2)
            
            if not chunks1 or not chunks2:
                return {"error": "논문 중 하나 이상을 찾을 수 없습니다"}
            
            # 첫 번째 논문으로 검색하여 두 번째 논문과의 유사도 확인
            search_results = self.repository.search(chunks1[0]['document'], 20)
            
            similarity_score = 0
            for result in search_results:
                if result['metadata']['paper_filename'] == paper2:
                    similarity_score = result['similarity']
                    break
            
            return {
                "paper1": {
                    "filename": paper1,
                    "title": paper1.replace('.pdf', ''),
                    "chunks": len(chunks1),
                    "source": chunks1[0]['metadata']['paper_source']
                },
                "paper2": {
                    "filename": paper2,
                    "title": paper2.replace('.pdf', ''),
                    "chunks": len(chunks2),
                    "source": chunks2[0]['metadata']['paper_source']
                },
                "similarity_score": similarity_score,
                "comparison_summary": f"두 논문의 유사도는 {similarity_score:.3f}입니다."
            }
            
        except Exception as e:
            logger.error(f"❌ 논문 비교 실패: {e}")
            return {"error": f"논문 비교 중 오류가 발생했습니다: {str(e)}"}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보 조회"""
        return self.repository.get_database_stats()
    
    def search_by_source(self, source: str, limit: int = 10) -> Dict[str, Any]:
        """
        특정 소스의 논문들 조회
        
        Args:
            source: 논문 소스 (ArXiv, IEEE, etc.)
            limit: 반환할 논문 수
            
        Returns:
            해당 소스의 논문 목록
        """
        try:
            papers = self.repository.get_papers_by_source(source)
            
            return {
                "source": source,
                "total_papers": len(papers),
                "papers": papers[:limit]
            }
            
        except Exception as e:
            logger.error(f"❌ 소스별 검색 실패: {e}")
            return {"error": f"소스별 검색 중 오류가 발생했습니다: {str(e)}"}


