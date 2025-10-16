#!/usr/bin/env python3
"""
Advanced RAG + Modular RAG 시스템
====================================

링크 참조: https://g3lu.tistory.com/42
- Advanced RAG: Pre-Retrieval + Post-Retrieval
- Modular RAG: Search Module + Memory Module + Fusion Module

작성자: WMS 팀
날짜: 2025년 10월 13일
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import faiss
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import openai
except ImportError as e:
    print(f"필수 라이브러리 설치 필요: {e}")

logger = logging.getLogger(__name__)

@dataclass
class RAGResult:
    """RAG 검색 결과 데이터 클래스"""
    content: str
    metadata: Dict[str, Any]
    similarity: float
    rank: int
    source: str = "retrieval"

class BaseRAGModule(ABC):
    """RAG 모듈 기본 클래스"""
    
    @abstractmethod
    def process(self, query: str, **kwargs) -> Any:
        pass

class PreRetrievalModule(BaseRAGModule):
    """
    Pre-Retrieval 모듈
    
    링크 참조 내용:
    - 데이터 품질 향상: 엔터티와 용어 모호성 제거
    - 인덱스 구조 최적화: 청크 크기 최적화, 그래프 구조 정보 추가
    - 메타데이터 추가: dates, chapters, subsections, purposes
    - 청크 최적화: 더 작은 조각으로 나누어 세부적인 특성 추출
    """
    
    def __init__(self):
        self.wms_keywords = {
            "systems": ["WCS", "WES", "MES", "ERP", "TMS", "YMS"],
            "robots": ["AMR", "AGV", "RTV", "CNV", "ASRS", "Sorter"],
            "operations": ["picking", "packing", "sorting", "storage", "dispatch"],
            "technologies": ["RFID", "barcode", "IoT", "AI", "ML", "computer vision"]
        }
        
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """쿼리 전처리 및 확장"""
        logger.info("Pre-Retrieval: 쿼리 최적화 시작")
        
        # 1. 쿼리 정제
        cleaned_query = self._clean_query(query)
        
        # 2. 전문 용어 확장
        expanded_query = self._expand_technical_terms(cleaned_query)
        
        # 3. 다중 쿼리 생성 (Modular RAG - Query Translation)
        multi_queries = self._generate_multiple_queries(expanded_query)
        
        # 4. 메타데이터 필터 생성
        metadata_filters = self._generate_metadata_filters(expanded_query)
        
        return {
            "original_query": query,
            "cleaned_query": cleaned_query,
            "expanded_query": expanded_query,
            "multi_queries": multi_queries,
            "metadata_filters": metadata_filters,
            "enhancement_applied": True
        }
    
    def _clean_query(self, query: str) -> str:
        """쿼리 정제"""
        # 특수문자 제거, 소문자 변환 등
        import re
        cleaned = re.sub(r'[^\w\s가-힣]', ' ', query)
        cleaned = ' '.join(cleaned.split())
        return cleaned
        
    def _expand_technical_terms(self, query: str) -> str:
        """전문 용어 확장"""
        expanded = query
        
        for category, terms in self.wms_keywords.items():
            for term in terms:
                if term.lower() in query.lower():
                    # 관련 용어들 추가
                    related_terms = [t for t in terms if t != term][:2]
                    if related_terms:
                        expanded += f" {' '.join(related_terms)}"
                        
        return expanded
    
    def _generate_multiple_queries(self, query: str) -> List[str]:
        """다중 쿼리 생성 (RAG-Fusion 개념)"""
        queries = [query]
        
        # 산업용 로봇 관점에서 쿼리 변형
        variations = [
            f"창고 자동화에서 {query}는 어떻게 사용되는가?",
            f"{query}의 기술적 특징과 장점은?",
            f"산업용 로봇 시스템에서 {query} 구현 방법",
            f"{query} 관련 최신 연구 동향"
        ]
        
        queries.extend(variations[:2])  # 최대 3개 쿼리
        return queries
        
    def _generate_metadata_filters(self, query: str) -> Dict[str, Any]:
        """메타데이터 필터 생성"""
        filters = {}
        
        # 기술 카테고리 감지
        for category, terms in self.wms_keywords.items():
            for term in terms:
                if term.lower() in query.lower():
                    filters["category"] = category
                    break
                    
        # 날짜 필터 (최신 논문 우선)
        filters["date_priority"] = "recent_first"
        
        return filters

class SearchModule(BaseRAGModule):
    """
    Search 모듈 (Modular RAG)
    
    링크 참조 내용:
    - 임베딩 유사도 기반 검색 외에도 추가적인 검색 시나리오
    - 특정 시나리오에 맞춰 LLM이 생성한 코드나 SQL 등을 사용하여 검색
    - 다양한 데이터 소스 사용 가능
    """
    
    def __init__(self, faiss_index_path: str, embedding_model: str = "jhgan/ko-sroberta-multitask"):
        self.faiss_index_path = Path(faiss_index_path)
        self.embedding_model = embedding_model
        self._load_resources()
        
    def _load_resources(self):
        """검색 리소스 로드"""
        logger.info("Search Module: 리소스 로드")
        
        # Faiss 인덱스 로드
        index_file = self.faiss_index_path / "warehouse_automation_knowledge.index"
        if index_file.exists():
            self.faiss_index = faiss.read_index(str(index_file))
        else:
            self.faiss_index = None
            
        # 문서 및 메타데이터 로드
        docs_file = self.faiss_index_path / "documents.json"
        meta_file = self.faiss_index_path / "metadata.json"
        
        if docs_file.exists() and meta_file.exists():
            with open(docs_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            with open(meta_file, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
        else:
            self.documents = []
            self.metadatas = []
            
        # 임베딩 모델 로드
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def process(self, query: str, **kwargs) -> List[RAGResult]:
        """다중 검색 전략 실행"""
        logger.info("Search Module: 다중 검색 시작")
        
        multi_queries = kwargs.get('multi_queries', [query])
        metadata_filters = kwargs.get('metadata_filters', {})
        
        all_results = []
        
        # 1. 벡터 검색 (각 쿼리별)
        for q in multi_queries:
            vector_results = self._vector_search(q, top_k=15)
            all_results.extend(vector_results)
            
        # 2. 하이브리드 검색 (TF-IDF + 벡터)
        hybrid_results = self._hybrid_search(query, top_k=10)
        all_results.extend(hybrid_results)
        
        # 3. 메타데이터 필터링
        filtered_results = self._apply_metadata_filters(all_results, metadata_filters)
        
        # 4. 중복 제거 및 순위 조정
        final_results = self._deduplicate_and_rank(filtered_results)
        
        return final_results[:20]  # 최대 20개 결과
        
    def _vector_search(self, query: str, top_k: int = 15) -> List[RAGResult]:
        """벡터 유사도 검색"""
        if not self.faiss_index:
            return []
            
        query_vec = self.embedder.embed_query(query)
        query_vec = np.array([query_vec], dtype='float32')
        
        distances, indices = self.faiss_index.search(query_vec, top_k)
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.documents):
                similarity = 1 - (dist * dist) / 2  # L2 → cosine
                results.append(RAGResult(
                    content=self.documents[idx],
                    metadata=self.metadatas[idx],
                    similarity=similarity,
                    rank=i + 1,
                    source="vector_search"
                ))
                
        return results
        
    def _hybrid_search(self, query: str, top_k: int = 10) -> List[RAGResult]:
        """하이브리드 검색 (TF-IDF + Vector)"""
        if not self.documents:
            return []
            
        # TF-IDF 검색
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        doc_vectors = vectorizer.fit_transform(self.documents)
        query_vector = vectorizer.transform([query])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        # 상위 결과 선별
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0.1:  # 임계값 설정
                results.append(RAGResult(
                    content=self.documents[idx],
                    metadata=self.metadatas[idx],
                    similarity=similarities[idx],
                    rank=i + 1,
                    source="hybrid_search"
                ))
                
        return results
        
    def _apply_metadata_filters(self, results: List[RAGResult], filters: Dict[str, Any]) -> List[RAGResult]:
        """메타데이터 필터 적용"""
        if not filters:
            return results
            
        filtered = []
        for result in results:
            if self._matches_filters(result.metadata, filters):
                filtered.append(result)
                
        return filtered
        
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """필터 조건 확인"""
        for key, value in filters.items():
            if key == "category" and key in metadata:
                if metadata[key] != value:
                    return False
        return True
        
    def _deduplicate_and_rank(self, results: List[RAGResult]) -> List[RAGResult]:
        """중복 제거 및 순위 재조정"""
        # 내용 기반 중복 제거
        seen_contents = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.content[:100])  # 처음 100자로 중복 확인
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
                
        # 유사도 기준 정렬
        unique_results.sort(key=lambda x: x.similarity, reverse=True)
        
        # 순위 재조정
        for i, result in enumerate(unique_results):
            result.rank = i + 1
            
        return unique_results

class MemoryModule(BaseRAGModule):
    """
    Memory 모듈 (Modular RAG)
    
    링크 참조 내용:
    - 벡터 데이터베이스에서 검색된 청크뿐만 아니라 시스템 메모리에 저장된 이전 쿼리와 결합
    - 현재 입력과 가장 유사한 답변을 찾는 모듈
    """
    
    def __init__(self, memory_file: str = "_0_1_core/rag/memory_cache.json"):
        self.memory_file = Path(memory_file)
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_memory()
        
    def _load_memory(self):
        """메모리 캐시 로드"""
        if self.memory_file.exists():
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                self.memory = json.load(f)
        else:
            self.memory = {
                "queries": [],
                "responses": [],
                "contexts": [],
                "timestamps": []
            }
            
    def _save_memory(self):
        """메모리 캐시 저장"""
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memory, f, ensure_ascii=False, indent=2)
            
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """메모리 기반 컨텍스트 확장"""
        logger.info("Memory Module: 이전 쿼리 기반 컨텍스트 확장")
        
        # 1. 유사한 이전 쿼리 검색
        similar_queries = self._find_similar_queries(query)
        
        # 2. 관련 컨텍스트 추출
        relevant_contexts = self._extract_relevant_contexts(similar_queries)
        
        # 3. 현재 쿼리 저장
        self._store_current_query(query)
        
        return {
            "similar_queries": similar_queries,
            "relevant_contexts": relevant_contexts,
            "memory_enhanced": len(similar_queries) > 0
        }
        
    def _find_similar_queries(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """유사한 이전 쿼리 검색"""
        if not self.memory["queries"]:
            return []
            
        similar = []
        for i, prev_query in enumerate(self.memory["queries"]):
            similarity = self._calculate_similarity(query, prev_query)
            if similarity > threshold:
                similar.append({
                    "query": prev_query,
                    "response": self.memory["responses"][i],
                    "context": self.memory["contexts"][i],
                    "timestamp": self.memory["timestamps"][i],
                    "similarity": similarity
                })
                
        # 유사도 순 정렬
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:3]  # 최대 3개
        
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """쿼리 간 유사도 계산"""
        # 간단한 Jaccard 유사도 계산
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)
        
    def _extract_relevant_contexts(self, similar_queries: List[Dict[str, Any]]) -> List[str]:
        """관련 컨텍스트 추출"""
        contexts = []
        for query_info in similar_queries:
            if query_info["context"]:
                contexts.extend(query_info["context"][:2])  # 최대 2개 컨텍스트
        return contexts
        
    def _store_current_query(self, query: str):
        """현재 쿼리를 메모리에 저장"""
        self.memory["queries"].append(query)
        self.memory["responses"].append("")  # 응답은 나중에 업데이트
        self.memory["contexts"].append([])   # 컨텍스트는 나중에 업데이트
        self.memory["timestamps"].append(datetime.now().isoformat())
        
        # 최대 100개 쿼리만 유지
        if len(self.memory["queries"]) > 100:
            for key in self.memory:
                self.memory[key] = self.memory[key][-100:]
                
        self._save_memory()
        
    def update_response(self, query: str, response: str, context: List[str]):
        """응답과 컨텍스트 업데이트"""
        if self.memory["queries"] and self.memory["queries"][-1] == query:
            self.memory["responses"][-1] = response
            self.memory["contexts"][-1] = context
            self._save_memory()

class FusionModule(BaseRAGModule):
    """
    Fusion 모듈 (Modular RAG)
    
    링크 참조 내용:
    - 유저의 의도를 정확하게 반영하지 않을 수도 있다는 차관에서 비롯
    - LLM을 통해 유저의 쿼리로부터 여러 개의 가상 쿼리를 생성하여 검색하는 방식 (RAG-Fusion)
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """RAG-Fusion 실행"""
        logger.info("Fusion Module: RAG-Fusion 쿼리 생성")
        
        # 1. 다양한 관점의 쿼리 생성
        fusion_queries = self._generate_fusion_queries(query)
        
        # 2. 각 쿼리 검색 결과 수집 (외부에서 제공된 결과 사용)
        search_results = kwargs.get('search_results', [])
        
        # 3. Reciprocal Rank Fusion 적용
        fused_results = self._apply_reciprocal_rank_fusion(search_results, fusion_queries)
        
        return {
            "fusion_queries": fusion_queries,
            "fused_results": fused_results,
            "fusion_applied": True
        }
        
    def _generate_fusion_queries(self, original_query: str) -> List[str]:
        """RAG-Fusion용 다양한 쿼리 생성"""
        if not self.openai_api_key:
            # OpenAI가 없는 경우 규칙 기반 쿼리 생성
            return self._rule_based_query_generation(original_query)
            
        try:
            prompt = f"""
다음 쿼리에 대해 창고 자동화 및 산업용 로봇 관점에서 5개의 다양한 검색 쿼리를 생성해주세요.

원본 쿼리: {original_query}

생성 조건:
1. 각 쿼리는 다른 관점이나 측면을 다뤄야 함
2. 창고 자동화, AMR, AGV, WCS, WES 등 관련 용어 포함
3. 한 줄당 하나의 쿼리만 작성
4. 번호나 불필요한 설명 없이 쿼리만 작성

예시:
- 기술적 구현 관점
- 운영 효율성 관점  
- 비용 효과 관점
- 안전성 관점
- 최신 동향 관점
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content
            queries = [line.strip() for line in generated_text.split('\n') if line.strip() and not line.strip().startswith('-')]
            
            return queries[:5]  # 최대 5개
            
        except Exception as e:
            logger.warning(f"OpenAI 쿼리 생성 실패: {e}. 규칙 기반으로 대체")
            return self._rule_based_query_generation(original_query)
            
    def _rule_based_query_generation(self, query: str) -> List[str]:
        """규칙 기반 쿼리 생성 (OpenAI 대체)"""
        base_query = query.strip()
        
        fusion_queries = [
            f"{base_query}의 기술적 구현 방법",
            f"창고 자동화에서 {base_query} 적용 사례",
            f"{base_query} 관련 최신 연구 동향",
            f"산업용 로봇에서 {base_query}의 효율성",
            f"{base_query}와 WCS WES 시스템 연동"
        ]
        
        return fusion_queries
        
    def _apply_reciprocal_rank_fusion(self, search_results: List[List[RAGResult]], 
                                    fusion_queries: List[str], k: int = 60) -> List[RAGResult]:
        """Reciprocal Rank Fusion (RRF) 적용"""
        if not search_results:
            return []
            
        # 문서별 점수 집계
        doc_scores = {}
        
        for query_idx, results in enumerate(search_results):
            for result in results:
                doc_id = hash(result.content[:100])  # 문서 식별자
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        'result': result,
                        'rrf_score': 0,
                        'query_appearances': 0
                    }
                
                # RRF 점수 계산: 1 / (k + rank)
                rrf_score = 1.0 / (k + result.rank)
                doc_scores[doc_id]['rrf_score'] += rrf_score
                doc_scores[doc_id]['query_appearances'] += 1
                
        # RRF 점수로 정렬
        sorted_docs = sorted(doc_scores.items(), 
                           key=lambda x: x[1]['rrf_score'], 
                           reverse=True)
        
        # 최종 결과 생성
        fused_results = []
        for i, (doc_id, doc_info) in enumerate(sorted_docs[:20]):  # 상위 20개
            result = doc_info['result']
            result.rank = i + 1
            result.similarity = doc_info['rrf_score']
            result.source = "fusion"
            fused_results.append(result)
            
        return fused_results

class PostRetrievalModule(BaseRAGModule):
    """
    Post-Retrieval 모듈
    
    링크 참조 내용:
    - Reranking: 검색된 정보를 재순위하여 가장 관련성 높은 답변 우선시
    - Prompt Compression: 검색된 정보에 Noisy가 많을 수 있으므로 관련 없는 정보를 압축하고 길이 줄이기
    """
    
    def __init__(self, reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"):
        self.reranking_model_name = reranking_model
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(reranking_model)
        except ImportError:
            self.cross_encoder = None
            logger.warning("CrossEncoder 설치 필요. 기본 점수로 리랭킹")
            
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Post-Retrieval 처리"""
        logger.info("Post-Retrieval: 리랭킹 및 압축 시작")
        
        search_results = kwargs.get('search_results', [])
        if not search_results:
            return {"reranked_results": [], "compressed_context": ""}
            
        # 1. 크로스 인코더로 리랭킹
        reranked_results = self._rerank_results(query, search_results)
        
        # 2. 컨텍스트 압축
        compressed_context = self._compress_context(reranked_results[:10])  # 상위 10개만
        
        # 3. 노이즈 필터링
        filtered_results = self._filter_noise(reranked_results)
        
        return {
            "reranked_results": filtered_results,
            "compressed_context": compressed_context,
            "post_retrieval_applied": True
        }
        
    def _rerank_results(self, query: str, results: List[RAGResult]) -> List[RAGResult]:
        """크로스 인코더로 리랭킹"""
        if not self.cross_encoder or not results:
            return results
            
        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = [(query, result.content[:512]) for result in results]
            
            # 크로스 인코더 점수 계산
            scores = self.cross_encoder.predict(query_doc_pairs)
            
            # 점수 기반 정렬
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 리랭킹된 결과 생성
            reranked = []
            for i, (result, score) in enumerate(scored_results):
                result.rank = i + 1
                result.similarity = float(score)
                result.source = result.source + "_reranked"
                reranked.append(result)
                
            return reranked
            
        except Exception as e:
            logger.warning(f"리랭킹 실패: {e}. 원본 순서 유지")
            return results
            
    def _compress_context(self, results: List[RAGResult]) -> str:
        """컨텍스트 압축"""
        if not results:
            return ""
            
        # 중요한 문장만 추출
        compressed_parts = []
        
        for result in results[:5]:  # 상위 5개만
            content = result.content
            
            # 문장 분할
            sentences = content.split('.')
            
            # 중요 키워드 포함 문장 우선 선택
            important_keywords = ["AGV", "AMR", "WCS", "WES", "automation", "robot", "warehouse"]
            important_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in important_keywords):
                    important_sentences.append(sentence.strip())
                    
            # 최대 2문장까지
            if important_sentences:
                compressed_parts.append('. '.join(important_sentences[:2]))
            else:
                # 중요 키워드가 없으면 첫 문장
                if sentences and sentences[0].strip():
                    compressed_parts.append(sentences[0].strip())
                    
        return ' | '.join(compressed_parts)
        
    def _filter_noise(self, results: List[RAGResult], min_similarity: float = 0.3) -> List[RAGResult]:
        """노이즈 필터링"""
        filtered = []
        
        for result in results:
            # 최소 유사도 임계값
            if result.similarity >= min_similarity:
                # 너무 짧은 내용 제외
                if len(result.content.strip()) >= 50:
                    filtered.append(result)
                    
        return filtered


class AdvancedRAGOrchestrator:
    """
    고도화된 RAG 시스템 오케스트레이터
    
    모든 모듈을 통합하여 Advanced RAG + Modular RAG 실행
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 모듈 초기화
        self.pre_retrieval = PreRetrievalModule()
        self.search_module = SearchModule(config.get("faiss_index_path", "2_vecdb/faiss_storage"))
        self.memory_module = MemoryModule(config.get("memory_file", "_0_1_core/rag/memory_cache.json"))
        self.fusion_module = FusionModule(config.get("openai_api_key"))
        self.post_retrieval = PostRetrievalModule(config.get("reranking_model", "cross-encoder/ms-marco-MiniLM-L-2-v2"))
        
        logger.info("고도화된 RAG 오케스트레이터 초기화 완료")
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """전체 RAG 파이프라인 실행"""
        logger.info(f"고도화된 RAG 파이프라인 시작: {query}")
        
        start_time = datetime.now()
        
        try:
            # 1. Pre-Retrieval: 쿼리 최적화
            pre_result = self.pre_retrieval.process(query)
            logger.info(f"Pre-Retrieval 완료: {len(pre_result.get('multi_queries', []))}개 쿼리 생성")
            
            # 2. Memory Module: 이전 컨텍스트 활용
            memory_result = self.memory_module.process(query)
            logger.info(f"Memory Module 완료: {len(memory_result.get('similar_queries', []))}개 유사 쿼리 발견")
            
            # 3. Search Module: 다중 검색 전략
            search_results = self.search_module.process(
                query, 
                multi_queries=pre_result.get('multi_queries', [query]),
                metadata_filters=pre_result.get('metadata_filters', {})
            )
            logger.info(f"Search Module 완료: {len(search_results)}개 결과 검색")
            
            # 4. Fusion Module: RAG-Fusion 적용
            fusion_queries = self.fusion_module._generate_fusion_queries(query)
            
            # 각 Fusion 쿼리로 추가 검색
            fusion_search_results = []
            for fusion_query in fusion_queries:
                fusion_results = self.search_module.process(fusion_query, top_k=10)
                fusion_search_results.append(fusion_results)
                
            # RRF 적용
            fusion_result = self.fusion_module.process(
                query,
                search_results=fusion_search_results
            )
            logger.info(f"Fusion Module 완료: {len(fusion_result.get('fused_results', []))}개 융합 결과")
            
            # 5. 검색 결과 통합
            all_results = search_results + fusion_result.get('fused_results', [])
            
            # 6. Post-Retrieval: 리랭킹 및 압축
            post_result = self.post_retrieval.process(
                query,
                search_results=all_results
            )
            logger.info(f"Post-Retrieval 완료: {len(post_result.get('reranked_results', []))}개 최종 결과")
            
            # 7. 메모리 업데이트
            final_context = [r.content for r in post_result.get('reranked_results', [])[:5]]
            self.memory_module.update_response(query, "", final_context)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "query": query,
                "pre_retrieval": pre_result,
                "memory_result": memory_result,
                "search_results": search_results,
                "fusion_result": fusion_result,
                "final_results": post_result.get('reranked_results', []),
                "compressed_context": post_result.get('compressed_context', ''),
                "processing_time": processing_time,
                "total_results": len(post_result.get('reranked_results', [])),
                "pipeline_success": True
            }
            
        except Exception as e:
            logger.error(f"RAG 파이프라인 실행 실패: {e}")
            return {
                "query": query,
                "error": str(e),
                "pipeline_success": False
            }

if __name__ == "__main__":
    # 테스트 코드
    config = {
        "faiss_index_path": "2_vecdb/faiss_storage",
        "memory_file": "_0_1_core/rag/memory_cache.json",
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "reranking_model": "cross-encoder/ms-marco-MiniLM-L-2-v2"
    }
    
    orchestrator = AdvancedRAGOrchestrator(config)
    
    test_query = "AGV와 AMR의 차이점은 무엇인가?"
    result = orchestrator.process_query(test_query)
    
    print(f"Query: {test_query}")
    print(f"Results: {len(result.get('final_results', []))}")
    print(f"Processing Time: {result.get('processing_time', 0):.2f}초")
