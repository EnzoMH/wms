## 현재 시스템 분석

**강점**: 
- 300개 논문 + 10개 기술블로그의 풍부한 데이터
- ..\2_vecdb\faiss_storage 확인


**한계**: 
- Naive RAG 구조 (단순 검색-생성)
- 단일 형태 데이터만 처리
- 메타데이터 활용 부족

```Powershell
PS C:\Users\user\{...}\{...}\2_vecdb\faiss_storage> tree /f
C:.
    config.json
    documents.json
    metadata.json
    warehouse_automation_knowledge.index


에 하위 폴더가 없습니다.
PS C:\Users\user\{...}\{...}\2_vecdb\faiss_storage> 

```

## Modular RAG 구성 방안

### 1. **Enhanced Retrieval Module**

```python
# 하이브리드 검색 구현
class HybridRetriever:
    def __init__(self, faiss_index, bm25_retriever):
        self.dense_retriever = faiss_index  # 현재 FAISS
        self.sparse_retriever = bm25_retriever
        
    def retrieve(self, query, k=10):
        # Dense retrieval (현재 방식)
        dense_results = self.dense_retriever.similarity_search(query, k=k//2)
        
        # Sparse retrieval (키워드 기반)
        sparse_results = self.sparse_retriever.get_relevant_documents(query, k=k//2)
        
        # 결과 융합 및 re-ranking
        return self.fusion_rerank(dense_results, sparse_results)
```

### 2. **Query Enhancement Module**

```python
# 쿼리 재작성 및 확장
class QueryEnhancer:
    def __init__(self, llm):
        self.llm = llm
        
    def enhance_query(self, original_query):
        # HyDE: 가상 답변 생성
        hypothetical_answer = self.generate_hypothetical_answer(original_query)
        
        # Multi-query: 다양한 관점의 쿼리 생성
        related_queries = self.generate_related_queries(original_query)
        
        # Step-back prompting: 추상적 개념 추출
        abstract_query = self.generate_abstract_query(original_query)
        
        return {
            'original': original_query,
            'hypothetical': hypothetical_answer,
            'related': related_queries,
            'abstract': abstract_query
        }
```

### 3. **Metadata-Driven Filtering Module**

현재 metadata.json을 활용한 지능형 필터링:

```python
class MetadataFilter:
    def __init__(self, metadata_store):
        self.metadata = metadata_store
        
    def filter_by_context(self, query, documents):
        # 논문 vs 블로그 구분
        doc_type = self.classify_query_type(query)
        
        # 날짜/주제/저자 기반 필터링
        if "최신" in query or "recent" in query:
            return self.filter_by_date(documents, recent=True)
        
        # 특정 도메인 필터링
        domain = self.extract_domain(query)
        if domain:
            return self.filter_by_domain(documents, domain)
            
        return documents
```

### 4. **Adaptive Retrieval Module**

```python
class AdaptiveRetriever:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        
    def should_retrieve(self, query, current_context=""):
        # 현재 컨텍스트로 답변 가능한지 판단
        confidence = self.llm.estimate_confidence(query, current_context)
        
        return confidence < 0.7  # 임계값 이하시 검색 수행
        
    def iterative_retrieve(self, query, max_iterations=3):
        context = ""
        for i in range(max_iterations):
            if not self.should_retrieve(query, context):
                break
                
            new_docs = self.retriever.retrieve(query + context)
            context += self.process_documents(new_docs)
            
        return context
```

### 5. **Multi-Format Parser Module**

향후 확장을 위한 구조:

```python
class DocumentProcessor:
    def __init__(self):
        self.parsers = {
            'pdf': PDFParser(),
            'image': ImageParser(),  # OCR + Vision LLM
            'xml': XMLParser(),
            'json': JSONParser(),
            'markdown': MarkdownParser()
        }
        
    def process_document(self, file_path, doc_type):
        parser = self.parsers.get(doc_type)
        if not parser:
            raise ValueError(f"Unsupported document type: {doc_type}")
            
        # 통합된 메타데이터 형식으로 변환
        return parser.parse(file_path)
```

### 6. **Context Validation Module**

```python
class ContextValidator:
    def __init__(self, llm):
        self.llm = llm
        
    def validate_relevance(self, query, retrieved_docs):
        # 검색된 문서의 관련성 점수화
        relevance_scores = []
        
        for doc in retrieved_docs:
            score = self.llm.score_relevance(query, doc.page_content)
            relevance_scores.append(score)
            
        # 임계값 이하 문서 필터링
        filtered_docs = [
            doc for doc, score in zip(retrieved_docs, relevance_scores)
            if score > 0.6
        ]
        
        return filtered_docs
```

## 실제 구현 우선순위

### 단기 (1-2주)
1. **하이브리드 검색**: BM25 + 현재 FAISS 결합
2. **메타데이터 필터링**: 기존 metadata.json 활용한 스마트 필터링
3. **Query Enhancement**: HyDE와 Multi-query 구현

### 중기 (1개월)
1. **Adaptive Retrieval**: 상황별 검색 전략
2. **Context Validation**: 관련성 검증 모듈
3. **Re-ranking**: 검색 결과 재정렬

### 장기 (2-3개월)
1. **Multi-format Support**: PDF, Image 등 다양한 형태 지원
2. **Memory Module**: 대화 히스토리 기반 개인화
3. **Real-time Update**: 새로운 문서 실시간 인덱싱

## LangChain 통합 예시

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# 현재 시스템에 추가
bm25_retriever = BM25Retriever.from_documents(documents)
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # Dense 검색에 더 높은 가중치
)

# Modular RAG 체인 구성
rag_chain = (
    QueryEnhancer() 
    | MetadataFilter() 
    | ensemble_retriever 
    | ContextValidator() 
    | ChatOllama()
)
```
