#!/usr/bin/env python3
"""
Faiss 인덱스 내부 구조 분석 도구
================================

wms_knowledge.index 파일의 내부 구조를 분석합니다.
"""

import faiss
import numpy as np
import json
from pathlib import Path

def inspect_faiss_index():
    """Faiss 인덱스의 내부 구조를 분석합니다."""
    
    print(" Faiss 인덱스 내부 구조 분석")
    print("=" * 50)
    
    # 인덱스 로드
    index_path = "../VectorDB/faiss_storage/wms_knowledge.index"
    index = faiss.read_index(index_path)
    
    print(f" 기본 정보:")
    print(f"   - 인덱스 타입: {type(index).__name__}")
    print(f"   - 벡터 차원: {index.d}")
    print(f"   - 총 벡터 수: {index.ntotal}")
    print(f"   - 메트릭 타입: {index.metric_type}")
    
    # HNSW 특화 정보
    if hasattr(index, 'hnsw'):
        hnsw = index.hnsw
        print(f"\n HNSW 그래프 구조:")
        try:
            print(f"   - 연결 수 (M): {getattr(hnsw, 'M', '알 수 없음')}")
            print(f"   - 최대 연결 수: {getattr(hnsw, 'max_M', '알 수 없음')}")
            print(f"   - 구축 시 탐색 깊이: {getattr(hnsw, 'efConstruction', '알 수 없음')}")
            print(f"   - 레벨 수: {getattr(hnsw, 'max_level', -1) + 1}")
            print(f"   - 진입점 ID: {getattr(hnsw, 'entry_point', '알 수 없음')}")
            
            # 사용 가능한 속성들 확인
            attrs = [attr for attr in dir(hnsw) if not attr.startswith('_')]
            print(f"   - 사용 가능한 속성: {attrs[:5]}...")  # 처음 5개만
        except Exception as e:
            print(f"   - HNSW 정보 접근 오류: {e}")
    
    # 메모리 사용량 추정
    vector_memory = index.ntotal * index.d * 4  # float32 = 4 bytes
    print(f"\n 메모리 사용량:")
    print(f"   - 벡터 데이터: {vector_memory / 1024 / 1024:.1f} MB")
    print(f"   - 인덱스 파일: {Path(index_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    # 샘플 벡터 확인
    print(f"\n 샘플 벡터 (첫 번째 문서):")
    if index.ntotal > 0:
        # 첫 번째 벡터 추출
        sample_vector = index.reconstruct(0)
        print(f"   - 벡터 형태: {sample_vector.shape}")
        print(f"   - 값 범위: [{sample_vector.min():.3f}, {sample_vector.max():.3f}]")
        print(f"   - 평균: {sample_vector.mean():.3f}")
        print(f"   - 표준편차: {sample_vector.std():.3f}")
        print(f"   - 첫 10개 값: {sample_vector[:10]}")
    
    # 검색 성능 테스트
    print(f"\n 검색 성능 테스트:")
    if index.ntotal > 0:
        # 랜덤 쿼리 벡터 생성
        query_vector = np.random.randn(1, index.d).astype(np.float32)
        
        import time
        start_time = time.time()
        distances, indices = index.search(query_vector, 5)
        search_time = time.time() - start_time
        
        print(f"   - 5개 결과 검색 시간: {search_time * 1000:.2f} ms")
        print(f"   - 초당 검색 가능 횟수: {1/search_time:.0f} queries/sec")
        print(f"   - 반환된 인덱스: {indices[0]}")
        print(f"   - 거리 값: {distances[0]}")

def compare_with_linear_search():
    """선형 검색과 성능 비교"""
    print(f"\n 선형 검색 vs Faiss 성능 비교")
    print("=" * 50)
    
    # 설정 로드
    with open("../VectorDB/faiss_storage/config.json", 'r') as f:
        config = json.load(f)
    
    total_docs = config['total_documents']
    dimension = config['dimension']
    
    # 이론적 계산
    print(f" 이론적 성능 비교 (문서 {total_docs}개, 차원 {dimension}):")
    
    # 선형 검색: O(n)
    linear_ops = total_docs * dimension  # 모든 벡터와 내적 계산
    print(f"   - 선형 검색 연산 수: {linear_ops:,}")
    
    # HNSW 검색: O(log n)
    import math
    hnsw_ops = math.log2(total_docs) * 32 * dimension  # 대략적 추정
    print(f"   - HNSW 검색 연산 수: {hnsw_ops:,.0f}")
    
    speedup = linear_ops / hnsw_ops
    print(f"   - 이론적 속도 향상: {speedup:.1f}x")
    
    # 메모리 효율성
    print(f"\n 메모리 효율성:")
    raw_size = total_docs * dimension * 4  # float32
    compressed_size = Path("../VectorDB/faiss_storage/wms_knowledge.index").stat().st_size
    compression_ratio = raw_size / compressed_size
    
    print(f"   - 원본 벡터 크기: {raw_size / 1024 / 1024:.1f} MB")
    print(f"   - 압축된 인덱스: {compressed_size / 1024 / 1024:.1f} MB") 
    print(f"   - 압축률: {compression_ratio:.1f}x")

if __name__ == "__main__":
    try:
        inspect_faiss_index()
        compare_with_linear_search()
        
        print(f"\n 결론:")
        print(f"   - Faiss 인덱스는 벡터들의 '지도'입니다")
        print(f"   - HNSW는 벡터 공간의 '고속도로 네트워크'")
        print(f"   - RAG에서 관련 문서를 빠르게 찾는 핵심 엔진")
        print(f"   - 의미적 유사도 기반 검색 가능")
        
    except Exception as e:
        print(f" 오류 발생: {e}")
        import traceback
        traceback.print_exc()
