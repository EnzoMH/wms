#!/usr/bin/env python3
"""
Faiss Builder 테스트 스크립트
===========================

WMS Faiss Builder의 기본 기능을 테스트합니다.

사용법:
    python test_faiss_builder.py
"""

import sys
from pathlib import Path
import logging

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

try:
    from faiss_builder import WMSFaissBuilder
except ImportError as e:
    print(f"❌ faiss_builder.py를 가져올 수 없습니다: {e}")
    sys.exit(1)


def test_faiss_builder():
    """Faiss Builder 기본 테스트"""
    print("🚀 WMS Faiss Builder 테스트 시작\n")
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 1. Faiss Builder 초기화
        print("1️⃣ Faiss Builder 초기화...")
        builder = WMSFaissBuilder(
            processed_data_dir="../ProcessedData",
            vector_db_dir="../VectorDB",
            embedding_model="korean_specialized"
        )
        print("✅ 초기화 완료\n")
        
        # 2. 데이터베이스 통계 확인
        print("2️⃣ 기존 데이터베이스 통계 확인...")
        builder.get_database_stats()
        print()
        
        # 3. 검색 테스트
        print("3️⃣ 벡터 검색 테스트...")
        
        test_queries = [
            "창고 자동화 시스템",
            "로봇 피킹 기술",
            "AMR 경로 계획",
            "재고 관리 최적화",
            "AGV 제어 시스템"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 테스트 쿼리 {i}: '{query}'")
            results = builder.test_search(query=query, top_k=3)
            
            if results:
                print(f"   ✅ {len(results['results'])} 개 결과 반환됨")
                for j, result in enumerate(results['results'][:2], 1):  # 상위 2개만 표시
                    print(f"   {j}. {result['metadata']['paper_filename'][:40]}... "
                          f"(유사도: {result['similarity']:.3f})")
            else:
                print("   ❌ 검색 결과 없음")
        
        print("\n4️⃣ 성능 통계...")
        builder.get_database_stats()
        
        print("\n🎉 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def interactive_search():
    """대화형 검색 모드"""
    print("🔍 대화형 검색 모드 시작")
    print("종료하려면 'quit' 또는 'q'를 입력하세요.\n")
    
    try:
        builder = WMSFaissBuilder(
            processed_data_dir="../ProcessedData",
            vector_db_dir="../VectorDB"
        )
        
        # 기존 인덱스 로드
        if not builder.load_existing_index():
            print("❌ 기존 인덱스를 찾을 수 없습니다. 먼저 build를 실행해주세요.")
            return
        
        while True:
            query = input("\n🔍 검색어를 입력하세요: ").strip()
            
            if query.lower() in ['quit', 'q', '종료']:
                print("👋 검색을 종료합니다.")
                break
            
            if not query:
                continue
            
            print(f"\n'{query}' 검색 중...")
            results = builder.test_search(query=query, top_k=5)
            
            if results and results['results']:
                print(f"\n📋 {len(results['results'])} 개 결과:")
                for i, result in enumerate(results['results'], 1):
                    metadata = result['metadata']
                    content = result['document'][:150] + "..."
                    
                    print(f"\n{i}. 📄 {metadata['paper_filename']}")
                    print(f"   🎯 유사도: {result['similarity']:.3f}")
                    print(f"   📝 청크 #{metadata['chunk_id']}")
                    print(f"   📃 내용: {content}")
            else:
                print("❌ 관련 문서를 찾을 수 없습니다.")
    
    except KeyboardInterrupt:
        print("\n👋 검색을 종료합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def main():
    """메인 함수"""
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_search()
    else:
        test_faiss_builder()


if __name__ == "__main__":
    main()



