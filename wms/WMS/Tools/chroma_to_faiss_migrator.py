#!/usr/bin/env python3
"""
ChromaDB → Faiss 마이그레이션 도구
================================

WMS ChromaDB의 벡터 데이터를 Faiss 형식으로 변환하여
본 프로젝트의 Langchain Application에 통합할 수 있도록 합니다.

작성자: WMS 연구팀
날짜: 2024년 1월 15일
버전: 1.0.0
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import argparse

# 필수 라이브러리 임포트
try:
    import chromadb
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("설치: pip install chromadb faiss-cpu sentence-transformers pandas tqdm")
    exit(1)


class ChromaToFaissMigrator:
    """ChromaDB 데이터를 Faiss로 마이그레이션하는 클래스"""
    
    def __init__(self, 
                 chroma_db_path: str = "../VectorDB/chroma_storage",
                 output_dir: str = "./faiss_output",
                 collection_name: str = "wms_research_papers"):
        """
        마이그레이터 초기화
        
        Args:
            chroma_db_path: ChromaDB 저장 경로
            output_dir: Faiss 파일 출력 디렉토리
            collection_name: ChromaDB 컬렉션 이름
        """
        self.chroma_db_path = Path(chroma_db_path)
        self.output_dir = Path(output_dir)
        self.collection_name = collection_name
        
        self.setup_logging()
        self.setup_directories()
        self.setup_chromadb()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chroma_to_faiss_migration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_directories(self):
        """출력 디렉토리 생성"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"출력 디렉토리 준비: {self.output_dir}")
        
    def setup_chromadb(self):
        """ChromaDB 연결"""
        self.logger.info("ChromaDB 연결 중...")
        
        if not self.chroma_db_path.exists():
            raise FileNotFoundError(f"ChromaDB를 찾을 수 없습니다: {self.chroma_db_path}")
            
        try:
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_db_path))
            self.collection = self.chroma_client.get_collection(self.collection_name)
            
            count = self.collection.count()
            self.logger.info(f"✅ ChromaDB 연결 완료: {count}개 문서 발견")
            
            if count == 0:
                raise ValueError("ChromaDB에 데이터가 없습니다!")
                
        except Exception as e:
            self.logger.error(f"❌ ChromaDB 연결 실패: {e}")
            raise
            
    def extract_all_data(self) -> Tuple[np.ndarray, List[str], List[Dict]]:
        """ChromaDB에서 모든 벡터, 문서, 메타데이터 추출"""
        self.logger.info("ChromaDB에서 데이터 추출 중...")
        
        # 모든 데이터 가져오기
        total_count = self.collection.count()
        batch_size = 1000  # 메모리 효율성을 위한 배치 처리
        
        all_embeddings = []
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        self.logger.info(f"총 {total_count}개 문서를 {batch_size} 배치로 처리...")
        
        # 배치별로 데이터 추출
        for offset in tqdm(range(0, total_count, batch_size), desc="데이터 추출"):
            limit = min(batch_size, total_count - offset)
            
            # ChromaDB에서 배치 데이터 가져오기
            batch_data = self.collection.get(
                limit=limit,
                offset=offset,
                include=['embeddings', 'documents', 'metadatas']
            )
            
            # 데이터 누적
            if batch_data['embeddings']:
                all_embeddings.extend(batch_data['embeddings'])
            if batch_data['documents']:
                all_documents.extend(batch_data['documents'])
            if batch_data['metadatas']:
                all_metadatas.extend(batch_data['metadatas'])
            if batch_data['ids']:
                all_ids.extend(batch_data['ids'])
        
        # numpy 배열로 변환
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        
        self.logger.info(f"✅ 데이터 추출 완료:")
        self.logger.info(f"   - 벡터 수: {len(all_embeddings)}")
        self.logger.info(f"   - 벡터 차원: {embeddings_array.shape[1] if len(all_embeddings) > 0 else 0}")
        self.logger.info(f"   - 문서 수: {len(all_documents)}")
        self.logger.info(f"   - 메타데이터 수: {len(all_metadatas)}")
        
        return embeddings_array, all_documents, all_metadatas, all_ids
    
    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Faiss 인덱스 생성"""
        self.logger.info("Faiss 인덱스 생성 중...")
        
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        self.logger.info(f"벡터 차원: {dimension}, 벡터 수: {n_vectors}")
        
        # 인덱스 타입 결정 (벡터 수에 따라)
        if n_vectors < 10000:
            # 작은 데이터셋: Flat (정확한 검색)
            index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도)
            self.logger.info("Faiss IndexFlatIP 사용 (정확한 검색)")
        else:
            # 큰 데이터셋: IVF (빠른 근사 검색)
            nlist = min(100, n_vectors // 100)  # 클러스터 수
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.logger.info(f"Faiss IndexIVFFlat 사용 (nlist={nlist})")
            
            # 인덱스 훈련
            self.logger.info("인덱스 훈련 중...")
            index.train(embeddings)
        
        # 벡터 추가
        self.logger.info("벡터를 인덱스에 추가 중...")
        index.add(embeddings)
        
        self.logger.info(f"✅ Faiss 인덱스 생성 완료: {index.ntotal}개 벡터")
        return index
    
    def save_faiss_data(self, 
                       index: faiss.Index, 
                       documents: List[str], 
                       metadatas: List[Dict],
                       ids: List[str]):
        """Faiss 인덱스와 관련 데이터 저장"""
        self.logger.info("Faiss 데이터 저장 중...")
        
        # 1. Faiss 인덱스 저장
        index_file = self.output_dir / "wms_knowledge.index"
        faiss.write_index(index, str(index_file))
        self.logger.info(f"✅ Faiss 인덱스 저장: {index_file}")
        
        # 2. 문서 텍스트 저장
        documents_file = self.output_dir / "documents.json"
        with open(documents_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 문서 텍스트 저장: {documents_file}")
        
        # 3. 메타데이터 저장
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 메타데이터 저장: {metadata_file}")
        
        # 4. ID 매핑 저장
        ids_file = self.output_dir / "ids.json"
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ ID 매핑 저장: {ids_file}")
        
        # 5. 통합 정보 저장
        info = {
            "migration_date": datetime.now().isoformat(),
            "source_collection": self.collection_name,
            "total_vectors": index.ntotal,
            "vector_dimension": index.d,
            "index_type": type(index).__name__,
            "files": {
                "index": "wms_knowledge.index",
                "documents": "documents.json", 
                "metadata": "metadata.json",
                "ids": "ids.json"
            },
            "usage_example": {
                "python": """
# Faiss 인덱스 로드 예시
import faiss
import json

# 인덱스 로드
index = faiss.read_index('wms_knowledge.index')

# 문서와 메타데이터 로드
with open('documents.json', 'r', encoding='utf-8') as f:
    documents = json.load(f)
with open('metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# 검색 예시
query_vector = ...  # 쿼리 벡터
k = 5  # 상위 5개 결과
distances, indices = index.search(query_vector, k)
"""
            }
        }
        
        info_file = self.output_dir / "migration_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self.logger.info(f"✅ 마이그레이션 정보 저장: {info_file}")
        
    def test_faiss_search(self, index: faiss.Index, documents: List[str], test_query: str = "warehouse automation"):
        """Faiss 검색 테스트"""
        self.logger.info(f"🔍 Faiss 검색 테스트: '{test_query}'")
        
        try:
            # 임베딩 모델로 쿼리 벡터화 (ChromaDB와 동일한 모델 사용)
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            query_vector = model.encode([test_query], convert_to_numpy=True).astype(np.float32)
            
            # 검색 수행
            k = 5
            distances, indices = index.search(query_vector, k)
            
            self.logger.info(f"📋 검색 결과 (상위 {k}개):")
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(documents):
                    doc_preview = documents[idx][:200] + "..." if len(documents[idx]) > 200 else documents[idx]
                    self.logger.info(f"\n{i+1}. 유사도: {distance:.4f}")
                    self.logger.info(f"   문서: {doc_preview}")
                    
        except Exception as e:
            self.logger.error(f"❌ 검색 테스트 실패: {e}")
    
    def run_migration(self, test_search: bool = True):
        """전체 마이그레이션 프로세스 실행"""
        self.logger.info("🚀 ChromaDB → Faiss 마이그레이션 시작")
        
        try:
            # 1. 데이터 추출
            embeddings, documents, metadatas, ids = self.extract_all_data()
            
            # 2. Faiss 인덱스 생성
            index = self.create_faiss_index(embeddings)
            
            # 3. 데이터 저장
            self.save_faiss_data(index, documents, metadatas, ids)
            
            # 4. 검색 테스트 (선택사항)
            if test_search:
                self.test_faiss_search(index, documents)
            
            # 5. 마이그레이션 보고서 생성
            self.generate_migration_report(len(embeddings), embeddings.shape[1] if len(embeddings) > 0 else 0)
            
            self.logger.info("🎉 마이그레이션 완료!")
            
        except Exception as e:
            self.logger.error(f"❌ 마이그레이션 실패: {e}")
            raise
    
    def generate_migration_report(self, vector_count: int, dimension: int):
        """마이그레이션 보고서 생성"""
        report = f"""
WMS ChromaDB → Faiss 마이그레이션 보고서
=====================================

마이그레이션 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
소스: ChromaDB ({self.collection_name})
대상: Faiss Index

데이터 통계:
- 총 벡터 수: {vector_count:,}
- 벡터 차원: {dimension}
- 소스 경로: {self.chroma_db_path}
- 출력 경로: {self.output_dir}

생성된 파일:
- wms_knowledge.index: Faiss 벡터 인덱스
- documents.json: 원본 문서 텍스트
- metadata.json: 문서 메타데이터
- ids.json: 문서 ID 매핑
- migration_info.json: 마이그레이션 상세 정보

본 프로젝트 통합 방법:
1. wms_knowledge.index 파일을 본 프로젝트의 벡터DB 디렉토리로 복사
2. documents.json, metadata.json도 함께 복사
3. Langchain Application에서 Faiss 인덱스 로드 코드 수정
4. 기존 Faiss 인덱스를 WMS 전문지식 인덱스로 교체

다음 단계:
1. 본 프로젝트의 Langchain 코드에서 Faiss 로드 부분 확인
2. 벡터 검색 로직이 WMS 도메인에 맞게 작동하는지 테스트
3. 프롬프트 템플릿을 물류/WMS 전문 용어에 맞게 조정

주의사항:
- 임베딩 모델이 본 프로젝트와 일치하는지 확인 필요
- 벡터 차원이 호환되는지 검증 필요
- 검색 성능 최적화를 위한 인덱스 튜닝 고려
"""
        
        report_file = self.output_dir / "migration_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(report)
        self.logger.info(f"📄 마이그레이션 보고서 저장: {report_file}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="ChromaDB → Faiss 마이그레이션 도구")
    parser.add_argument("--chroma-db", default="../VectorDB/chroma_storage",
                       help="ChromaDB 저장 경로")
    parser.add_argument("--output-dir", default="./faiss_output",
                       help="Faiss 파일 출력 디렉토리")
    parser.add_argument("--collection", default="wms_research_papers",
                       help="ChromaDB 컬렉션 이름")
    parser.add_argument("--no-test", action="store_true",
                       help="검색 테스트 건너뛰기")
    
    args = parser.parse_args()
    
    # 마이그레이터 실행
    migrator = ChromaToFaissMigrator(
        chroma_db_path=args.chroma_db,
        output_dir=args.output_dir,
        collection_name=args.collection
    )
    
    migrator.run_migration(test_search=not args.no_test)


if __name__ == "__main__":
    main()
