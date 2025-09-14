#!/usr/bin/env python3
"""
벡터 차원 호환성 분석기
=====================

본 프로젝트와의 임베딩 모델 호환성을 확인하고
필요시 차원 변환을 수행하는 도구

작성자: WMS 연구팀
날짜: 2024년 1월 15일
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import openai
except ImportError as e:
    print(f"필수 패키지 설치 필요: {e}")
    exit(1)


class VectorDimensionAnalyzer:
    """벡터 차원 분석 및 변환 도구"""
    
    def __init__(self):
        self.setup_logging()
        self.available_models = self.get_available_embedding_models()
        
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_available_embedding_models(self) -> Dict[str, Dict]: # 768 차원, 1536 차원의 임베딩 모델 imprt ㅂ
        """사용 가능한 임베딩 모델들과 차원 정보"""
        return {
            'sentence_transformers': {
                'all-MiniLM-L6-v2': {'dimension': 384, 'speed': 'fast', 'quality': 'good'},
                'all-mpnet-base-v2': {'dimension': 768, 'speed': 'medium', 'quality': 'excellent'},
                'all-MiniLM-L12-v2': {'dimension': 384, 'speed': 'medium', 'quality': 'very_good'},
                'paraphrase-multilingual-MiniLM-L12-v2': {'dimension': 384, 'speed': 'medium', 'quality': 'good'},
                'distiluse-base-multilingual-cased': {'dimension': 512, 'speed': 'fast', 'quality': 'good'}
            },
            'openai': {
                'text-embedding-ada-002': {'dimension': 1536, 'speed': 'medium', 'quality': 'excellent'},
                'text-embedding-3-small': {'dimension': 1536, 'speed': 'fast', 'quality': 'very_good'},
                'text-embedding-3-large': {'dimension': 3072, 'speed': 'slow', 'quality': 'excellent'}
            },
            'huggingface': {
                'BAAI/bge-small-en-v1.5': {'dimension': 384, 'speed': 'fast', 'quality': 'good'},
                'BAAI/bge-base-en-v1.5': {'dimension': 768, 'speed': 'medium', 'quality': 'very_good'},
                'BAAI/bge-large-en-v1.5': {'dimension': 1024, 'speed': 'slow', 'quality': 'excellent'}
            }
        }
    
    def analyze_current_vectors(self, faiss_index_path: str) -> Dict:
        """현재 벡터 인덱스 분석"""
        self.logger.info(f"벡터 인덱스 분석 중: {faiss_index_path}")
        
        try:
            index = faiss.read_index(faiss_index_path)
            
            analysis = {
                'total_vectors': index.ntotal,
                'vector_dimension': index.d,
                'index_type': type(index).__name__,
                'is_trained': getattr(index, 'is_trained', True),
                'metric_type': 'Inner Product' if 'IP' in type(index).__name__ else 'L2'
            }
            
            # 샘플 벡터 추출하여 통계 분석
            if index.ntotal > 0:
                sample_size = min(100, index.ntotal)
                sample_vectors = np.zeros((sample_size, index.d), dtype=np.float32)
                
                # 샘플 벡터 추출
                for i in range(sample_size):
                    vector = index.reconstruct(i)
                    sample_vectors[i] = vector
                
                # 통계 계산
                analysis.update({
                    'vector_stats': {
                        'mean_norm': float(np.mean(np.linalg.norm(sample_vectors, axis=1))),
                        'std_norm': float(np.std(np.linalg.norm(sample_vectors, axis=1))),
                        'mean_values': sample_vectors.mean(axis=0).tolist()[:10],  # 처음 10개 차원만
                        'std_values': sample_vectors.std(axis=0).tolist()[:10]
                    }
                })
            
            self.logger.info(f"✅ 분석 완료: {analysis['total_vectors']}개 벡터, {analysis['vector_dimension']}차원")
            return analysis
            
        except Exception as e:
            self.logger.error(f"❌ 벡터 분석 실패: {e}")
            return {}
    
    def check_compatibility(self, current_dimension: int, target_model: str) -> Dict:
        """호환성 확인"""
        compatibility_info = {
            'is_compatible': False,
            'dimension_match': False,
            'recommended_action': '',
            'conversion_needed': False
        }
        
        # 타겟 모델의 차원 찾기
        target_dimension = None
        for provider, models in self.available_models.items():
            if target_model in models:
                target_dimension = models[target_model]['dimension']
                break
        
        if target_dimension is None:
            compatibility_info['recommended_action'] = f"알 수 없는 모델: {target_model}"
            return compatibility_info
        
        if current_dimension == target_dimension:
            compatibility_info.update({
                'is_compatible': True,
                'dimension_match': True,
                'recommended_action': '호환 가능 - 추가 작업 불필요'
            })
        else:
            compatibility_info.update({
                'conversion_needed': True,
                'current_dimension': current_dimension,
                'target_dimension': target_dimension,
                'recommended_action': f'차원 변환 필요: {current_dimension} → {target_dimension}'
            })
        
        return compatibility_info
    
    def convert_vector_dimensions(self, 
                                 source_vectors: np.ndarray, 
                                 target_dimension: int,
                                 method: str = 'pca') -> np.ndarray:
        """벡터 차원 변환"""
        self.logger.info(f"벡터 차원 변환: {source_vectors.shape[1]} → {target_dimension}")
        
        if source_vectors.shape[1] == target_dimension:
            self.logger.info("차원이 이미 일치함")
            return source_vectors
        
        if method == 'pca':
            if source_vectors.shape[1] > target_dimension:
                # 차원 축소
                pca = PCA(n_components=target_dimension)
                converted_vectors = pca.fit_transform(source_vectors)
                self.logger.info(f"PCA 차원 축소 완료 (설명 분산: {pca.explained_variance_ratio_.sum():.3f})")
            else:
                # 차원 확장 (제로 패딩)
                padding_size = target_dimension - source_vectors.shape[1]
                padding = np.zeros((source_vectors.shape[0], padding_size), dtype=source_vectors.dtype)
                converted_vectors = np.hstack([source_vectors, padding])
                self.logger.info(f"제로 패딩으로 차원 확장 완료")
                
        elif method == 'truncate':
            if source_vectors.shape[1] > target_dimension:
                # 단순 절단
                converted_vectors = source_vectors[:, :target_dimension]
                self.logger.info("벡터 절단으로 차원 축소 완료")
            else:
                # 제로 패딩
                padding_size = target_dimension - source_vectors.shape[1]
                padding = np.zeros((source_vectors.shape[0], padding_size), dtype=source_vectors.dtype)
                converted_vectors = np.hstack([source_vectors, padding])
                self.logger.info("제로 패딩으로 차원 확장 완료")
        
        return converted_vectors.astype(np.float32)
    
    def create_compatible_index(self, 
                               original_index_path: str,
                               target_model: str,
                               output_path: str,
                               conversion_method: str = 'pca') -> bool:
        """호환 가능한 인덱스 생성"""
        self.logger.info(f"호환 인덱스 생성: {target_model}")
        
        try:
            # 원본 인덱스 로드
            original_index = faiss.read_index(original_index_path)
            
            # 타겟 차원 확인
            target_dimension = None
            for provider, models in self.available_models.items():
                if target_model in models:
                    target_dimension = models[target_model]['dimension']
                    break
            
            if target_dimension is None:
                raise ValueError(f"알 수 없는 모델: {target_model}")
            
            # 벡터 추출
            n_vectors = original_index.ntotal
            original_vectors = np.zeros((n_vectors, original_index.d), dtype=np.float32)
            
            for i in range(n_vectors):
                original_vectors[i] = original_index.reconstruct(i)
            
            # 차원 변환
            converted_vectors = self.convert_vector_dimensions(
                original_vectors, target_dimension, conversion_method
            )
            
            # 새 인덱스 생성
            if n_vectors < 10000:
                new_index = faiss.IndexFlatIP(target_dimension)
            else:
                nlist = min(100, n_vectors // 100)
                quantizer = faiss.IndexFlatIP(target_dimension)
                new_index = faiss.IndexIVFFlat(quantizer, target_dimension, nlist)
                new_index.train(converted_vectors)
            
            # 벡터 추가
            new_index.add(converted_vectors)
            
            # 저장
            faiss.write_index(new_index, output_path)
            
            self.logger.info(f"✅ 호환 인덱스 생성 완료: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 호환 인덱스 생성 실패: {e}")
            return False
    
    def recommend_optimal_model(self, use_case: str = 'general') -> Dict:
        """최적 모델 추천"""
        recommendations = {
            'general': {
                'model': 'sentence-transformers/all-mpnet-base-v2',
                'reason': '범용성과 성능의 균형',
                'dimension': 768
            },
            'speed_priority': {
                'model': 'sentence-transformers/all-MiniLM-L6-v2', 
                'reason': '빠른 속도, 적은 메모리 사용',
                'dimension': 384
            },
            'quality_priority': {
                'model': 'text-embedding-ada-002',
                'reason': '최고 품질, 상용 서비스',
                'dimension': 1536
            },
            'multilingual': {
                'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                'reason': '다국어 지원',
                'dimension': 384
            }
        }
        
        return recommendations.get(use_case, recommendations['general'])


def main():
    """테스트 실행"""
    analyzer = VectorDimensionAnalyzer()
    
    print("🔍 사용 가능한 임베딩 모델:")
    for provider, models in analyzer.available_models.items():
        print(f"\n📦 {provider.upper()}:")
        for model, info in models.items():
            print(f"  - {model}: {info['dimension']}차원 ({info['quality']} 품질, {info['speed']} 속도)")
    
    print("\n💡 추천 모델:")
    for use_case in ['general', 'speed_priority', 'quality_priority']:
        rec = analyzer.recommend_optimal_model(use_case)
        print(f"  - {use_case}: {rec['model']} ({rec['dimension']}차원) - {rec['reason']}")


if __name__ == "__main__":
    main()
