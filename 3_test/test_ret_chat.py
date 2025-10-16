#!/usr/bin/env python3
"""
EXAONE RAG 간단 대화 테스트
===========================

VectorDB (Faiss) + EXAONE-4.0-1.2B로 대화형 RAG 시스템
"""

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent))

# 필수 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# EXAONE Wrapper
from _0_1_core.rag.wrapper import ExaoneRAGWrapper

# 환경 변수 로드
load_dotenv()

class ExaoneRAGChat:
    """EXAONE RAG 대화 시스템"""
    
    def __init__(self, use_vllm: bool = False):
        """
        Args:
            use_vllm: True면 vLLM 서버 사용, False면 OpenAI 폴백
        """
        print("\n" + "="*70)
        print("EXAONE RAG 대화 시스템")
        print("="*70)
        
        self.use_vllm = use_vllm
        
        # 1. VectorDB 로드
        self.load_vectordb()
        
        # 2. 임베딩 모델 로드
        self.load_embedding()
        
        # 3. EXAONE 초기화
        self.init_exaone()
    
    def load_vectordb(self):
        """VectorDB 로드"""
        print("\n[1/3] VectorDB 로드 중...")
        
        db_path = Path("2_vecdb/faiss_storage")
        
        if not db_path.exists():
            print("[X] VectorDB가 없습니다!")
            print("    먼저 VectorDB를 구축하세요:")
            print("    python 0_core/vectorDB/faiss_builder.py --action build")
            sys.exit(1)
        
        # Faiss 인덱스
        index_file = db_path / "warehouse_automation_knowledge.index"
        if not index_file.exists():
            print(f"[X] 인덱스 파일 없음: {index_file}")
            sys.exit(1)
        
        self.index = faiss.read_index(str(index_file))
        
        # HNSW efSearch 증가 (검색 품질 향상)
        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = 200
        
        # 문서 & 메타데이터
        with open(db_path / "documents.json", 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        with open(db_path / "metadata.json", 'r', encoding='utf-8') as f:
            self.metadatas = json.load(f)
        
        print(f"[OK] VectorDB 로드 완료")
        print(f"    총 {len(self.documents):,}개 청크")
    
    def load_embedding(self):
        """임베딩 모델 로드"""
        print("\n[2/3] 임베딩 모델 로드 중...")
        
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.embedding = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            model_kwargs={'device': device, 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        print(f"[OK] 임베딩 모델 로드 완료 (device: {device})")
    
    def init_exaone(self):
        """EXAONE 초기화"""
        print("\n[3/3] EXAONE 초기화 중...")
        
        if self.use_vllm:
            # vLLM 서버 사용
            try:
                self.exaone = ExaoneRAGWrapper(
                    use_vllm=True,
                    vllm_url="http://localhost:8080/v1"
                )
                print("[OK] EXAONE vLLM 클라이언트 초기화 완료")
                print("    URL: http://localhost:8080/v1")
            except Exception as e:
                print(f"[X] vLLM 초기화 실패: {e}")
                print("[!] Transformers 모드로 폴백 시도...")
                self._init_transformers_fallback()
        else:
            # OpenAI 폴백
            self._init_openai_fallback()
    
    def _init_transformers_fallback(self):
        """Transformers 직접 사용 (GPU 폴백)"""
        try:
            print("[*] EXAONE Transformers 모드 로딩 중...")
            print("    (최초 실행 시 모델 다운로드: 약 2.5GB, 3-5분 소요)")
            
            self.exaone = ExaoneRAGWrapper(use_vllm=False)
            self.openai_client = None
            
            print("[OK] EXAONE Transformers 초기화 완료")
            print("    모드: Transformers (GPU 직접 사용, bfloat16)")
            
        except Exception as e:
            print(f"[X] Transformers 초기화 실패: {e}")
            print("[!] OpenAI로 최종 폴백합니다...")
            self._init_openai_fallback()
    
    def _init_openai_fallback(self):
        """OpenAI 폴백"""
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            print("[X] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다!")
            print("    .env 파일에 OPENAI_API_KEY를 추가하세요.")
            sys.exit(1)
        
        self.openai_client = OpenAI(api_key=api_key)
        self.exaone = None  # OpenAI 사용
        print("[OK] OpenAI 폴백 초기화 완료 (gpt-4o-mini)")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """VectorDB 검색"""
        # 쿼리 임베딩
        query_vec = self.embedding.embed_query(query)
        query_vec = np.array([query_vec], dtype='float32')
        
        # Faiss 검색
        distances, indices = self.index.search(query_vec, top_k)
        
        # 결과 정리
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.documents):
                # L2 거리 → 코사인 유사도 근사
                cosine_sim = 1 - (dist * dist) / 2
                
                results.append({
                    'rank': i + 1,
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'similarity': cosine_sim
                })
        
        return results
    
    def ask(self, query: str):
        """질문하기"""
        print("\n" + "="*70)
        print(f"질문: {query}")
        print("="*70)
        
        # 1. 검색
        print("\n[*] VectorDB 검색 중...")
        results = self.search(query, top_k=10)
        print(f"[OK] 검색 완료: {len(results)}개 결과")
        
        # 상위 3개 출력
        print("\n검색 결과 (상위 3개):")
        print("-"*70)
        for r in results[:3]:
            doc_type = '뉴스' if r['metadata'].get('document_type') == 'news_article' else '논문'
            print(f"\n[{r['rank']}] {doc_type} | 유사도: {r['similarity']:.3f}")
            print(f"    제목: {r['metadata'].get('paper_filename', 'Unknown')[:60]}")
            print(f"    내용: {r['content'][:150]}...")
        
        # 2. EXAONE 답변 생성
        print("\n[*] EXAONE 답변 생성 중...")
        
        # 컨텍스트 준비 (상위 5개)
        contexts = [r['content'] for r in results[:5]]
        
        if self.exaone:
            # EXAONE 사용
            result = self.exaone.query_rag(
                query=query,
                contexts=contexts,
                temperature=0.1
            )
            
            if result['success']:
                answer = result['response']
                print(f"[OK] 답변 생성 완료 (모델: {result['model']})")
            else:
                print(f"[X] 답변 생성 실패: {result.get('error')}")
                answer = "답변 생성에 실패했습니다."
        else:
            # OpenAI 폴백
            answer = self._generate_openai_answer(query, results[:5])
            print("[OK] 답변 생성 완료 (모델: gpt-4o-mini)")
        
        # 3. 답변 출력
        print("\n" + "="*70)
        print("답변:")
        print("="*70)
        print(answer)
        print("\n" + "="*70)
    
    def _generate_openai_answer(self, query: str, results: List[Dict]) -> str:
        """OpenAI로 답변 생성"""
        # 컨텍스트 포맷팅
        context_text = "\n\n".join([
            f"[{r['rank']}] {r['metadata'].get('paper_filename', 'Unknown')}\n{r['content'][:500]}"
            for r in results
        ])
        
        prompt = f"""당신은 창고 자동화 시스템(WMS) 전문 컨설턴트입니다.

질문: {query}

제공된 자료:
{context_text}

답변 규칙:
1. 제공된 자료 기반으로만 답변
2. 구체적인 수치가 있으면 인용
3. 한국어로 명확하게 설명
4. 출처 명시

답변:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 WMS 전문 컨설턴트입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[X] OpenAI 답변 생성 실패: {e}"
    
    def interactive(self):
        """대화형 모드"""
        print("\n" + "="*70)
        print("대화형 모드 시작")
        print("="*70)
        print("사용법:")
        print("  - 질문 입력 후 Enter")
        print("  - 'q' 입력 시 종료")
        print("  - 'help' 입력 시 도움말")
        print("")
        
        # 샘플 질문 제시
        sample_queries = [
            "AGV와 AMR의 차이점은?",
            "창고 자동화의 장점은?",
            "WMS 시스템의 핵심 기능은?",
            "RFID 기술이 창고에서 어떻게 활용되나요?",
        ]
        
        print("샘플 질문:")
        for i, q in enumerate(sample_queries, 1):
            print(f"  {i}. {q}")
        print("")
        
        while True:
            try:
                user_input = input("질문> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['q', 'quit', 'exit', '종료']:
                    print("\n[*] 종료합니다.")
                    break
                
                if user_input.lower() == 'help':
                    print("\n도움말:")
                    print("  - 자연어로 질문하세요")
                    print("  - 검색 결과는 Faiss VectorDB에서 가져옵니다")
                    print("  - 답변은 EXAONE 또는 OpenAI가 생성합니다")
                    print("")
                    continue
                
                # 숫자 입력 시 샘플 질문 사용
                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(sample_queries):
                        user_input = sample_queries[idx]
                        print(f"\n[*] 선택한 질문: {user_input}")
                    else:
                        print("[!] 잘못된 번호입니다.")
                        continue
                
                # 질문 처리
                self.ask(user_input)
                
            except KeyboardInterrupt:
                print("\n\n[*] 종료합니다.")
                break
            except Exception as e:
                print(f"\n[X] 오류 발생: {e}")
                continue


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="EXAONE RAG 대화 테스트")
    parser.add_argument("--vllm", action="store_true", 
                       help="vLLM 서버 사용 (기본: OpenAI 폴백)")
    parser.add_argument("--query", type=str, 
                       help="단일 질문 (대화형 모드 대신)")
    
    args = parser.parse_args()
    
    # RAG 시스템 초기화
    try:
        rag = ExaoneRAGChat(use_vllm=args.vllm)
    except Exception as e:
        print(f"\n[X] 초기화 실패: {e}")
        return
    
    # 단일 질문 모드 vs 대화형 모드
    if args.query:
        rag.ask(args.query)
    else:
        rag.interactive()


if __name__ == "__main__":
    main()

