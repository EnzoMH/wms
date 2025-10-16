#!/usr/bin/env python3
"""
EXAONE-4.0-1.2B 래퍼 for Advanced RAG
=====================================

LG AI Research의 한국어 특화 모델을 RAG 시스템에 통합
"""

import json
import re
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# 성능 모니터링
try:
    from _0_1_core.rag.performance_monitor import PerformanceMonitor, format_metrics_display
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False
    print("[!] 성능 모니터링 비활성화 (psutil 설치 필요)")

logger = logging.getLogger(__name__)

class ExaoneRAGWrapper:
    """EXAONE-4.0-1.2B RAG 시스템 래퍼"""
    
    def __init__(self, 
                 model_name: str = "LGAI-EXAONE/EXAONE-4.0-1.2B",
                 use_vllm: bool = False,
                 use_ollama: bool = False,
                 ollama_model: str = "hf.co/LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF:Q4_K_M",
                 vllm_url: str = "http://localhost:8080/v1"):
        """
        Args:
            model_name: EXAONE 모델 이름
            use_vllm: vLLM 서버 사용 여부
            use_ollama: Ollama 사용 여부 (GPU 부담 적음, 권장!)
            ollama_model: Ollama 모델 이름
            vllm_url: vLLM 서버 URL
        """
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.vllm_url = vllm_url
        
        # 성능 모니터 초기화
        if MONITOR_AVAILABLE:
            self.monitor = PerformanceMonitor(model_name=model_name)
        else:
            self.monitor = None
        
        if use_ollama:
            self._init_ollama_client()
        elif use_vllm:
            self._init_vllm_client()
        else:
            self._init_transformers_model()
    
    def _init_ollama_client(self):
        """Ollama 클라이언트 초기화 (GPU 부담 적음, 권장!)"""
        try:
            from langchain_ollama import ChatOllama
            
            self.llm = ChatOllama(
                model=self.ollama_model,
                temperature=0.1,
                num_ctx=4096,  # Context window
            )
            
            logger.info(f"[OK] EXAONE Ollama 클라이언트 초기화 완료: {self.ollama_model}")
            logger.info("[*] Ollama는 CPU/GPU 자동 선택, 메모리 효율적")
            
            # 연결 테스트
            try:
                logger.info("[*] Ollama 연결 테스트 중...")
                test_response = self.llm.invoke("test")
                logger.info("[OK] Ollama 연결 확인 완료")
            except Exception as e:
                logger.warning(f"[!] Ollama 연결 테스트 실패: {e}")
                logger.warning(f"[!] 다음 명령으로 모델 다운로드: ollama run {self.ollama_model}")
            
        except ImportError:
            logger.error("[X] langchain-ollama 설치 필요: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"[X] Ollama 초기화 실패: {e}")
            raise
    
    def _init_vllm_client(self):
        """vLLM 클라이언트 초기화"""
        try:
            from langchain_openai import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key="EMPTY",
                openai_api_base=self.vllm_url,
                temperature=0.1,  # 한국어 권장 설정
                max_tokens=2048,
            )
            
            logger.info(f"[OK] EXAONE vLLM 클라이언트 초기화 완료: {self.vllm_url}")
            
            # 연결 테스트
            logger.info("[*] vLLM 서버 연결 테스트 중...")
            test_response = self.llm.invoke("test")
            logger.info("[OK] vLLM 서버 연결 확인 완료")
            
        except ImportError:
            logger.error("[X] langchain_openai 설치 필요: pip install langchain-openai")
            raise
        except Exception as e:
            logger.error(f"[X] vLLM 서버 연결 실패: {e}")
            raise
    
    def _init_transformers_model(self):
        """Transformers로 직접 모델 로드"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"[*] EXAONE 모델 로딩 중: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("[OK] EXAONE Transformers 모델 로딩 완료")
            
        except ImportError:
            logger.error("[X] transformers 설치 필요: pip install transformers>=4.54.0")
            raise
        except Exception as e:
            logger.error(f"[X] 모델 로딩 실패: {e}")
            raise
    
    def generate(self, 
                 prompt: str, 
                 temperature: float = 0.1,
                 max_tokens: int = 2048,
                 use_reasoning: bool = False) -> str:
        """
        텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            temperature: 생성 온도 (한국어는 0.1 권장)
            max_tokens: 최대 토큰 수
            use_reasoning: Reasoning 모드 사용 여부
        
        Returns:
            생성된 텍스트
        """
        if use_reasoning:
            prompt = self._wrap_reasoning(prompt)
        
        if self.use_ollama:
            return self._generate_ollama(prompt, temperature, max_tokens)
        elif self.use_vllm:
            return self._generate_vllm(prompt, temperature, max_tokens)
        else:
            return self._generate_transformers(prompt, temperature, max_tokens)
    
    def _wrap_reasoning(self, prompt: str) -> str:
        """Reasoning 모드 프롬프트 래핑"""
        return f"""<think>
Let me analyze this step by step for better accuracy...
</think>

{prompt}"""
    
    def _post_process_response(self, text: str) -> str:
        """
        응답 후처리 - 원치 않는 정체성 언급 제거
        
        EXAONE 모델이 학습된 정체성(LG AI Research, EXAONE 등)을 
        VisionSpace로 대체
        """
        replacements = {
            "EXAONE": "VisionSpace AI",
            "LG AI Research": "VisionSpace",
            "GPT-4": "VisionSpace AI",
            "GPT-3": "VisionSpace AI",
            "저는 LG AI Research에서 개발한": "저는 VisionSpace에서 개발한",
            "저는 LG AI에서 개발한": "저는 VisionSpace에서 개발한",
            "LG에서 개발한": "VisionSpace에서 개발한",
            "LG AI Research에서": "VisionSpace에서",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Ollama로 생성"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content
            
            # 1. <think> 태그 제거 (혹시 모를 reasoning 출력 방지)
            if '</think>' in content:
                # </think> 이후의 내용만 추출
                content = content.split('</think>')[-1].strip()
            
            # 2. 정체성 후처리 (EXAONE → VisionSpace AI)
            content = self._post_process_response(content)
            
            return content
            
        except Exception as e:
            logger.error(f"[X] Ollama 생성 실패: {e}")
            raise
    
    def _generate_vllm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """vLLM으로 생성"""
        try:
            response = self.llm.invoke(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content
            
        except Exception as e:
            logger.error(f"[X] vLLM 생성 실패: {e}")
            raise
    
    def _generate_transformers(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Transformers로 생성"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant specialized in warehouse automation and logistics."},
                {"role": "user", "content": prompt}
            ]
            
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.1,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2 if temperature > 0.5 else 1.0,
            )
            
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 제거
            if messages[-1]["content"] in full_text:
                response = full_text.split(messages[-1]["content"])[-1].strip()
            else:
                response = full_text
            
            return response
            
        except Exception as e:
            logger.error(f"[X] Transformers 생성 실패: {e}")
            raise
    
    def _is_identity_question(self, query: str) -> bool:
        """정체성 관련 질문인지 판단"""
        identity_keywords = [
            "이름", "누구", "넌 누구", "당신은 누구",
            "who are you", "what are you", "your name",
            "네 이름", "니 이름", "자기소개"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in identity_keywords)
    
    def query_rag(self, 
                  query: str, 
                  contexts: List[str],
                  temperature: float = 0.1,
                  show_metrics: bool = True) -> Dict[str, Any]:
        """
        RAG 쿼리 수행
        
        Args:
            query: 사용자 질문
            contexts: 검색된 컨텍스트 리스트
            temperature: 생성 온도
            show_metrics: 성능 메트릭 표시 여부
        
        Returns:
            응답 및 메타데이터 (성능 메트릭 포함)
        """
        # 정체성 관련 질문은 미리 정의된 답변 반환
        if self._is_identity_question(query):
            predefined_response = (
                "저는 VisionSpace에서 개발한 창고 자동화 및 물류 시스템 전문 AI 어시스턴트입니다. "
                "AGV, AMR, WMS, 재고 관리, 피킹 시스템 등 물류 자동화에 관한 질문에 답변해드릴 수 있습니다."
            )
            return {
                "query": query,
                "response": predefined_response,
                "contexts_used": 0,
                "model": self.model_name,
                "success": True,
                "performance": {}
            }
        
        # 성능 모니터링 시작
        if self.monitor:
            self.monitor.start_monitoring()
        
        # 컨텍스트 포맷팅
        context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""당신은 창고 자동화 및 물류 전문가입니다. 주어진 컨텍스트를 기반으로 질문에 답변하세요.

컨텍스트:
{context_text}

질문: {query}

답변 (컨텍스트 기반, 한국어로 명확하게):"""
        
        try:
            response = self.generate(
                prompt, 
                temperature=temperature,
                max_tokens=2048,
                use_reasoning=False
            )
            
            # 성능 모니터링 종료
            metrics = None
            metrics_dict = {}
            if self.monitor:
                metrics = self.monitor.stop_monitoring(
                    input_text=prompt,
                    output_text=response
                )
                metrics_dict = {
                    "inference_time_ms": metrics.inference_time_ms,
                    "latency_ms": metrics.latency_ms,
                    "total_tokens": metrics.total_tokens,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "tokens_per_second": metrics.tokens_per_second,
                    "memory_used_mb": metrics.memory_used_mb,
                    "memory_percent": metrics.memory_percent,
                    "gpu_memory_used_mb": metrics.gpu_memory_used_mb,
                    "gpu_memory_percent": metrics.gpu_memory_percent,
                    "energy_joules": metrics.energy_joules,
                    "flops_per_token": metrics.flops_per_token,
                    "throughput_tokens_sec": metrics.throughput_tokens_sec,
                    "device": metrics.device
                }
                
                if show_metrics:
                    print(format_metrics_display(metrics))
            
            return {
                "query": query,
                "response": response,
                "contexts_used": len(contexts),
                "model": self.model_name,
                "success": True,
                "performance": metrics_dict
            }
            
        except Exception as e:
            logger.error(f"[X] RAG 쿼리 실패: {e}")
            return {
                "query": query,
                "response": "",
                "error": str(e),
                "success": False,
                "performance": {}
            }
    
    def detect_language(self, text: str) -> str:
        """언어 감지 (한국어/영어)"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'\w', text))
        
        if total_chars == 0:
            return 'en'
        
        ko_ratio = korean_chars / total_chars
        return 'ko' if ko_ratio > 0.3 else 'en'


class BilingualRAGEvaluator:
    """한영 혼합 RAG 평가기"""
    
    def __init__(self, 
                 exaone_wrapper: Optional[ExaoneRAGWrapper] = None,
                 openai_api_key: Optional[str] = None):
        """
        Args:
            exaone_wrapper: EXAONE 래퍼 (한국어용)
            openai_api_key: OpenAI API 키 (영어 폴백용)
        """
        self.exaone = exaone_wrapper or ExaoneRAGWrapper(use_vllm=True)
        self.openai_api_key = openai_api_key
        
        if openai_api_key:
            self._init_openai_fallback()
    
    def _init_openai_fallback(self):
        """OpenAI 폴백 초기화"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain.llms.base import LLM
            
            self.openai_llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.openai_api_key,
                temperature=0.1
            )
            
            logger.info("[OK] OpenAI 폴백 초기화 완료")
            
        except Exception as e:
            logger.warning(f"[!] OpenAI 폴백 초기화 실패: {e}")
            self.openai_llm = None
    
    def query(self, query: str, contexts: List[str]) -> Dict[str, Any]:
        """언어 감지 후 최적 모델로 쿼리"""
        lang = self.exaone.detect_language(query)
        
        if lang == 'ko':
            logger.info("[*] 한국어 감지 - EXAONE 사용")
            return self.exaone.query_rag(query, contexts)
        else:
            logger.info("[*] 영어 감지 - EXAONE 사용 (영어도 지원)")
            # EXAONE은 영어도 잘 처리하므로 그대로 사용
            return self.exaone.query_rag(query, contexts)


# 헬퍼 함수
def create_exaone_rag_system(use_vllm: bool = True, 
                              vllm_url: str = "http://localhost:8080/v1") -> ExaoneRAGWrapper:
    """
    EXAONE RAG 시스템 생성
    
    Args:
        use_vllm: vLLM 사용 여부 (권장)
        vllm_url: vLLM 서버 URL
    
    Returns:
        ExaoneRAGWrapper 인스턴스
    """
    try:
        wrapper = ExaoneRAGWrapper(use_vllm=use_vllm, vllm_url=vllm_url)
        logger.info("[OK] EXAONE RAG 시스템 생성 완료")
        return wrapper
        
    except Exception as e:
        logger.error(f"[X] EXAONE RAG 시스템 생성 실패: {e}")
        raise


if __name__ == "__main__":
    # 테스트
    print("[*] EXAONE RAG Wrapper 테스트")
    
    # vLLM 사용 (권장)
    try:
        rag = ExaoneRAGWrapper(use_vllm=True)
        
        test_query = "AGV와 AMR의 차이점은?"
        test_contexts = [
            "AGV는 정해진 경로를 따라 이동하는 자동화 운반차량입니다.",
            "AMR은 자율주행이 가능한 모바일 로봇으로, 환경을 인식하고 경로를 스스로 계획합니다."
        ]
        
        result = rag.query_rag(test_query, test_contexts)
        
        print(f"\n[OK] 테스트 완료:")
        print(f"질문: {result['query']}")
        print(f"응답: {result['response']}")
        print(f"사용 컨텍스트: {result['contexts_used']}개")
        
    except Exception as e:
        print(f"[X] 테스트 실패: {e}")
        print("[!] vLLM 서버가 실행 중인지 확인하세요")
        print("[!] 실행 명령: vllm serve LGAI-EXAONE/EXAONE-4.0-1.2B --host 0.0.0.0 --port 8080")

