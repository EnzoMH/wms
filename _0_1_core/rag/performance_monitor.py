#!/usr/bin/env python3
"""
RAG 성능 모니터링 모듈
======================

RAGAS 스타일의 성능 지표 수집 및 표시
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json

# NVIDIA GPU 모니터링
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[!] pynvml 설치 필요: pip install nvidia-ml-py")

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭 데이터 클래스"""
    # 시간 지표
    inference_time_ms: float  # 추론 시간 (밀리초)
    latency_ms: float  # 전체 지연 시간
    
    # 토큰 지표
    total_tokens: int  # 총 토큰 수
    input_tokens: int  # 입력 토큰
    output_tokens: int  # 출력 토큰
    tokens_per_second: float  # 초당 토큰 수
    
    # 메모리 지표
    memory_used_mb: float  # 사용된 RAM (MB)
    memory_percent: float  # RAM 사용률 (%)
    gpu_memory_used_mb: Optional[float] = None  # GPU 메모리 (MB)
    gpu_memory_percent: Optional[float] = None  # GPU 사용률 (%)
    
    # 효율성 지표
    energy_joules: Optional[float] = None  # 에너지 소비 (줄)
    flops_per_token: Optional[float] = None  # 토큰당 FLOPS
    throughput_tokens_sec: float = 0.0  # 처리량
    
    # 모델 정보
    model_name: str = "unknown"
    device: str = "cpu"


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self, model_name: str = "EXAONE-4.0-1.2B"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 초기 메모리 상태 저장
        self.initial_memory = psutil.virtual_memory().used / (1024 ** 2)
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            self.initial_gpu_memory = None
            
        # 성능 로그 저장 경로
        self.log_file = Path("_0_1_core/rag/performance_log.json")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # EXAONE Tokenizer 로드 (정확한 토큰 계산용)
        self._init_tokenizer()
        
        # NVIDIA GPU 모니터링 초기화
        self._init_gpu_monitoring()
    
    def _init_gpu_monitoring(self):
        """NVIDIA GPU 모니터링 초기화 (pynvml)"""
        self.gpu_handle = None
        self.gpu_power_samples = []
        
        if not PYNVML_AVAILABLE or not torch.cuda.is_available():
            return
        
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU 정보 출력
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self.gpu_handle) / 1000  # W
            
            logger.info(f"[OK] NVIDIA GPU 모니터링 활성화")
            logger.info(f"    GPU: {gpu_name}")
            logger.info(f"    전력 한도: {power_limit:.1f}W")
            
        except Exception as e:
            logger.warning(f"[!] GPU 모니터링 초기화 실패: {e}")
            self.gpu_handle = None
    
    def _init_tokenizer(self):
        """EXAONE Tokenizer 초기화"""
        try:
            from transformers import AutoTokenizer
            
            # EXAONE 공식 tokenizer 로드
            # Vocab Size: 102,400 (한국어/영어/스페인어 지원)
            # model_name이 이미 full path면 그대로 사용, 아니면 LGAI-EXAONE 접두사 추가
            tokenizer_name = self.model_name if "/" in self.model_name else f"LGAI-EXAONE/{self.model_name}"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )
            logger.info(f"[OK] EXAONE Tokenizer 로드 완료 (vocab_size: {self.tokenizer.vocab_size})")
            
        except Exception as e:
            logger.warning(f"[!] Tokenizer 로드 실패: {e}. 추정 방식 사용")
            self.tokenizer = None
        
    def start_monitoring(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used / (1024 ** 2)
        
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            torch.cuda.synchronize()  # GPU 동기화
        else:
            self.start_gpu_memory = None
        
        # GPU 전력 모니터링 시작 (pynvml)
        self.gpu_power_samples = []
        if self.gpu_handle:
            try:
                # 시작 전력 측정
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                self.gpu_power_samples.append(power_mw / 1000)  # W로 변환
            except Exception as e:
                logger.warning(f"[!] GPU 전력 측정 실패: {e}")
            
    def stop_monitoring(self, 
                       input_text: str = "", 
                       output_text: str = "") -> PerformanceMetrics:
        """
        모니터링 종료 및 메트릭 계산
        
        Args:
            input_text: 입력 텍스트 (토큰 계산용)
            output_text: 출력 텍스트 (토큰 계산용)
        
        Returns:
            PerformanceMetrics 객체
        """
        end_time = time.time()
        
        # 시간 계산
        elapsed_time = end_time - self.start_time
        inference_time_ms = elapsed_time * 1000
        latency_ms = inference_time_ms
        
        # 메모리 계산
        current_memory = psutil.virtual_memory().used / (1024 ** 2)
        memory_used_mb = current_memory - self.start_memory
        memory_percent = psutil.virtual_memory().percent
        
        # GPU 메모리
        gpu_memory_used_mb = None
        gpu_memory_percent = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_memory_used_mb = current_gpu_memory - self.start_gpu_memory
            gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100
        
        # 토큰 계산 (간단한 추정)
        input_tokens = self._estimate_tokens(input_text)
        output_tokens = self._estimate_tokens(output_text)
        total_tokens = input_tokens + output_tokens
        
        # 처리량 계산
        if elapsed_time > 0:
            tokens_per_second = output_tokens / elapsed_time
            throughput_tokens_sec = tokens_per_second
        else:
            tokens_per_second = 0.0
            throughput_tokens_sec = 0.0
        
        # 에너지 계산 (실제 GPU 전력 측정 또는 추정)
        energy_joules = None
        avg_power_watts = None
        
        if self.gpu_handle:
            try:
                # 종료 전력 측정
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                self.gpu_power_samples.append(power_mw / 1000)  # W로 변환
                
                # 평균 전력 계산 (시작 + 종료의 평균)
                if len(self.gpu_power_samples) >= 2:
                    avg_power_watts = sum(self.gpu_power_samples) / len(self.gpu_power_samples)
                    # 에너지 = 전력 * 시간 (Joules = Watts * seconds)
                    energy_joules = avg_power_watts * elapsed_time
                    logger.debug(f"[*] 실제 GPU 전력: {avg_power_watts:.2f}W, 에너지: {energy_joules:.2f}J")
                    
            except Exception as e:
                logger.warning(f"[!] GPU 전력 측정 실패: {e}")
        
        # 폴백: 추정 방식 (pynvml 없거나 실패 시)
        if energy_joules is None and torch.cuda.is_available() and gpu_memory_used_mb:
            # 간단한 추정: GPU 메모리 사용량 기반
            # RTX 4090: 약 450W, RTX 5090: 약 575W
            estimated_power_watts = 450 * (gpu_memory_percent / 100) if gpu_memory_percent else 200
            energy_joules = estimated_power_watts * elapsed_time
            logger.debug(f"[*] 추정 GPU 전력: {estimated_power_watts:.2f}W, 에너지: {energy_joules:.2f}J")
        
        # FLOPS 추정 (모델 크기 기반)
        flops_per_token = self._estimate_flops_per_token()
        
        metrics = PerformanceMetrics(
            inference_time_ms=inference_time_ms,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            memory_used_mb=memory_used_mb,
            memory_percent=memory_percent,
            gpu_memory_used_mb=gpu_memory_used_mb,
            gpu_memory_percent=gpu_memory_percent,
            energy_joules=energy_joules,
            flops_per_token=flops_per_token,
            throughput_tokens_sec=throughput_tokens_sec,
            model_name=self.model_name,
            device=self.device
        )
        
        # 로그 저장
        self._save_log(metrics)
        
        return metrics
    
    def _estimate_tokens(self, text: str) -> int:
        """
        토큰 수 계산 (EXAONE 공식 tokenizer 사용)
        
        EXAONE-4.0-1.2B tokenizer:
        - Vocab Size: 102,400
        - 한국어/영어/스페인어 지원
        - Hugging Face: https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B
        """
        if not text:
            return 0
        
        # EXAONE tokenizer 사용 (정확한 계산)
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.warning(f"[!] Tokenizer 사용 실패: {e}. 추정 방식 사용")
        
        # 폴백: 간단한 추정 방식
        korean_chars = len([c for c in text if '\uac00' <= c <= '\ud7a3'])
        english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
        
        estimated_tokens = int(korean_chars * 0.7 + english_words * 1.3)
        return max(estimated_tokens, 1)
    
    def _estimate_flops_per_token(self) -> float:
        """토큰당 FLOPS 추정"""
        # EXAONE-4.0-1.2B: 약 1.2B 파라미터
        # 1 토큰당 약 2 * params FLOPS (forward pass)
        if "1.2B" in self.model_name:
            params = 1.2e9
        elif "32B" in self.model_name:
            params = 32e9
        else:
            params = 1e9  # 기본값
        
        flops_per_token = 2 * params
        return flops_per_token
    
    def _save_log(self, metrics: PerformanceMetrics):
        """성능 로그 저장"""
        try:
            # 기존 로그 로드
            if self.log_file.exists():
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            # 새 로그 추가
            log_entry = asdict(metrics)
            log_entry['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            logs.append(log_entry)
            
            # 최근 100개만 유지
            logs = logs[-100:]
            
            # 저장
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"성능 로그 저장 실패: {e}")
    
    def get_average_metrics(self, last_n: int = 10) -> Dict[str, float]:
        """최근 N개 메트릭의 평균 계산"""
        try:
            if not self.log_file.exists():
                return {}
            
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            if not logs:
                return {}
            
            recent_logs = logs[-last_n:]
            
            # 평균 계산
            avg_metrics = {
                'avg_latency_ms': sum(log['latency_ms'] for log in recent_logs) / len(recent_logs),
                'avg_tokens_per_second': sum(log['tokens_per_second'] for log in recent_logs) / len(recent_logs),
                'avg_memory_mb': sum(log['memory_used_mb'] for log in recent_logs) / len(recent_logs),
                'total_queries': len(recent_logs)
            }
            
            if any(log.get('gpu_memory_used_mb') for log in recent_logs):
                gpu_logs = [log for log in recent_logs if log.get('gpu_memory_used_mb')]
                avg_metrics['avg_gpu_memory_mb'] = sum(log['gpu_memory_used_mb'] for log in gpu_logs) / len(gpu_logs)
            
            return avg_metrics
            
        except Exception as e:
            logger.warning(f"평균 메트릭 계산 실패: {e}")
            return {}


def format_metrics_display(metrics: PerformanceMetrics) -> str:
    """메트릭을 보기 좋게 포맷팅"""
    lines = [
        "\n" + "=" * 60,
        "성능 메트릭 (Performance Metrics)",
        "=" * 60,
        "",
        f"[시간 지표]",
        f"  추론 시간: {metrics.inference_time_ms:.2f} ms",
        f"  지연 시간: {metrics.latency_ms:.2f} ms",
        "",
        f"[토큰 지표]",
        f"  총 토큰: {metrics.total_tokens}",
        f"  입력 토큰: {metrics.input_tokens}",
        f"  출력 토큰: {metrics.output_tokens}",
        f"  처리량: {metrics.tokens_per_second:.2f} tokens/sec",
        "",
        f"[메모리 사용량]",
        f"  RAM: {metrics.memory_used_mb:.2f} MB ({metrics.memory_percent:.1f}%)",
    ]
    
    if metrics.gpu_memory_used_mb is not None:
        lines.extend([
            f"  GPU: {metrics.gpu_memory_used_mb:.2f} MB ({metrics.gpu_memory_percent:.1f}%)",
        ])
    
    if metrics.energy_joules is not None:
        # 평균 전력 계산 (역산)
        if hasattr(metrics, '_avg_power_watts') and metrics._avg_power_watts:
            avg_power = metrics._avg_power_watts
        else:
            # 대략적인 평균 전력 계산 (에너지 / 시간)
            avg_power = metrics.energy_joules / (metrics.latency_ms / 1000)
        
        lines.extend([
            "",
            f"[에너지 효율]",
            f"  평균 전력: {avg_power:.2f} W (실측)" if PYNVML_AVAILABLE else f"  평균 전력: {avg_power:.2f} W (추정)",
            f"  소비 에너지: {metrics.energy_joules:.2f} J ({metrics.energy_joules/1000:.4f} kJ)",
            f"  탄소 배출량: {metrics.energy_joules * 0.0005 / 1000:.6f} kg CO2 (한국 기준 약 0.5kg/kWh)",
        ])
    
    if metrics.flops_per_token is not None:
        lines.extend([
            "",
            f"[효율성 지표]",
            f"  토큰당 FLOPS: {metrics.flops_per_token:.2e}",
        ])
    
    lines.extend([
        "",
        f"[모델 정보]",
        f"  모델: {metrics.model_name}",
        f"  디바이스: {metrics.device}",
        "=" * 60,
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 테스트
    print("[*] EXAONE 성능 모니터링 테스트\n")
    
    monitor = PerformanceMonitor()
    
    # 토큰 계산 테스트
    print("=" * 60)
    print("토큰 계산 테스트 (EXAONE 공식 tokenizer)")
    print("=" * 60)
    
    test_texts = {
        "한국어": "AGV와 AMR의 차이점은 무엇인가요?",
        "영어": "What is the difference between AGV and AMR?",
        "혼합": "AGV는 정해진 경로를 따라 이동하는 자동화 운반차량입니다.",
    }
    
    for lang, text in test_texts.items():
        tokens = monitor._estimate_tokens(text)
        print(f"\n[{lang}] {text}")
        print(f"토큰 수: {tokens}")
        print(f"문자 수: {len(text)}")
        print(f"토큰/문자 비율: {tokens/len(text):.2f}")
    
    # 성능 모니터링 테스트
    print("\n\n" + "=" * 60)
    print("성능 모니터링 테스트")
    print("=" * 60 + "\n")
    
    monitor.start_monitoring()
    time.sleep(1)  # 시뮬레이션
    
    metrics = monitor.stop_monitoring(
        input_text="AGV와 AMR의 차이점은?",
        output_text="AGV는 정해진 경로를 따라 이동하는 자동화 운반차량이고, AMR은 자율주행 로봇입니다."
    )
    
    print(format_metrics_display(metrics))
    
    # 평균 메트릭
    avg = monitor.get_average_metrics()
    if avg:
        print("\n평균 메트릭:", avg)

