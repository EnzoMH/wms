#!/usr/bin/env python3
"""
시스템 서비스
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import random
import logging

logger = logging.getLogger(__name__)


class SystemService:
    """시스템 모니터링 서비스"""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """시스템 리소스 통계"""
        try:
            import psutil
            
            # RAM 정보
            ram = psutil.virtual_memory()
            ram_used = ram.used / (1024**3)
            ram_total = ram.total / (1024**3)
            ram_percent = ram.percent
            
            # CPU 정보
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            disk_used = disk.used / (1024**3)
            disk_total = disk.total / (1024**3)
            
            # GPU 정보 (있으면)
            gpu_memory_used = 0
            gpu_memory_total = 0
            gpu_memory_percent = 0
            
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100 if gpu_memory_total > 0 else 0
            except:
                pass
            
            return {
                "gpu_memory_used": round(gpu_memory_used, 2),
                "gpu_memory_total": round(gpu_memory_total, 2),
                "gpu_memory_percent": round(gpu_memory_percent, 2),
                "ram_used": round(ram_used, 2),
                "ram_total": round(ram_total, 2),
                "ram_percent": round(ram_percent, 2),
                "cpu_percent": round(cpu_percent, 2),
                "disk_used": round(disk_used, 2),
                "disk_total": round(disk_total, 2)
            }
        except Exception as e:
            logger.error(f"시스템 통계 수집 실패: {e}")
            return {
                "gpu_memory_used": 0,
                "gpu_memory_total": 0,
                "gpu_memory_percent": 0,
                "ram_used": 0,
                "ram_total": 0,
                "ram_percent": 0,
                "cpu_percent": 0,
                "disk_used": 0,
                "disk_total": 0
            }
    
    @staticmethod
    def get_recent_activities(limit: int = 10) -> List[Dict[str, Any]]:
        """최근 활동 로그 (더미 데이터, 추후 DB 연동)"""
        # TODO: 실제 활동 로그를 DB나 파일에서 읽어오도록 구현
        activities = [
            {
                "message": "벡터DB 증분 업데이트 완료",
                "type": "success",
                "details": {"vectors_added": 50}
            },
            {
                "message": "뉴스 크롤링 완료",
                "type": "info",
                "details": {"articles": 10}
            },
            {
                "message": "청크 최적화 완료",
                "type": "success",
                "details": {"chunks": 465}
            }
        ]
        
        # 타임스탬프 추가
        for i, activity in enumerate(activities):
            activity["timestamp"] = (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat()
        
        return activities[:limit]

