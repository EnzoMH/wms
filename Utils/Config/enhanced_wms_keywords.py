#!/usr/bin/env python3
"""
고도화된 WMS 키워드 관리자
==========================

창고관리시스템(WMS)과 산업용 협동로봇 분야의 전문 키워드와 용어를 체계적으로 관리합니다.
논문 수집, 텍스트 처리, 벡터 검색에 최적화된 키워드 세트를 제공합니다.

작성자: AI Assistant
날짜: 2025년 9월 15일
버전: 1.0.0
"""

import json
import logging
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re


class EnhancedWMSKeywords:
    """WMS와 산업용 협동로봇 분야의 고도화된 키워드 관리 클래스"""
    
    def __init__(self):
        """키워드 관리자 초기화"""
        self.setup_logging()
        self.initialize_keywords()
        self.logger.info("🔍 Enhanced WMS Keywords 시스템 초기화 완료")
    
    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_keywords(self):
        """전문 키워드 세트 초기화"""
        
        # 1. 핵심 WMS 키워드
        self.core_wms_keywords = {
            "warehouse_management": [
                "warehouse management system", "WMS", "warehouse optimization",
                "inventory management", "inventory control", "inventory optimization",
                "warehouse automation", "automated warehouse", "smart warehouse",
                "warehouse operations", "warehouse efficiency", "warehouse productivity"
            ],
            "storage_systems": [
                "automated storage retrieval system", "AS/RS", "ASRS",
                "automated storage", "storage optimization", "rack optimization",
                "dynamic storage", "intelligent storage", "flexible storage"
            ],
            "material_handling": [
                "material handling", "material flow", "material transport",
                "goods movement", "cargo handling", "freight handling",
                "warehouse logistics", "intralogistics", "internal logistics"
            ]
        }
        
        # 2. 산업용 협동로봇 키워드
        self.collaborative_robot_keywords = {
            "cobot_general": [
                "collaborative robot", "cobot", "cobots", "collaborative robotics",
                "human-robot collaboration", "HRC", "human-robot interaction",
                "safe robotics", "collaborative automation"
            ],
            "industrial_robots": [
                "industrial robot", "industrial robotics", "manufacturing robot",
                "production robot", "assembly robot", "pick and place robot",
                "articulated robot", "SCARA robot", "delta robot"
            ],
            "robot_applications": [
                "robotic picking", "robotic sorting", "robotic packing",
                "robotic palletizing", "robotic depalletizing",
                "bin picking", "piece picking", "order picking robot"
            ]
        }
        
        # 3. 자동화 및 IoT 키워드
        self.automation_iot_keywords = {
            "automation": [
                "warehouse automation", "logistics automation", "supply chain automation",
                "automated guided vehicle", "AGV", "autonomous mobile robot", "AMR",
                "automated conveyor", "robotic automation", "process automation"
            ],
            "iot_sensors": [
                "IoT warehouse", "warehouse IoT", "sensor networks",
                "RFID tracking", "barcode scanning", "vision systems",
                "warehouse sensors", "smart sensors", "connected warehouse"
            ],
            "ai_ml": [
                "warehouse AI", "artificial intelligence warehouse",
                "machine learning logistics", "predictive analytics warehouse",
                "computer vision warehouse", "deep learning robotics",
                "AI optimization", "intelligent warehouse systems"
            ]
        }
        
        # 4. 물류 및 공급망 키워드
        self.logistics_supply_chain_keywords = {
            "logistics": [
                "warehouse logistics", "logistics optimization", "distribution center",
                "fulfillment center", "cross docking", "logistics planning",
                "warehouse planning", "logistics management"
            ],
            "supply_chain": [
                "supply chain management", "supply chain optimization",
                "supply chain automation", "supply chain visibility",
                "demand forecasting", "inventory forecasting",
                "supply chain analytics", "logistics network"
            ],
            "e_commerce": [
                "e-commerce fulfillment", "online order fulfillment",
                "omnichannel fulfillment", "last mile delivery",
                "order processing", "order management", "fulfillment automation"
            ]
        }
        
        # 5. 성능 및 최적화 키워드
        self.performance_optimization_keywords = {
            "optimization": [
                "warehouse optimization", "layout optimization", "route optimization",
                "path planning", "task scheduling", "resource allocation",
                "throughput optimization", "efficiency improvement"
            ],
            "performance_metrics": [
                "warehouse KPI", "warehouse metrics", "operational efficiency",
                "productivity measurement", "throughput analysis",
                "cycle time", "order accuracy", "inventory accuracy"
            ],
            "algorithms": [
                "optimization algorithm", "scheduling algorithm", "routing algorithm",
                "genetic algorithm", "simulated annealing", "particle swarm optimization",
                "reinforcement learning", "heuristic optimization"
            ]
        }
        
        # 6. 기술 및 시스템 통합 키워드
        self.technology_integration_keywords = {
            "wms_integration": [
                "WMS integration", "ERP integration", "system integration",
                "warehouse management software", "warehouse control system", "WCS",
                "warehouse execution system", "WES", "middleware integration"
            ],
            "digital_technologies": [
                "digital twin warehouse", "digital transformation",
                "Industry 4.0 warehouse", "cyber-physical systems",
                "blockchain logistics", "cloud warehouse management"
            ],
            "data_analytics": [
                "warehouse analytics", "big data warehouse", "real-time analytics",
                "predictive maintenance", "warehouse intelligence",
                "data-driven warehouse", "business intelligence warehouse"
            ]
        }
        
        # 모든 키워드를 통합
        self.all_keywords = self._combine_all_keywords()
        
        # 검색 최적화용 키워드 생성
        self.search_optimized_keywords = self._create_search_optimized_keywords()
        
        self.logger.info(f"📚 총 {len(self.all_keywords)}개의 전문 키워드 로드 완료")
    
    def _combine_all_keywords(self) -> List[str]:
        """모든 카테고리의 키워드를 하나의 리스트로 통합"""
        all_keywords = []
        
        keyword_categories = [
            self.core_wms_keywords,
            self.collaborative_robot_keywords,
            self.automation_iot_keywords,
            self.logistics_supply_chain_keywords,
            self.performance_optimization_keywords,
            self.technology_integration_keywords
        ]
        
        for category in keyword_categories:
            for subcategory, keywords in category.items():
                all_keywords.extend(keywords)
        
        # 중복 제거 및 정렬
        return sorted(list(set(all_keywords)))
    
    def _create_search_optimized_keywords(self) -> Dict[str, List[str]]:
        """검색 엔진별 최적화된 키워드 생성"""
        return {
            "arxiv": [
                "warehouse management system optimization",
                "collaborative robot warehouse automation",
                "automated storage retrieval system",
                "AGV warehouse navigation",
                "robotic picking system",
                "warehouse layout optimization",
                "inventory management algorithm",
                "supply chain automation robotics"
            ],
            "semantic_scholar": [
                "warehouse management system",
                "collaborative robotics",
                "automated guided vehicle",
                "warehouse automation",
                "inventory optimization",
                "robotic material handling",
                "smart warehouse IoT",
                "logistics automation"
            ],
            "ieee": [
                "warehouse management system",
                "collaborative robot safety",
                "automated storage system",
                "warehouse robotics",
                "inventory control system",
                "logistics automation",
                "human-robot collaboration",
                "warehouse optimization algorithm"
            ],
            "google_scholar": [
                "warehouse management system WMS",
                "collaborative robot cobot warehouse",
                "automated warehouse system",
                "robotic warehouse automation",
                "smart warehouse management",
                "warehouse robotics applications",
                "inventory management optimization",
                "logistics automation systems"
            ]
        }
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """카테고리별 키워드 반환"""
        category_map = {
            "core_wms": self.core_wms_keywords,
            "collaborative_robot": self.collaborative_robot_keywords,
            "automation_iot": self.automation_iot_keywords,
            "logistics_supply_chain": self.logistics_supply_chain_keywords,
            "performance_optimization": self.performance_optimization_keywords,
            "technology_integration": self.technology_integration_keywords
        }
        
        if category in category_map:
            keywords = []
            for subcategory, keyword_list in category_map[category].items():
                keywords.extend(keyword_list)
            return keywords
        else:
            self.logger.warning(f"알 수 없는 카테고리: {category}")
            return []
    
    def get_search_keywords(self, source: str = "all", limit: int = 20) -> List[str]:
        """검색 소스별 최적화된 키워드 반환"""
        if source == "all":
            return self.all_keywords[:limit]
        elif source in self.search_optimized_keywords:
            return self.search_optimized_keywords[source][:limit]
        else:
            self.logger.warning(f"지원하지 않는 검색 소스: {source}")
            return self.all_keywords[:limit]
    
    def get_high_priority_keywords(self, count: int = 15) -> List[str]:
        """높은 우선순위 키워드 반환 (논문 수집 시 사용)"""
        high_priority = [
            "warehouse management system",
            "collaborative robot",
            "warehouse automation", 
            "automated storage retrieval",
            "AGV warehouse",
            "robotic picking",
            "warehouse optimization",
            "inventory management system",
            "warehouse robotics",
            "smart warehouse",
            "automated guided vehicle",
            "warehouse IoT",
            "supply chain automation",
            "material handling robot",
            "warehouse efficiency optimization"
        ]
        return high_priority[:count]
    
    def expand_keyword(self, base_keyword: str) -> List[str]:
        """기본 키워드를 확장하여 관련 키워드들 생성"""
        expansion_map = {
            "warehouse": ["storage", "depot", "facility", "center"],
            "robot": ["robotic", "robotics", "automation", "automated"],
            "management": ["control", "optimization", "planning", "coordination"],
            "system": ["technology", "solution", "platform", "framework"],
            "automated": ["automatic", "autonomous", "smart", "intelligent"],
            "optimization": ["improvement", "enhancement", "efficiency", "performance"]
        }
        
        expanded_keywords = [base_keyword]
        
        for word, synonyms in expansion_map.items():
            if word in base_keyword.lower():
                for synonym in synonyms:
                    expanded_keyword = base_keyword.lower().replace(word, synonym)
                    if expanded_keyword != base_keyword.lower():
                        expanded_keywords.append(expanded_keyword)
        
        return expanded_keywords
    
    def filter_keywords_by_relevance(self, text: str, threshold: float = 0.7) -> List[str]:
        """텍스트 내용과 관련성이 높은 키워드 필터링"""
        text_lower = text.lower()
        relevant_keywords = []
        
        for keyword in self.all_keywords:
            # 키워드가 텍스트에 직접 포함되어 있는 경우
            if keyword.lower() in text_lower:
                relevant_keywords.append(keyword)
                continue
            
            # 키워드의 주요 단어들이 텍스트에 포함되어 있는 경우
            keyword_words = keyword.lower().split()
            if len(keyword_words) > 1:
                word_matches = sum(1 for word in keyword_words if word in text_lower)
                if word_matches / len(keyword_words) >= threshold:
                    relevant_keywords.append(keyword)
        
        return relevant_keywords
    
    def get_keyword_variations(self, keyword: str) -> List[str]:
        """키워드의 다양한 변형들 생성"""
        variations = [keyword]
        
        # 복수형/단수형 변환
        if keyword.endswith('s') and len(keyword) > 3:
            variations.append(keyword[:-1])
        else:
            variations.append(keyword + 's')
        
        # 하이픈 처리
        if '-' in keyword:
            variations.append(keyword.replace('-', ' '))
            variations.append(keyword.replace('-', ''))
        
        # 줄임말 확장
        abbreviation_map = {
            "WMS": "warehouse management system",
            "AGV": "automated guided vehicle", 
            "AMR": "autonomous mobile robot",
            "ASRS": "automated storage retrieval system",
            "IoT": "internet of things",
            "AI": "artificial intelligence",
            "ML": "machine learning"
        }
        
        for abbr, full_form in abbreviation_map.items():
            if abbr.lower() in keyword.lower():
                variations.append(keyword.lower().replace(abbr.lower(), full_form))
            if full_form in keyword.lower():
                variations.append(keyword.replace(full_form, abbr))
        
        return list(set(variations))  # 중복 제거
    
    def generate_search_queries(self, max_queries: int = 10) -> List[str]:
        """논문 검색에 최적화된 쿼리 생성"""
        base_queries = [
            "warehouse management system optimization",
            "collaborative robot warehouse automation", 
            "automated storage retrieval system design",
            "AGV path planning warehouse",
            "robotic picking system warehouse",
            "warehouse layout optimization algorithm",
            "inventory management system automation",
            "smart warehouse IoT integration",
            "warehouse robotics material handling",
            "supply chain automation warehouse",
            "warehouse efficiency improvement",
            "automated warehouse control system",
            "warehouse management system integration",
            "collaborative robotics safety warehouse",
            "warehouse automation technology"
        ]
        
        return base_queries[:max_queries]
    
    def save_keywords_to_file(self, filepath: str = "enhanced_wms_keywords.json"):
        """키워드 정보를 JSON 파일로 저장"""
        keyword_data = {
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "total_keywords": len(self.all_keywords),
                "categories": list(self.core_wms_keywords.keys()) + 
                            list(self.collaborative_robot_keywords.keys()) +
                            list(self.automation_iot_keywords.keys()) +
                            list(self.logistics_supply_chain_keywords.keys()) +
                            list(self.performance_optimization_keywords.keys()) +
                            list(self.technology_integration_keywords.keys())
            },
            "keywords": {
                "core_wms": self.core_wms_keywords,
                "collaborative_robot": self.collaborative_robot_keywords,
                "automation_iot": self.automation_iot_keywords,
                "logistics_supply_chain": self.logistics_supply_chain_keywords,
                "performance_optimization": self.performance_optimization_keywords,
                "technology_integration": self.technology_integration_keywords
            },
            "search_optimized": self.search_optimized_keywords,
            "all_keywords": self.all_keywords
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(keyword_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"💾 키워드 데이터 저장 완료: {filepath}")
        except Exception as e:
            self.logger.error(f"키워드 저장 실패: {e}")
    
    def load_keywords_from_file(self, filepath: str) -> bool:
        """JSON 파일에서 키워드 정보 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keyword_data = json.load(f)
            
            if "keywords" in keyword_data:
                self.core_wms_keywords = keyword_data["keywords"].get("core_wms", {})
                self.collaborative_robot_keywords = keyword_data["keywords"].get("collaborative_robot", {})
                self.automation_iot_keywords = keyword_data["keywords"].get("automation_iot", {})
                self.logistics_supply_chain_keywords = keyword_data["keywords"].get("logistics_supply_chain", {})
                self.performance_optimization_keywords = keyword_data["keywords"].get("performance_optimization", {})
                self.technology_integration_keywords = keyword_data["keywords"].get("technology_integration", {})
                
                self.all_keywords = keyword_data.get("all_keywords", [])
                self.search_optimized_keywords = keyword_data.get("search_optimized", {})
                
                self.logger.info(f"📂 키워드 데이터 로드 완료: {filepath}")
                return True
            
        except FileNotFoundError:
            self.logger.warning(f"키워드 파일을 찾을 수 없음: {filepath}")
        except Exception as e:
            self.logger.error(f"키워드 로드 실패: {e}")
        
        return False
    
    def get_statistics(self) -> Dict[str, int]:
        """키워드 통계 정보 반환"""
        stats = {
            "total_keywords": len(self.all_keywords),
            "core_wms": sum(len(keywords) for keywords in self.core_wms_keywords.values()),
            "collaborative_robot": sum(len(keywords) for keywords in self.collaborative_robot_keywords.values()),
            "automation_iot": sum(len(keywords) for keywords in self.automation_iot_keywords.values()),
            "logistics_supply_chain": sum(len(keywords) for keywords in self.logistics_supply_chain_keywords.values()),
            "performance_optimization": sum(len(keywords) for keywords in self.performance_optimization_keywords.values()),
            "technology_integration": sum(len(keywords) for keywords in self.technology_integration_keywords.values())
        }
        
        return stats
    
    def print_summary(self):
        """키워드 시스템 요약 정보 출력"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("🔍 Enhanced WMS Keywords 시스템 요약")
        print("="*60)
        print(f"📚 전체 키워드 수: {stats['total_keywords']}")
        print("\n📊 카테고리별 키워드 수:")
        print(f"  🏭 핵심 WMS: {stats['core_wms']}")
        print(f"  🤖 협동로봇: {stats['collaborative_robot']}")
        print(f"  ⚡ 자동화/IoT: {stats['automation_iot']}")
        print(f"  📦 물류/공급망: {stats['logistics_supply_chain']}")
        print(f"  📈 성능/최적화: {stats['performance_optimization']}")
        print(f"  🔧 기술/통합: {stats['technology_integration']}")
        
        print(f"\n🎯 우선순위 키워드 (상위 5개):")
        for i, keyword in enumerate(self.get_high_priority_keywords(5), 1):
            print(f"  {i}. {keyword}")
        
        print("="*60)


def main():
    """테스트 및 데모 실행"""
    # Enhanced WMS Keywords 시스템 초기화
    keywords = EnhancedWMSKeywords()
    
    # 시스템 요약 출력
    keywords.print_summary()
    
    # 키워드 저장
    keywords.save_keywords_to_file("enhanced_wms_keywords.json")
    
    # 검색 최적화 키워드 테스트
    print("\n🔍 ArXiv 최적화 키워드:")
    for keyword in keywords.get_search_keywords("arxiv", 5):
        print(f"  - {keyword}")
    
    # 키워드 확장 테스트
    print("\n🔄 키워드 확장 예시:")
    base_keyword = "warehouse robot"
    expanded = keywords.expand_keyword(base_keyword)
    print(f"  기본: {base_keyword}")
    print(f"  확장: {', '.join(expanded[:3])}")


if __name__ == "__main__":
    main()
