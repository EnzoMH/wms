#!/usr/bin/env python3
"""
고도화된 WMS 전문 키워드 및 용어 정의
=====================================

SmartFactory, AMR, AGV, CNV, RTV 등 산업용 협동로봇과 
고도화된 물류 자동화 시스템의 전문 용어들을 포함

작성자: WMS 연구팀  
날짜: 2024년 1월 15일
"""

class EnhancedWMSKeywords:
    """고도화된 WMS 전문 키워드 관리 클래스"""
    
    def __init__(self):
        self.setup_advanced_keywords()
        self.setup_professional_terms()
        self.setup_search_combinations()
    
    def setup_advanced_keywords(self):
        """고도화된 검색 키워드 설정"""
        self.advanced_search_keywords = [
            # 핵심 로봇 기술
            "AMR autonomous mobile robot warehouse",
            "AGV automated guided vehicle logistics", 
            "CNV conveyor system automation",
            "RTV return to vendor process",
            "collaborative robot cobot warehouse",
            "palletizing robot automation",
            "sortation robot system",
            
            # 스마트팩토리 통합
            "smart factory warehouse integration",
            "Industry 4.0 warehouse management",
            "digital twin warehouse simulation",
            "cyber physical system logistics",
            "IoT sensor network warehouse",
            
            # 고급 제어 시스템
            "WCS warehouse control system",
            "WES warehouse execution system", 
            "MES manufacturing execution system warehouse",
            "SCADA warehouse automation",
            "PLC programmable logic controller warehouse",
            
            # 첨단 피킹 기술
            "pick to light system optimization",
            "put to wall sorting system",
            "voice picking technology",
            "vision guided picking robot",
            "goods to person automation",
            "person to goods optimization",
            
            # 자동창고 시스템
            "AS/RS automated storage retrieval system",
            "VLM vertical lift module",
            "carousel storage system",
            "shuttle system warehouse",
            "miniload automated storage",
            
            # 최적화 알고리즘
            "slotting optimization algorithm",
            "wave planning optimization",
            "batch picking optimization",
            "route optimization warehouse",
            "inventory placement optimization",
            
            # 성능 지표
            "warehouse KPI dashboard",
            "throughput optimization metrics",
            "order fulfillment accuracy",
            "labor productivity measurement",
            "equipment utilization rate",
            
            # 통합 시스템
            "ERP WMS integration",
            "TMS transportation management integration",
            "OMS order management system",
            "YMS yard management system",
            "LMS labor management system"
        ]
    
    def setup_professional_terms(self):
        """전문 용어 분류 체계"""
        self.professional_terms = {
            'robot_systems': {
                'mobile_robots': ['AMR', 'AGV', 'autonomous mobile robot', 'automated guided vehicle'],
                'stationary_robots': ['palletizer', 'depalletizer', 'sortation robot', 'pick and place robot'],
                'collaborative_robots': ['cobot', 'collaborative robot', 'human robot collaboration'],
                'conveyor_systems': ['CNV', 'conveyor belt', 'sortation conveyor', 'merge conveyor']
            },
            
            'control_systems': {
                'warehouse_control': ['WCS', 'warehouse control system', 'material flow control'],
                'execution_systems': ['WES', 'warehouse execution system', 'task management'],
                'manufacturing_integration': ['MES', 'manufacturing execution system', 'production planning'],
                'enterprise_systems': ['ERP integration', 'SAP WM', 'Oracle WMS']
            },
            
            'picking_technologies': {
                'light_systems': ['pick to light', 'put to light', 'put to wall', 'light directed picking'],
                'voice_systems': ['voice picking', 'voice directed picking', 'speech recognition'],
                'vision_systems': ['vision guided picking', 'computer vision', 'barcode scanning', 'RFID'],
                'automation_levels': ['goods to person', 'person to goods', 'fully automated picking']
            },
            
            'storage_systems': {
                'automated_storage': ['AS/RS', 'automated storage retrieval', 'miniload', 'unit load'],
                'vertical_systems': ['VLM', 'vertical lift module', 'vertical carousel', 'tower storage'],
                'horizontal_systems': ['horizontal carousel', 'shuttle system', 'flow rack'],
                'high_density': ['drive in rack', 'push back rack', 'pallet flow rack']
            },
            
            'optimization_algorithms': {
                'inventory_optimization': ['slotting optimization', 'ABC analysis', 'velocity based slotting'],
                'order_optimization': ['wave planning', 'batch optimization', 'zone picking'],
                'routing_optimization': ['shortest path', 'traveling salesman', 'route optimization'],
                'resource_optimization': ['labor scheduling', 'equipment allocation', 'capacity planning']
            },
            
            'smart_factory_integration': {
                'industry_4_0': ['Industry 4.0', 'smart factory', 'digital transformation'],
                'digital_technologies': ['digital twin', 'cyber physical system', 'IoT integration'],
                'data_analytics': ['predictive analytics', 'machine learning warehouse', 'AI optimization'],
                'connectivity': ['5G warehouse', 'edge computing', 'cloud integration']
            },
            
            'performance_metrics': {
                'operational_kpis': ['throughput rate', 'order accuracy', 'cycle time', 'fill rate'],
                'efficiency_metrics': ['labor productivity', 'equipment utilization', 'space utilization'],
                'quality_metrics': ['picking accuracy', 'damage rate', 'inventory accuracy'],
                'cost_metrics': ['cost per shipment', 'labor cost ratio', 'automation ROI']
            },
            
            'process_optimization': {
                'inbound_processes': ['receiving optimization', 'putaway strategy', 'cross docking'],
                'storage_processes': ['inventory management', 'cycle counting', 'replenishment'],
                'outbound_processes': ['order picking', 'packing optimization', 'shipping consolidation'],
                'returns_processing': ['RTV', 'return to vendor', 'reverse logistics', 'returns processing']
            }
        }
    
    def setup_search_combinations(self):
        """검색 조합 패턴"""
        self.search_combinations = {
            'technology_combinations': [
                "AMR AGV warehouse automation comparison",
                "collaborative robot warehouse safety",
                "IoT sensor AMR fleet management", 
                "digital twin warehouse simulation AMR",
                "machine learning AGV path optimization"
            ],
            
            'integration_patterns': [
                "WCS WES integration architecture",
                "ERP WMS real time synchronization",
                "MES warehouse execution integration",
                "smart factory warehouse connectivity"
            ],
            
            'optimization_patterns': [
                "slotting optimization machine learning",
                "wave planning genetic algorithm",
                "AMR task allocation optimization",
                "multi objective warehouse optimization"
            ]
        }
    
    def get_all_search_keywords(self):
        """모든 검색 키워드 반환"""
        all_keywords = []
        
        # 고급 검색 키워드
        all_keywords.extend(self.advanced_search_keywords)
        
        # 기술 조합 키워드
        all_keywords.extend(self.search_combinations['technology_combinations'])
        all_keywords.extend(self.search_combinations['integration_patterns'])
        all_keywords.extend(self.search_combinations['optimization_patterns'])
        
        return all_keywords
    
    def get_professional_term_list(self):
        """전문 용어 리스트 반환"""
        all_terms = []
        
        for category in self.professional_terms.values():
            for subcategory in category.values():
                all_terms.extend(subcategory)
        
        return list(set(all_terms))  # 중복 제거
    
    def get_category_terms(self, category: str, subcategory: str = None):
        """특정 카테고리의 용어 반환"""
        if category not in self.professional_terms:
            return []
        
        if subcategory:
            return self.professional_terms[category].get(subcategory, [])
        else:
            all_terms = []
            for subcat in self.professional_terms[category].values():
                all_terms.extend(subcat)
            return all_terms


# 사용 예시
if __name__ == "__main__":
    keywords = EnhancedWMSKeywords()
    
    print("🔍 고도화된 검색 키워드 (일부):")
    for i, keyword in enumerate(keywords.get_all_search_keywords()[:10], 1):
        print(f"{i:2d}. {keyword}")
    
    print(f"\n📊 총 검색 키워드 수: {len(keywords.get_all_search_keywords())}")
    print(f"📊 총 전문 용어 수: {len(keywords.get_professional_term_list())}")
    
    print("\n🤖 로봇 시스템 용어:")
    robot_terms = keywords.get_category_terms('robot_systems')
    print(f"   {', '.join(robot_terms[:10])}...")
    
    print("\n🏭 스마트팩토리 용어:")
    smart_factory_terms = keywords.get_category_terms('smart_factory_integration')
    print(f"   {', '.join(smart_factory_terms[:10])}...")
