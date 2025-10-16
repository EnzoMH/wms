#!/usr/bin/env python3
"""
크롤링 대상 사이트 및 키워드 설정
====================================

Supply Chain, Logistics, SmartFactory 관련 뉴스 소스 및 검색 키워드 관리
"""

# RSS 피드 소스 (crawlable_site.md 기반 대폭 확장)
RSS_FEEDS = {
    # === 기존 핵심 사이트 ===
    'supply_chain_dive': 'https://www.supplychaindive.com/feeds/news/',
    'logistics_mgmt': 'https://www.logisticsmgmt.com/rss/topic/3-warehouse-dc',
    'mhl_news': 'https://www.mhlnews.com/rss-feeds',
    'dc_velocity': 'https://www.dcvelocity.com/rss/articles',
    
    # === 글로벌 물류 미디어 ===
    'inbound_logistics': 'https://www.inboundlogistics.com/cms/feed/',
    'logistics_business': 'https://www.logisticsbusiness.com/feed/',
    'supply_chain_brain': 'https://www.supplychainbrain.com/rss',
    'supply_chain_247': 'https://www.supplychain247.com/rss/news',
    'modern_materials_handling': 'https://www.mmh.com/rss',
    'warehouse_automation': 'https://www.warehouseautomation.ca/feed/',
    
    # === 산업 협회 및 연구기관 ===
    'cscmp': 'https://cscmp.org/news-resources/blog/rss',  # Council of Supply Chain Management
    'werc': 'https://werc.org/news-events/news/rss',  # Warehousing Education Research Council
    'mhi_solutions': 'https://www.mhi.org/news/rss',  # Material Handling Institute
    'promat': 'https://www.promatshow.com/news-media/press-releases/rss',
    
    # === 자동화 & 로보틱스 전문 ===
    'robotics_business_review': 'https://www.roboticsbusinessreview.com/feed/',
    'automation_world': 'https://www.automationworld.com/rss.xml',
    'robotics_247': 'https://www.robotics247.com/rss',
    'interact_analysis': 'https://www.interactanalysis.com/feed/',  # 시장 분석
    
    # === WMS/TMS 소프트웨어 전문 ===
    'logistics_tech': 'https://www.logisticstech.com/feed/',
    'supplychaindigital': 'https://supplychaindigital.com/feed',
    'logistics_it': 'https://www.logisticsindustry.com/rss',
    
    # === 유럽 물류 미디어 ===
    'logistics_manager_uk': 'https://logisticsmanager.com/feed/',
    'transport_intelligence': 'https://www.ti-insight.com/feed/',
    'log_net': 'https://www.log-net.com/rss',  # 독일
    
    # === 아시아-태평양 ===
    'logistics_asia': 'https://www.logisticsinsider.in/feed/',
    'asia_pacific_logistics': 'https://www.logisticsmiddleeast.com/rss',
    
    # === 전자상거래 물류 ===
    'ecommerce_logistics': 'https://ecommercelogistics.co.uk/feed/',
    'parcel_shipping': 'https://parcelshippingindex.com/feed/',
    
    # === 지속가능성 & 그린 로지스틱스 ===
    'green_supply_chain': 'https://www.greenbiz.com/topic/supply-chain/feed',
    'sustainable_logistics': 'https://www.logisticsviewpoints.com/feed/',
}

# 검색 키워드 (crawlable_site.md 기반 확장)
KEYWORDS = [
    # WMS 관련
    'WMS', 'warehouse management system', 'WMS implementation',
    'WMS ROI', 'WMS case study', 'WMS selection',
    
    # 자동화
    'warehouse automation', 'logistics automation', 'smart warehouse',
    'automated storage retrieval system', 'ASRS', 'AS/RS',
    'goods to person', 'GTP', 'pick to light', 'put to light',
    
    # 로보틱스
    'AGV', 'automated guided vehicle', 'AMR', 'autonomous mobile robot',
    'collaborative robot', 'cobot', 'picking robot', 'palletizing robot',
    
    # 기술
    'warehouse IoT', 'RFID warehouse', 'barcode scanning',
    'voice picking', 'augmented reality picking', 'AR warehouse',
    'digital twin warehouse', 'warehouse AI', 'machine learning warehouse',
    
    # 프로세스
    'order fulfillment', 'inventory optimization', 'slotting optimization',
    'cross docking', 'wave picking', 'zone picking', 'batch picking',
    'cycle counting', 'yard management', 'labor management system',
    
    # 산업별
    'e-commerce fulfillment', 'omnichannel logistics', '3PL automation',
    'cold chain warehouse', 'pharmaceutical warehouse', 'food distribution',
    
    # 비즈니스
    'warehouse ROI', 'cost benefit analysis', 'productivity improvement',
    'throughput optimization', 'space utilization', 'labor cost reduction',
    
    # 기업
    'Amazon fulfillment', 'DHL innovation', 'FedEx automation', 
    'Walmart distribution', 'Alibaba logistics', 'JD.com warehouse',
    'Ocado technology', 'Zappos warehouse',
    
    # 트렌드
    'dark warehouse', 'lights out warehouse', 'micro fulfillment',
    'urban warehouse', 'sustainable warehouse', 'green logistics',
    'warehouse 4.0', 'industry 4.0 warehouse',
]

# 웹 스크래핑 대상 (RSS 없는 중요 사이트들 - crawlable_site.md 기반)
SCRAPING_TARGETS = {
    # === 컨설팅 회사 케이스 스터디 ===
    'gartner_supply_chain': {
        'url': 'https://www.gartner.com/en/supply-chain',
        'type': 'dynamic',  # JavaScript 렌더링 필요
        'selectors': {
            'articles': '.article-card',
            'title': 'h3',
            'link': 'a',
            'date': '.date'
        }
    },
    'mckinsey_operations': {
        'url': 'https://www.mckinsey.com/capabilities/operations/our-insights',
        'type': 'dynamic',
        'selectors': {
            'articles': '.insight-card',
            'title': '.insight-title',
            'link': 'a',
        }
    },
    'accenture_supply_chain': {
        'url': 'https://www.accenture.com/us-en/insights/supply-chain-operations',
        'type': 'dynamic'  # JavaScript 렌더링 필요
    },
    
    # === 벤더 케이스 스터디 ===
    'manhattan_associates_resources': {
        'url': 'https://www.manh.com/resources/case-studies',
        'type': 'static',
        'filters': ['warehouse', 'fulfillment', 'distribution']
    },
    'blue_yonder_success': {
        'url': 'https://blueyonder.com/success-stories',
        'type': 'dynamic'
    },
    'oracle_wms_customers': {
        'url': 'https://www.oracle.com/customers/supply-chain-management/',
        'type': 'dynamic'
    },
    'sap_ewm_stories': {
        'url': 'https://www.sap.com/products/scm/ewm-warehouse-management/customers.html',
        'type': 'dynamic'
    },
    
    # === 자동화 벤더 ===
    'amazon_robotics_news': {
        'url': 'https://www.amazonrobotics.com/news',
        'type': 'static'
    },
    'zebra_perspectives': {
        'url': 'https://www.zebra.com/us/en/about-zebra/newsroom.html',
        'type': 'dynamic'
    },
    'honeywell_intelligrated': {
        'url': 'https://intelligrated.com/resources/',
        'type': 'static'
    },
    'dematic_insights': {
        'url': 'https://www.dematic.com/en/resources/',
        'type': 'dynamic'
    },
    'swisslog_blog': {
        'url': 'https://www.swisslog.com/en-us/insights',
        'type': 'dynamic'
    },
    'knapp_blog': {
        'url': 'https://www.knapp.com/en/news-media/',
        'type': 'static'
    },
    
    # === AGV/AMR 전문 ===
    'mobile_industrial_robots': {
        'url': 'https://www.mobile-industrial-robots.com/insights/',
        'type': 'dynamic'
    },
    'geek_plus': {
        'url': 'https://www.geekplus.com/resource-center',
        'type': 'dynamic'
    },
    'locus_robotics_blog': {
        'url': 'https://locusrobotics.com/blog/',
        'type': 'static'
    },
    'fetch_robotics': {
        'url': 'https://fetchrobotics.com/resources/',
        'type': 'static'
    },
}

# 제외 키워드 (노이즈 제거)
EXCLUDE_KEYWORDS = [
    'residential storage', 'self storage', 'mini warehouse',
    'data warehouse', 'cloud warehouse',  # IT 데이터웨어하우스 제외
]

