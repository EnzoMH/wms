#!/usr/bin/env python3
"""
산업 뉴스 & Case Study 수집기
============================

Supply Chain, Logistics, SmartFactory 관련 뉴스 및 사례 연구 수집

작성자: 신명호
날짜: 2025년 10월 13일
"""

import os
import json
import time
import logging
import feedparser
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from bs4 import BeautifulSoup

# 크롤링 대상 사이트 및 키워드 import
from site_config import RSS_FEEDS, KEYWORDS, SCRAPING_TARGETS, EXCLUDE_KEYWORDS

# Selenium 관련 import (JavaScript 렌더링용)
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium이 설치되지 않았습니다. JavaScript 렌더링이 필요한 사이트는 수집할 수 없습니다.")
    print("설치: pip install selenium")
    print("Chrome 드라이버도 필요합니다: https://chromedriver.chromium.org/")

class IndustryNewsScaper:
    """산업 뉴스 및 Case Study 수집기"""
    
    def __init__(self, output_dir: str = "../../1_data/0_news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # 크롤링 대상 사이트 및 키워드 로드 (0_site.py에서 import)
        self.rss_feeds = RSS_FEEDS
        self.keywords = KEYWORDS
        self.scraping_targets = SCRAPING_TARGETS
        self.exclude_keywords = EXCLUDE_KEYWORDS
        
    def setup_logging(self):
        """로깅 설정"""
        log_file = self.output_dir / "news_scraper.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8', errors='ignore'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def scrape_rss_feed(self, source_name: str, feed_url: str, max_articles: int = 50):
        """RSS 피드에서 기사 수집"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📰 {source_name} RSS 피드 수집 시작...")
        self.logger.info(f"{'='*60}")
        
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                # 키워드 필터링
                title = entry.get('title', '').lower()
                summary = entry.get('summary', '').lower()
                
                if any(kw.lower() in title or kw.lower() in summary for kw in self.keywords):
                    article = {
                        'source': source_name,
                        'title': entry.get('title', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'summary': entry.get('summary', ''),
                        'collected_at': datetime.now().isoformat()
                    }
                    articles.append(article)
                    self.logger.info(f"수집: {article['title'][:60]}...")
            
            self.logger.info(f"총 {len(articles)}개 관련 기사 수집 완료")        
        except Exception as e:
            self.logger.error(f"RSS 피드 수집 실패: {e}")
            return []
        
        return articles
    
    def scrape_static_site(self, site_name: str, config: Dict):
        """정적 사이트 크롤링 (Selenium 없이)"""
        self.logger.info(f"📄 {site_name} 정적 사이트 크롤링...")
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(config['url'], headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 기본 선택자로 콘텐츠 추출
            articles = []
            selectors = config.get('selectors', {})
            
            # 다양한 선택자 시도
            article_selectors = [
                selectors.get('articles', ''),
                '.case-study', '.success-story', '.resource-item',
                '.blog-post', '.news-item', '.article-card',
                'article', '.content-item'
            ]
            
            for selector in article_selectors:
                if selector:
                    items = soup.select(selector)
                    if items:
                        self.logger.info(f"[Success!]{site_name}: {len(items)}개 아이템 발견 (selector: {selector})")
                        
                        for item in items[:10]:  # 최대 10개
                            try:
                                # 제목 추출
                                title_elem = item.find(['h1', 'h2', 'h3', 'h4']) or item
                                title = title_elem.get_text(strip=True) if title_elem else ""
                                
                                # 링크 추출
                                link_elem = item.find('a') if item.name != 'a' else item
                                link = link_elem.get('href', '') if link_elem else ""
                                
                                if link and link.startswith('/'):
                                    from urllib.parse import urljoin
                                    link = urljoin(config['url'], link)
                                
                                # 설명 추출
                                desc_elem = item.find(['p', '.description', '.summary'])
                                description = desc_elem.get_text(strip=True) if desc_elem else ""
                                
                                # 키워드 필터링
                                content_text = f"{title} {description}".lower()
                                if any(keyword in content_text for keyword in self.keywords[:20]):  # 처음 20개 키워드만
                                    
                                    article = {
                                        'title': title,
                                        'url': link,
                                        'description': description,
                                        'source': site_name,
                                        'date_collected': datetime.now().isoformat(),
                                        'type': 'static_scraping'
                                    }
                                    
                                    articles.append(article)
                                    self.logger.info(f"📄 수집: {title[:50]}...")
                                    
                            except Exception as e:
                                self.logger.debug(f"아이템 처리 중 오류: {e}")
                                continue
                                
                        break  # 성공적으로 수집했으면 다른 선택자는 시도 안함
            
            return articles
            
        except Exception as e:
            self.logger.error(f"[ X ] {site_name} 정적 크롤링 실패: {e}")
            return []
    
    def scrape_dynamic_site(self, site_name: str, config: Dict):
        """동적 사이트 크롤링 (Selenium 사용)"""
        if not SELENIUM_AVAILABLE:
            self.logger.warning(f"⚠️ {site_name}: Selenium 미설치로 동적 사이트 크롤링 불가")
            return []
            
        self.logger.info(f"{site_name} 동적 사이트 크롤링 (Selenium)...")
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        driver = None
        articles = []
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.get(config['url'])
            
            # JavaScript 렌더링 대기
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            time.sleep(3)  # 추가 대기
            
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # 정적 사이트와 동일한 로직으로 처리
            selectors = config.get('selectors', {})
            article_selectors = [
                selectors.get('articles', ''),
                '.case-study', '.success-story', '.resource-item',
                '.insight-card', '.article-card', '.news-item',
                'article', '.content-item'
            ]
            
            for selector in article_selectors:
                if selector:
                    items = soup.select(selector)
                    if items:
                        self.logger.info(f"[Success!]{site_name}: {len(items)}개 아이템 발견 (selector: {selector})")
                        
                        for item in items[:10]:  # 최대 10개
                            try:
                                title_elem = item.find(['h1', 'h2', 'h3', 'h4']) or item
                                title = title_elem.get_text(strip=True) if title_elem else ""
                                
                                link_elem = item.find('a') if item.name != 'a' else item
                                link = link_elem.get('href', '') if link_elem else ""
                                
                                if link and link.startswith('/'):
                                    from urllib.parse import urljoin
                                    link = urljoin(config['url'], link)
                                
                                desc_elem = item.find(['p', '.description', '.summary'])
                                description = desc_elem.get_text(strip=True) if desc_elem else ""
                                
                                content_text = f"{title} {description}".lower()
                                if any(keyword in content_text for keyword in self.keywords[:20]):
                                    
                                    article = {
                                        'title': title,
                                        'url': link,
                                        'description': description,
                                        'source': site_name,
                                        'date_collected': datetime.now().isoformat(),
                                        'type': 'dynamic_scraping'
                                    }
                                    
                                    articles.append(article)
                                    self.logger.info(f"📄 수집: {title[:50]}...")
                                    
                            except Exception as e:
                                self.logger.debug(f"아이템 처리 중 오류: {e}")
                                continue
                                
                        break
            
            return articles
            
        except Exception as e:
            self.logger.error(f" {site_name} 동적 크롤링 실패: {e}")
            return []
        finally:
            if driver:
                driver.quit()
    
    def scrape_all_targets(self):
        """모든 웹 스크래핑 대상 사이트 크롤링"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("웹 스크래핑 대상 사이트들 크롤링 시작...")
        self.logger.info(f"{'='*60}")
        
        all_articles = []
        
        for site_name, config in self.scraping_targets.items():
            try:
                if config.get('type') == 'static':
                    articles = self.scrape_static_site(site_name, config)
                elif config.get('type') == 'dynamic':
                    articles = self.scrape_dynamic_site(site_name, config)
                else:
                    # 기본적으로 정적 시도 후 실패하면 동적 시도
                    articles = self.scrape_static_site(site_name, config)
                    if not articles:
                        articles = self.scrape_dynamic_site(site_name, config)
                
                if articles:
                    self.save_articles(articles, f"scraping_{site_name}")
                    all_articles.extend(articles)
                
                time.sleep(2)  # 사이트에 부하 주지 않기 위해 대기
                
            except Exception as e:
                self.logger.error(f"[ X ] {site_name} 크롤링 중 오류: {e}")
                continue
        
        self.logger.info(f"[Success!]웹 스크래핑 완료: 총 {len(all_articles)}개 기사")
        return all_articles
    
    def scrape_aws_case_studies(self, max_pages: int = 3):
        """AWS Case Study 페이지 크롤링"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("AWS Case Studies 수집 시작...")
        self.logger.info(f"{'='*60}")
        
        # AWS Case Studies는 JavaScript 렌더링이 필요할 수 있음
        # 여기서는 기본 구조만 제공
        base_url = "https://aws.amazon.com/solutions/case-studies/"
        
        self.logger.info(" AWS Case Studies는 JavaScript가 필요할 수 있습니다")
        self.logger.info(" 수동 수집을 권장합니다:")
        self.logger.info("   1. https://aws.amazon.com/solutions/case-studies/ 방문")
        self.logger.info("   2. 'logistics', 'warehouse', 'supply chain' 검색")
        self.logger.info("   3. PDF 다운로드 후 1_data/0_news/case_studies/ 에 저장")
        
        return []
    
    def save_articles(self, articles: List[Dict], source_name: str):
        """수집한 기사를 JSON으로 저장"""
        if not articles:
            return
        
        output_file = self.output_dir / f"{source_name}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"저장 완료: {output_file}")
        
        # 텍스트 파일로도 저장 (RAG 시스템에서 쉽게 읽기)
        txt_file = self.output_dir / f"{source_name}_{datetime.now().strftime('%Y%m%d')}.txt"
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(f"{'='*80}\n")
                f.write(f"제목: {article['title']}\n")
                f.write(f"출처: {article['source']}\n")
                f.write(f"링크: {article['link']}\n")
                f.write(f"날짜: {article['published']}\n")
                f.write(f"\n내용:\n{article['summary']}\n")
                f.write(f"{'='*80}\n\n")
        
        self.logger.info(f" 텍스트 파일 저장: {txt_file}")
    
    def run(self):
        """전체 수집 프로세스 실행"""
        self.logger.info("\n" + "="*60)
        self.logger.info(" 산업 뉴스 & Case Study 수집기 시작")
        self.logger.info("="*60 + "\n")
        
        all_articles = []
        
        # RSS 피드 수집
        for source_name, feed_url in self.rss_feeds.items():
            articles = self.scrape_rss_feed(source_name, feed_url)
            if articles:
                self.save_articles(articles, source_name)
                all_articles.extend(articles)
            time.sleep(2)  # API 제한 고려
        
        # AWS Case Studies (Selenium 사용)
        aws_articles = self.scrape_aws_case_studies()
        if aws_articles:
            self.save_articles(aws_articles, "AWS_Case_Studies")
            all_articles.extend(aws_articles)
        
        # 웹 스크래핑 대상 사이트들
        self.logger.info("🔄 3단계: 벤더 케이스 스터디 & 전문 사이트 크롤링...")
        scraping_articles = self.scrape_all_targets()
        all_articles.extend(scraping_articles)
        
        # 최종 리포트
        self.logger.info(f"\n{'='*60}")
        self.logger.info(" 수집 완료 리포트")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"총 수집 기사: {len(all_articles)}개")
        self.logger.info(f"저장 위치: {self.output_dir}")
        self.logger.info(f"\n 다음 단계:")
        self.logger.info(f"   1. {self.output_dir}/*.txt 파일 확인")
        self.logger.info(f"   2. text_extractor.py로 텍스트 청크 생성")
        self.logger.info(f"   3. faiss_builder.py로 벡터DB 업데이트")
        
        return len(all_articles)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="산업 뉴스 & Case Study 수집기"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../../1_data/0_news",
        help="수집한 기사를 저장할 디렉토리"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=50,
        help="피드당 최대 수집 기사 수"
    )
    
    args = parser.parse_args()
    
    scraper = IndustryNewsScaper(output_dir=args.output_dir)
    total = scraper.run()
    
    print(f"\n 수집 완료! 총 {total}개 기사")

