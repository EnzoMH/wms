#!/usr/bin/env python3
"""
WMS 트렌드 시각화 도구
==================

WMS 연구 데이터를 위한 상호작용 시각화 및 트렌드 분석을 생성합니다.
차트, 대시보드, 연구 트렌드의 시계열 분석을 생성합니다.

작성자: 신명호
날짜: 2025년 9월 3일
버전: 1.0.0
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
import argparse
from collections import defaultdict, Counter

# 시각화 라이브러리 import
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    from wordcloud import WordCloud
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"필수 패키지가 누락되었습니다. 설치해주세요: {e}")
    print("다음 명령어 실행: pip install matplotlib seaborn pandas numpy plotly wordcloud scikit-learn")
    exit(1)


class WMSTrendVisualizer:
    """WMS 연구 트렌드 시각화를 생성하는 메인 클래스입니다."""
    
    def __init__(self, papers_dir: str = "../Papers", processed_dir: str = "../ProcessedData", 
                 analysis_dir: str = "../Analysis", output_dir: str = "../Analysis"):
        """
        트렌드 시각화 도구를 초기화합니다.
        
        Args:
            papers_dir: 연구 논문이 포함된 디렉토리
            processed_dir: 처리된 텍스트 데이터가 포함된 디렉토리
            analysis_dir: 분석 결과가 포함된 디렉토리
            output_dir: 시각화 결과를 저장할 디렉토리
        """
        self.papers_dir = Path(papers_dir)
        self.processed_dir = Path(processed_dir)
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        self.setup_directories()
        
        # 데이터 저장소
        self.papers_data = []
        self.keywords_data = pd.DataFrame()
        self.trends_data = pd.DataFrame()
        
        # 시각화 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def setup_logging(self):
        """시각화 도구에 대한 로깅을 구성합니다."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('trend_visualizer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """출력 디렉토리를 생성합니다."""
        (self.output_dir / "charts").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "interactive").mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directories ready: {self.output_dir}")
    
    def load_all_data(self):
        """시각화를 위한 모든 가용 데이터를 로드합니다."""
        self.logger.info("모든 소스에서 데이터를 로드하는 중...")
        
        # 논문 메타데이터 로드
        self.load_papers_metadata()
        
        # 키워드 데이터 로드
        self.load_keywords_data()
        
        # 트렌드 분석 데이터 로드
        self.load_trends_data()
        
        # 연구 트렌드 JSON 로드
        self.load_research_trends()
        
        self.logger.info("데이터 로딩 완료")
    
    def load_papers_metadata(self):
        """모든 소스에서 논문 메타데이터를 로드합니다."""
        papers = []
        
        # ArXiv 논문
        arxiv_file = self.papers_dir / "ArXiv" / "metadata.json"
        if arxiv_file.exists():
            with open(arxiv_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for paper in data.get('papers', []):
                    papers.append({
                        'id': f"arxiv_{paper['id'].split('/')[-1]}",
                        'title': paper['title'],
                        'authors': len(paper['authors']),
                        'published': paper['published'],
                        'year': int(paper['published'].split('-')[0]),
                        'source': 'ArXiv',
                        'categories': ', '.join(paper.get('categories', [])),
                        'abstract_length': len(paper.get('abstract', ''))
                    })
        
        # Semantic Scholar 논문  
        ss_file = self.papers_dir / "SemanticScholar" / "search_results.json"
        if ss_file.exists():
            with open(ss_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for paper in data.get('papers', []):
                    papers.append({
                        'id': f"ss_{paper['id']}",
                        'title': paper['title'],
                        'authors': len(paper['authors']),
                        'year': paper['year'] if paper['year'] else 2024,
                        'source': 'Semantic Scholar',
                        'venue': paper.get('venue', ''),
                        'citations': paper.get('citation_count', 0),
                        'abstract_length': len(paper.get('abstract', ''))
                    })
        
        self.papers_data = papers
        self.logger.info(f"Loaded {len(papers)} papers for analysis")
    
    def load_keywords_data(self):
        """키워드 분석 데이터를 로드합니다."""
        keywords_file = self.processed_dir / "keywords.csv"
        if keywords_file.exists():
            self.keywords_data = pd.read_csv(keywords_file)
            self.logger.info(f"Loaded {len(self.keywords_data)} keywords")
    
    def load_trends_data(self):
        """트렌드 분석 데이터를 로드합니다."""
        trends_file = self.analysis_dir / "trend_analysis.csv"
        if trends_file.exists():
            self.trends_data = pd.read_csv(trends_file)
            self.logger.info(f"Loaded trend data with {len(self.trends_data)} entries")
    
    def load_research_trends(self):
        """연구 트렌드 JSON 데이터를 로드합니다."""
        trends_file = self.processed_dir / "research_trends.json"
        if trends_file.exists():
            with open(trends_file, 'r', encoding='utf-8') as f:
                self.research_trends = json.load(f)
                self.logger.info("연구 트렌드 데이터 로드 완료")
    
    def create_publication_trends(self):
        """발행 트렌드 시각화를 생성합니다."""
        self.logger.info("발행 트렌드 시각화 생성 중...")
        
        if not self.papers_data:
            self.logger.warning("트렌드 분석에 사용할 논문 데이터가 없습니다")
            return
        
        # DataFrame으로 변환
        df = pd.DataFrame(self.papers_data)
        
        # 연도별 발행 트렌드
        yearly_counts = df['year'].value_counts().sort_index()
        
        # matplotlib 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 연간 발행 트렌드
        ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2, markersize=8)
        ax1.set_title('연도별 WMS 연구 발행물', fontsize=16, fontweight='bold')
        ax1.set_xlabel('연도')
        ax1.set_ylabel('발행물 수')
        ax1.grid(True, alpha=0.3)
        
        # 소스별 발행물
        source_counts = df['source'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        ax2.bar(source_counts.index, source_counts.values, color=colors[:len(source_counts)])
        ax2.set_title('소스별 발행물', fontsize=16, fontweight='bold')
        ax2.set_xlabel('소스')
        ax2.set_ylabel('발행물 수')
        
        # 막대 그래프에 값 라벨 추가
        for i, v in enumerate(source_counts.values):
            ax2.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "publication_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 상호작용 Plotly 버전 생성
        fig_interactive = make_subplots(
            rows=2, cols=1,
            subplot_titles=('연도별 발행물', '소스별 발행물'),
            vertical_spacing=0.12
        )
        
        # 연간 트렌드
        fig_interactive.add_trace(
            go.Scatter(
                x=yearly_counts.index,
                y=yearly_counts.values,
                mode='lines+markers',
                name='Publications',
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=10)
            ),
            row=1, col=1
        )
        
        # 소스 분포
        fig_interactive.add_trace(
            go.Bar(
                x=source_counts.index,
                y=source_counts.values,
                name='Sources',
                marker_color=colors[:len(source_counts)],
                text=source_counts.values,
                textposition='outside'
            ),
            row=2, col=1
        )
        
        fig_interactive.update_layout(
            title_text="WMS 연구 발행 트렌드",
            height=800,
            showlegend=False
        )
        
        # 상호작용 그래프 저장
        pyo.plot(fig_interactive, filename=str(self.output_dir / "interactive" / "publication_trends.html"), 
                auto_open=False)
        
        self.logger.info("발행 트렌드 시각화 생성 완료")
    
    def create_keyword_analysis(self):
        """키워드 분석 시각화를 생성합니다."""
        self.logger.info("키워드 분석 시각화 생성 중...")
        
        if self.keywords_data.empty:
            self.logger.warning("사용할 키워드 데이터가 없습니다")
            return
        
        # 상위 키워드 막대 차트
        top_keywords = self.keywords_data.head(20)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 빈도 차트
        bars1 = ax1.barh(top_keywords['keyword'], top_keywords['frequency'], color='steelblue')
        ax1.set_title('빈도별 상위 20개 키워드', fontsize=14, fontweight='bold')
        ax1.set_xlabel('빈도')
        
        # 값 라벨 추가
        for bar in bars1:
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        # 평균 점수 차트
        if 'avg_score' in self.keywords_data.columns:
            bars2 = ax2.barh(top_keywords['keyword'], 
                           top_keywords['avg_score'].astype(float), color='coral')
            ax2.set_title('TF-IDF 점수별 상위 20개 키워드', fontsize=14, fontweight='bold')
            ax2.set_xlabel('평균 TF-IDF 점수')
            
            for bar in bars2:
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "keyword_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 워드클라우드 생성
        self.create_wordcloud()
        
        self.logger.info("키워드 분석 시각화 생성 완료")
    
    def create_wordcloud(self):
        """키워드로부터 워드클라우드를 생성합니다."""
        if self.keywords_data.empty:
            return
        
        # 단어 빈도 준비
        word_freq = {}
        for _, row in self.keywords_data.head(50).iterrows():
            word_freq[row['keyword']] = row['frequency']
        
        # 워드클라우드 생성
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5
        ).generate_from_frequencies(word_freq)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('WMS 연구 키워드 워드클라우드', fontsize=20, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "keywords_wordcloud.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("워드클라우드 생성 완료")
    
    def create_technology_adoption_trends(self):
        """기술 도입 트렌드 시각화를 생성합니다."""
        self.logger.info("기술 도입 트렌드 생성 중...")
        
        if self.trends_data.empty:
            self.logger.warning("사용할 트렌드 데이터가 없습니다")
            return
        
        # 최근 연도만 필터링
        recent_trends = self.trends_data[self.trends_data['year'] >= 2020]
        
        # 시간에 따른 기술 도입
        fig, ax = plt.subplots(figsize=(14, 8))
        
        technologies = recent_trends['technology'].unique()[:8]  # 상위 8가지 기술
        colors = plt.cm.Set3(np.linspace(0, 1, len(technologies)))
        
        for i, tech in enumerate(technologies):
            tech_data = recent_trends[recent_trends['technology'] == tech]
            if not tech_data.empty:
                ax.plot(tech_data['year'], tech_data['adoption_rate'], 
                       marker='o', linewidth=2, label=tech, color=colors[i])
        
        ax.set_title('WMS 기술 도입 트렌드 (2020-2024)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('도입률 (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "technology_adoption_trends.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROI vs 도입률 산점도 차트 생성
        self.create_roi_adoption_scatter()
        
        self.logger.info("기술 도입 트렌드 생성 완료")
    
    def create_roi_adoption_scatter(self):
        """ROI vs 도입률 산점도 차트를 생성합니다."""
        if self.trends_data.empty:
            return
        
        # 각 기술의 최신 데이터 획득
        latest_data = self.trends_data.loc[self.trends_data.groupby('technology')['year'].idxmax()]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 산점도 차트 생성
        scatter = ax.scatter(latest_data['adoption_rate'], latest_data['roi_reported'],
                           s=latest_data['research_papers'] * 10,  # 연구 논문 수에 따른 크기
                           alpha=0.6, c=latest_data['year'], cmap='viridis')
        
        # 각 점에 라벨 추가
        for _, row in latest_data.iterrows():
            ax.annotate(row['technology'], 
                       (row['adoption_rate'], row['roi_reported']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_xlabel('도입률 (%)')
        ax.set_ylabel('보고된 ROI (%)')
        ax.set_title('기술 ROI vs 도입률\\n(버블 크기 = 연구 논문 수)', 
                    fontsize=14, fontweight='bold')
        
        # 컴러바 추가
        cbar = plt.colorbar(scatter)
        cbar.set_label('Year')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "roi_adoption_scatter.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_research_evolution_timeline(self):
        """연구 진화 타임라인을 생성합니다."""
        self.logger.info("연구 진화 타임라인 생성 중...")
        
        if not hasattr(self, 'research_trends'):
            self.logger.warning("사용할 연구 트렌드 데이터가 없습니다")
            return
        
        # 연간 트렌드 추출
        yearly_data = self.research_trends.get('yearly_trends', {})
        
        if not yearly_data:
            return
        
        # 타임라인용 데이터 준비
        years = sorted(yearly_data.keys())
        paper_counts = [yearly_data[year]['paper_count'] for year in years]
        avg_citations = [yearly_data[year]['avg_citation_count'] for year in years]
        
        # 타임라인 그래프 생성
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 논문 수 타임라인
        bars1 = ax1.bar(years, paper_counts, color='steelblue', alpha=0.7)
        ax1.set_title('연구 성과 타임라인', fontsize=16, fontweight='bold')
        ax1.set_ylabel('논문 수')
        
        # 값 라벨 추가
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 인용 타임라인
        line2 = ax2.plot(years, avg_citations, marker='o', linewidth=3, 
                        markersize=8, color='coral')
        ax2.set_title('평균 인용 수 타임라인', fontsize=16, fontweight='bold')
        ax2.set_xlabel('연도')
        ax2.set_ylabel('평균 인용 수')
        ax2.grid(True, alpha=0.3)
        
        # 값 라벨 추가 for citations
        for i, (year, citations) in enumerate(zip(years, avg_citations)):
            ax2.annotate(f'{citations:.1f}', (year, citations),
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "charts" / "research_evolution_timeline.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("연구 진화 타임라인 생성 완료")
    
    def create_interactive_dashboard(self):
        """종합적인 상호작용 대시보드를 생성합니다."""
        self.logger.info("상호작용 대시보드 생성 중...")
        
        # 여러 하위 그래프를 가진 메인 대시보드 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                '연도별 발행물', '소스별 발행물',
                '상위 키워드', '기술 도입',
                '연구 중점 영역', '지리적 분포'
            ),
            specs=[
                [{"secondary_y": False}, {"type": "pie"}],
                [{"type": "bar"}, {"secondary_y": False}],
                [{"type": "pie"}, {"type": "bar"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 연도별 발행물
        if self.papers_data:
            df = pd.DataFrame(self.papers_data)
            yearly_counts = df['year'].value_counts().sort_index()
            
            fig.add_trace(
                go.Scatter(
                    x=yearly_counts.index,
                    y=yearly_counts.values,
                    mode='lines+markers',
                    name='Publications',
                    line=dict(width=3)
                ),
                row=1, col=1
            )
            
            # 소스별 발행물 (원형 차트)
            source_counts = df['source'].value_counts()
            fig.add_trace(
                go.Pie(
                    labels=source_counts.index,
                    values=source_counts.values,
                    name="Sources"
                ),
                row=1, col=2
            )
        
        # 상위 키워드
        if not self.keywords_data.empty:
            top_keywords = self.keywords_data.head(15)
            fig.add_trace(
                go.Bar(
                    x=top_keywords['frequency'],
                    y=top_keywords['keyword'],
                    orientation='h',
                    name='Keywords'
                ),
                row=2, col=1
            )
        
        # 기술 도입 (트렌드 데이터가 있는 경우)
        if not self.trends_data.empty:
            recent_trends = self.trends_data[self.trends_data['year'] >= 2022]
            tech_adoption = recent_trends.groupby('technology')['adoption_rate'].mean().sort_values(ascending=False).head(10)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(tech_adoption))),
                    y=tech_adoption.values,
                    mode='markers+lines',
                    name='Adoption Rate',
                    text=tech_adoption.index,
                    textposition='top center'
                ),
                row=2, col=2
            )
        
        # 연구 중점 영역 (연구 트렌드에서)
        if hasattr(self, 'research_trends'):
            focus_areas = self.research_trends.get('research_focus_areas', {})
            if focus_areas:
                areas = list(focus_areas.keys())
                percentages = [focus_areas[area]['percentage'] for area in areas]
                
                fig.add_trace(
                    go.Pie(
                        labels=areas,
                        values=percentages,
                        name="Research Focus"
                    ),
                    row=3, col=1
                )
            
            # 지리적 분포
            geo_data = self.research_trends.get('geographic_distribution', {})
            if geo_data:
                regions = list(geo_data.keys())
                paper_counts = [geo_data[region]['paper_count'] for region in regions]
                
                fig.add_trace(
                    go.Bar(
                        x=regions,
                        y=paper_counts,
                        name='Geographic Distribution'
                    ),
                    row=3, col=2
                )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title_text="WMS 연구 분석 대시보드",
            height=1200,
            showlegend=False,
            font=dict(size=10)
        )
        
        # 상호작용 대시보드 저장
        pyo.plot(fig, filename=str(self.output_dir / "interactive" / "research_dashboard.html"), 
                auto_open=False)
        
        self.logger.info("상호작용 대시보드 생성 완료")
    
    def create_all_visualizations(self):
        """모든 시각화 결과물을 생성합니다."""
        self.logger.info("종합적인 시각화 생성 시작...")
        
        # 모든 데이터 로드
        self.load_all_data()
        
        # 모든 시각화 생성
        self.create_publication_trends()
        self.create_keyword_analysis()
        self.create_technology_adoption_trends()
        self.create_research_evolution_timeline()
        self.create_interactive_dashboard()
        
        # 요약 보고서 생성
        self.generate_visualization_report()
        
        self.logger.info("모든 시각화가 성공적으로 생성되었습니다!")
    
    def generate_visualization_report(self):
        """생성된 모든 시각화의 보고서를 생성합니다."""
        report = f"""
WMS Research Trend Visualization Report
======================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Sources: Papers, Keywords, Trends, Analysis Results

STATIC VISUALIZATIONS (PNG)
============================

Charts Generated:
- publication_trends.png: Publication volume and source distribution over time
- keyword_analysis.png: Top keywords by frequency and TF-IDF scores  
- keywords_wordcloud.png: Visual word cloud of research keywords
- technology_adoption_trends.png: Technology adoption rates over time
- roi_adoption_scatter.png: ROI vs adoption rate correlation analysis
- research_evolution_timeline.png: Research output and citation trends

INTERACTIVE VISUALIZATIONS (HTML)
==================================

Interactive Charts:
- publication_trends.html: Interactive publication trend analysis
- research_dashboard.html: Comprehensive research analytics dashboard

INSIGHTS GENERATED
==================

1. Publication Trends: {len(self.papers_data)} papers analyzed across {len(set([p['source'] for p in self.papers_data]))} sources
2. Keyword Analysis: {len(self.keywords_data)} unique keywords identified
3. Technology Focus: {len(self.trends_data['technology'].unique()) if not self.trends_data.empty else 'N/A'} technologies tracked
4. Temporal Coverage: {min([p['year'] for p in self.papers_data]) if self.papers_data else 'N/A'}-{max([p['year'] for p in self.papers_data]) if self.papers_data else 'N/A'}

USAGE RECOMMENDATIONS
=====================

1. Use static charts for presentations and reports
2. Use interactive dashboard for detailed data exploration
3. Refer to keyword analysis for research gap identification
4. Monitor technology adoption trends for investment decisions

FILES LOCATION
==============

Static Charts: {self.output_dir}/charts/
Interactive Charts: {self.output_dir}/interactive/

Next Steps:
- Review visualizations for insights
- Update data regularly for trend monitoring
- Customize charts for specific reporting needs
- Integrate findings into research strategy
"""
        
        report_file = self.output_dir / "visualization_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"Visualization report saved to: {report_file}")


def main():
    """트렌드 시각화 도구를 실행하는 메인 함수입니다."""
    parser = argparse.ArgumentParser(description="WMS 연구 트렌드 시각화 도구")
    parser.add_argument("--papers-dir", default="../Papers", help="연구 논문이 포함된 디렉토리")
    parser.add_argument("--processed-dir", default="../ProcessedData", help="처리된 데이터가 포함된 디렉토리")
    parser.add_argument("--analysis-dir", default="../Analysis", help="분석 결과가 포함된 디렉토리")
    parser.add_argument("--output-dir", default="../Analysis", help="시각화 결과물을 위한 출력 디렉토리")
    parser.add_argument("--chart-type", choices=['all', 'static', 'interactive'], 
                       default='all', help="생성할 차트 유형")
    
    args = parser.parse_args()
    
    visualizer = WMSTrendVisualizer(args.papers_dir, args.processed_dir, args.analysis_dir, args.output_dir)
    
    if args.chart_type == 'all':
        visualizer.create_all_visualizations()
    elif args.chart_type == 'static':
        visualizer.load_all_data()
        visualizer.create_publication_trends()
        visualizer.create_keyword_analysis()
        visualizer.create_technology_adoption_trends()
        visualizer.create_research_evolution_timeline()
    elif args.chart_type == 'interactive':
        visualizer.load_all_data()
        visualizer.create_interactive_dashboard()


if __name__ == "__main__":
    main()
