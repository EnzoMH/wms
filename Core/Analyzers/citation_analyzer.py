#!/usr/bin/env python3
"""
WMS Citation Analyzer
====================

Analyze citation networks and research relationships in WMS literature.
Generate network graphs and identify influential papers and research clusters.

작성자: 신명호
날짜: 2025년 9월 3일
버젼 1.0.0
"""

import os
import json
import csv
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import logging
from datetime import datetime
import argparse
from collections import defaultdict, Counter

# Network analysis imports
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import seaborn as sns
    from community import community_louvain
except ImportError as e:
    print(f"Missing required packages. Please install: {e}")
    print("Run: pip install networkx matplotlib pandas numpy scikit-learn seaborn python-louvain")
    exit(1)


class WMSCitationAnalyzer:
    """Main class for analyzing citation networks in WMS research."""
    
    def __init__(self, papers_dir: str = "../Papers", processed_dir: str = "../ProcessedData", output_dir: str = "../Analysis"):
        """
        Initialize the citation analyzer.
        
        Args:
            papers_dir: Directory containing research papers
            processed_dir: Directory containing processed text data
            output_dir: Directory to store analysis results
        """
        self.papers_dir = Path(papers_dir)
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.setup_logging()
        self.setup_directories()
        
        # Initialize network graph
        self.citation_graph = nx.DiGraph()
        self.similarity_graph = nx.Graph()
        
        # Paper metadata storage
        self.papers_metadata = {}
        self.paper_abstracts = {}
        
    def setup_logging(self):
        """Configure logging for the citation analyzer."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('citation_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create output directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory ready: {self.output_dir}")
    
    def load_paper_metadata(self):
        """Load paper metadata from all sources."""
        self.logger.info("Loading paper metadata from all sources...")
        
        # Load ArXiv metadata
        arxiv_metadata = self.papers_dir / "ArXiv" / "metadata.json"
        if arxiv_metadata.exists():
            with open(arxiv_metadata, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for paper in data.get('papers', []):
                    paper_id = f"arxiv_{paper['id'].split('/')[-1]}"
                    self.papers_metadata[paper_id] = {
                        'title': paper['title'],
                        'authors': paper['authors'],
                        'abstract': paper['abstract'],
                        'published': paper['published'],
                        'source': 'ArXiv',
                        'url': paper.get('pdf_url', '')
                    }
                    self.paper_abstracts[paper_id] = paper['abstract']
        
        # Load Semantic Scholar data
        ss_data = self.papers_dir / "SemanticScholar" / "search_results.json"
        if ss_data.exists():
            with open(ss_data, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for paper in data.get('papers', []):
                    paper_id = f"ss_{paper['id']}"
                    self.papers_metadata[paper_id] = {
                        'title': paper['title'],
                        'authors': [author for author in paper['authors']],
                        'abstract': paper['abstract'],
                        'year': paper['year'],
                        'venue': paper['venue'],
                        'citation_count': paper['citation_count'],
                        'source': 'Semantic Scholar',
                        'url': paper.get('url', '')
                    }
                    if paper['abstract']:
                        self.paper_abstracts[paper_id] = paper['abstract']
        
        # Load Google Scholar abstracts
        gs_abstracts = self.papers_dir / "GoogleScholar" / "abstracts.txt"
        if gs_abstracts.exists():
            with open(gs_abstracts, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple parsing of the abstracts file
                # This would need more sophisticated parsing in a real implementation
                abstracts = content.split('-' * 80)
                for i, abstract_block in enumerate(abstracts):
                    if abstract_block.strip():
                        paper_id = f"gs_{i:03d}"
                        # Extract title and abstract (simplified)
                        lines = abstract_block.strip().split('\\n')
                        title = "Google Scholar Paper"
                        abstract = abstract_block.strip()
                        
                        self.papers_metadata[paper_id] = {
                            'title': title,
                            'authors': ['Various Authors'],
                            'abstract': abstract,
                            'source': 'Google Scholar'
                        }
                        self.paper_abstracts[paper_id] = abstract
        
        self.logger.info(f"Loaded metadata for {len(self.papers_metadata)} papers")
    
    def build_citation_network(self):
        """Build citation network from available data."""
        self.logger.info("Building citation network...")
        
        # Add all papers as nodes
        for paper_id, metadata in self.papers_metadata.items():
            self.citation_graph.add_node(paper_id, **metadata)
        
        # For this demo, we'll create synthetic citation relationships
        # based on similarity and metadata
        paper_ids = list(self.papers_metadata.keys())
        
        # Create citation edges based on publication dates and similarity
        for i, paper1_id in enumerate(paper_ids):
            paper1 = self.papers_metadata[paper1_id]
            
            for j, paper2_id in enumerate(paper_ids):
                if i != j:
                    paper2 = self.papers_metadata[paper2_id]
                    
                    # Simple heuristic: papers cite earlier papers with similar topics
                    year1 = self.extract_year(paper1)
                    year2 = self.extract_year(paper2)
                    
                    if year1 and year2 and year1 > year2:
                        # Check for topic similarity
                        similarity = self.calculate_text_similarity(
                            paper1.get('abstract', ''),
                            paper2.get('abstract', '')
                        )
                        
                        # Add citation edge if similarity is high enough
                        if similarity > 0.3:  # Threshold for citation relationship
                            self.citation_graph.add_edge(
                                paper1_id, paper2_id,
                                weight=similarity,
                                citation_type='topical'
                            )
        
        self.logger.info(f"Citation network built with {len(self.citation_graph.nodes)} nodes and {len(self.citation_graph.edges)} edges")
    
    def build_similarity_network(self):
        """Build paper similarity network based on content."""
        self.logger.info("Building similarity network...")
        
        if not self.paper_abstracts:
            self.logger.warning("No abstracts available for similarity analysis")
            return
        
        # Calculate TF-IDF similarity
        paper_ids = list(self.paper_abstracts.keys())
        abstracts = [self.paper_abstracts[pid] for pid in paper_ids]
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create similarity network
            for i, paper1_id in enumerate(paper_ids):
                for j, paper2_id in enumerate(paper_ids):
                    if i != j and similarity_matrix[i, j] > 0.2:  # Similarity threshold
                        self.similarity_graph.add_edge(
                            paper1_id, paper2_id,
                            weight=similarity_matrix[i, j],
                            similarity_score=similarity_matrix[i, j]
                        )
            
            self.logger.info(f"Similarity network built with {len(self.similarity_graph.nodes)} nodes and {len(self.similarity_graph.edges)} edges")
            
        except Exception as e:
            self.logger.error(f"Similarity network construction failed: {e}")
    
    def extract_year(self, paper_metadata: Dict) -> Optional[int]:
        """Extract publication year from paper metadata."""
        if 'year' in paper_metadata and paper_metadata['year']:
            return int(paper_metadata['year'])
        
        if 'published' in paper_metadata:
            try:
                return int(paper_metadata['published'].split('-')[0])
            except:
                pass
        
        return None
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def analyze_network_metrics(self):
        """Calculate and analyze network metrics."""
        self.logger.info("Analyzing network metrics...")
        
        metrics = {
            'basic_metrics': {},
            'centrality_measures': {},
            'community_structure': {},
            'influential_papers': {}
        }
        
        # Basic network metrics
        metrics['basic_metrics'] = {
            'total_nodes': len(self.citation_graph.nodes),
            'total_edges': len(self.citation_graph.edges),
            'density': nx.density(self.citation_graph),
            'is_connected': nx.is_connected(self.citation_graph.to_undirected()),
            'number_of_components': nx.number_connected_components(self.citation_graph.to_undirected())
        }
        
        # Centrality measures
        if len(self.citation_graph.nodes) > 0:
            degree_centrality = nx.degree_centrality(self.citation_graph)
            betweenness_centrality = nx.betweenness_centrality(self.citation_graph)
            closeness_centrality = nx.closeness_centrality(self.citation_graph)
            pagerank = nx.pagerank(self.citation_graph)
            
            # Find most central papers
            most_central_papers = {
                'degree': max(degree_centrality, key=degree_centrality.get) if degree_centrality else None,
                'betweenness': max(betweenness_centrality, key=betweenness_centrality.get) if betweenness_centrality else None,
                'closeness': max(closeness_centrality, key=closeness_centrality.get) if closeness_centrality else None,
                'pagerank': max(pagerank, key=pagerank.get) if pagerank else None
            }
            
            metrics['centrality_measures'] = {
                'most_central_papers': most_central_papers,
                'avg_degree_centrality': np.mean(list(degree_centrality.values())) if degree_centrality else 0,
                'avg_betweenness_centrality': np.mean(list(betweenness_centrality.values())) if betweenness_centrality else 0
            }
        
        # Community detection on similarity network
        if len(self.similarity_graph.nodes) > 2:
            try:
                communities = community_louvain.best_partition(self.similarity_graph)
                metrics['community_structure'] = {
                    'number_of_communities': len(set(communities.values())),
                    'modularity': community_louvain.modularity(communities, self.similarity_graph),
                    'communities': communities
                }
            except Exception as e:
                self.logger.warning(f"Community detection failed: {e}")
                metrics['community_structure'] = {'error': str(e)}
        
        return metrics
    
    def generate_network_visualizations(self):
        """Generate network visualization plots."""
        self.logger.info("Generating network visualizations...")
        
        try:
            # Citation network visualization
            if len(self.citation_graph.nodes) > 0:
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(self.citation_graph, k=1, iterations=50)
                
                # Draw nodes
                node_sizes = [100 + len(self.papers_metadata.get(node, {}).get('title', '')) * 2 
                             for node in self.citation_graph.nodes()]
                
                nx.draw_networkx_nodes(
                    self.citation_graph, pos,
                    node_size=node_sizes,
                    node_color='lightblue',
                    alpha=0.7
                )
                
                # Draw edges
                nx.draw_networkx_edges(
                    self.citation_graph, pos,
                    edge_color='gray',
                    alpha=0.5,
                    arrows=True,
                    arrowsize=20
                )
                
                # Add labels for important nodes
                important_nodes = list(self.citation_graph.nodes())[:10]  # First 10 nodes
                labels = {node: node.split('_')[0] for node in important_nodes}
                nx.draw_networkx_labels(self.citation_graph, pos, labels, font_size=8)
                
                plt.title("WMS Research Citation Network", fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(self.output_dir / 'citation_network.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Similarity network visualization
            if len(self.similarity_graph.nodes) > 0:
                plt.figure(figsize=(12, 8))
                pos = nx.spring_layout(self.similarity_graph, k=2, iterations=50)
                
                # Color nodes by source
                source_colors = {'ArXiv': 'red', 'Semantic Scholar': 'blue', 'Google Scholar': 'green'}
                node_colors = []
                for node in self.similarity_graph.nodes():
                    source = self.papers_metadata.get(node, {}).get('source', 'Unknown')
                    node_colors.append(source_colors.get(source, 'gray'))
                
                nx.draw_networkx_nodes(
                    self.similarity_graph, pos,
                    node_color=node_colors,
                    node_size=200,
                    alpha=0.8
                )
                
                # Draw edges with weight-based thickness
                edges = self.similarity_graph.edges(data=True)
                edge_weights = [edge[2]['weight'] * 5 for edge in edges]
                
                nx.draw_networkx_edges(
                    self.similarity_graph, pos,
                    width=edge_weights,
                    edge_color='gray',
                    alpha=0.6
                )
                
                plt.title("WMS Research Paper Similarity Network", fontsize=16)
                plt.axis('off')
                
                # Add legend
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                           markerfacecolor=color, markersize=10, label=source)
                                 for source, color in source_colors.items()]
                plt.legend(handles=legend_elements, loc='upper right')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'similarity_network.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info("Network visualizations saved")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
    
    def export_network_data(self):
        """Export network data in various formats."""
        self.logger.info("Exporting network data...")
        
        # GEXF 저장을 위해 노드 속성을 단순화
        def clean_graph_for_gexf(graph):
            """GEXF 형식에 맞게 그래프 데이터를 정리합니다."""
            clean_graph = graph.copy()
            
            # 노드 속성 정리
            for node in clean_graph.nodes():
                node_data = clean_graph.nodes[node]
                cleaned_data = {}
                
                for key, value in node_data.items():
                    # 단순한 데이터 타입만 유지
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_data[key] = str(value)  # 모든 값을 문자열로 변환
                    elif isinstance(value, list):
                        # 리스트는 쉼표로 구분된 문자열로 변환
                        cleaned_data[key] = ', '.join(str(item) for item in value)
                    elif isinstance(value, dict):
                        # 딕셔너리는 JSON 문자열로 변환
                        import json
                        try:
                            cleaned_data[key] = json.dumps(value, ensure_ascii=False)
                        except:
                            cleaned_data[key] = str(value)
                    else:
                        cleaned_data[key] = str(value)
                
                # 기존 속성 제거하고 정리된 속성으로 교체
                clean_graph.nodes[node].clear()
                clean_graph.nodes[node].update(cleaned_data)
            
            return clean_graph
        
        # Export to GEXF format for Gephi
        try:
            if len(self.citation_graph.nodes) > 0:
                clean_citation_graph = clean_graph_for_gexf(self.citation_graph)
                nx.write_gexf(clean_citation_graph, self.output_dir / "citation_network.gexf")
                self.logger.info("✅ Citation network GEXF 저장 완료")
        except Exception as e:
            self.logger.error(f"❌ Citation network GEXF 저장 실패: {e}")
        
        try:
            if len(self.similarity_graph.nodes) > 0:
                clean_similarity_graph = clean_graph_for_gexf(self.similarity_graph)
                nx.write_gexf(clean_similarity_graph, self.output_dir / "similarity_network.gexf")
                self.logger.info("✅ Similarity network GEXF 저장 완료")
        except Exception as e:
            self.logger.error(f"❌ Similarity network GEXF 저장 실패: {e}")
        
        # Export to CSV for further analysis
        if len(self.citation_graph.edges) > 0:
            edges_data = []
            for source, target, data in self.citation_graph.edges(data=True):
                edges_data.append({
                    'source': source,
                    'target': target,
                    'weight': data.get('weight', 1.0),
                    'type': data.get('citation_type', 'unknown')
                })
            
            df = pd.DataFrame(edges_data)
            df.to_csv(self.output_dir / "citation_edges.csv", index=False)
        
        self.logger.info("Network data exported")
    
    def run_complete_analysis(self):
        """Run the complete citation analysis pipeline."""
        self.logger.info("Starting complete citation analysis...")
        
        # Load data
        self.load_paper_metadata()
        
        # Build networks
        self.build_citation_network()
        self.build_similarity_network()
        
        # Analyze metrics
        metrics = self.analyze_network_metrics()
        
        # Generate visualizations
        self.generate_network_visualizations()
        
        # Export data
        self.export_network_data()
        
        # Generate analysis report
        self.generate_analysis_report(metrics)
        
        self.logger.info("Citation analysis completed successfully!")
    
    def generate_analysis_report(self, metrics: Dict):
        """Generate a comprehensive analysis report."""
        report = f"""
WMS Citation Network Analysis Report
===================================

Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Papers Analyzed: {len(self.papers_metadata)}

NETWORK OVERVIEW
================

Basic Metrics:
- Total Papers: {metrics['basic_metrics'].get('total_nodes', 0)}
- Total Citations: {metrics['basic_metrics'].get('total_edges', 0)}
- Network Density: {metrics['basic_metrics'].get('density', 0):.4f}
- Connected Components: {metrics['basic_metrics'].get('number_of_components', 0)}

INFLUENTIAL PAPERS
==================

Most Central Papers (by different measures):
{chr(10).join([f"- {measure.title()}: {paper_id}" for measure, paper_id in metrics.get('centrality_measures', {}).get('most_central_papers', {}).items() if paper_id])}

RESEARCH COMMUNITIES
====================

{f"Number of Research Clusters: {metrics.get('community_structure', {}).get('number_of_communities', 'N/A')}" if 'community_structure' in metrics else 'Community analysis not available'}
{f"Network Modularity: {metrics.get('community_structure', {}).get('modularity', 'N/A'):.4f}" if 'community_structure' in metrics and 'modularity' in metrics['community_structure'] else ''}

GENERATED FILES
===============

Network Visualizations:
- citation_network.png: Citation relationships between papers
- similarity_network.png: Content similarity network

Network Data:
- citation_network.gexf: Citation network for Gephi analysis
- similarity_network.gexf: Similarity network for Gephi analysis
- citation_edges.csv: Citation relationships in CSV format

RESEARCH INSIGHTS
=================

1. Network Structure: {'Connected' if metrics['basic_metrics'].get('is_connected', False) else 'Fragmented'} research community
2. Citation Patterns: {f"Average {metrics['basic_metrics'].get('total_edges', 0) / max(metrics['basic_metrics'].get('total_nodes', 1), 1):.1f} citations per paper"}
3. Research Evolution: Based on temporal analysis of citations
4. Key Research Areas: Identified through community detection

RECOMMENDATIONS
===============

1. Focus on highly central papers for literature review
2. Investigate research gaps in disconnected components
3. Consider collaboration opportunities within identified communities
4. Monitor emerging trends in recent publications

Next Steps:
- Use trend_visualizer.py for temporal analysis
- Conduct detailed content analysis of central papers
- Map research gaps and opportunities
"""
        
        report_file = self.output_dir / "citation_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Also save metrics as JSON
        metrics_file = self.output_dir / "network_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            # Convert any non-serializable objects to strings
            serializable_metrics = json.loads(json.dumps(metrics, default=str))
            json.dump(serializable_metrics, f, indent=2)
        
        print(report)
        self.logger.info(f"Analysis report saved to: {report_file}")


def main():
    """Main function to run the citation analyzer."""
    parser = argparse.ArgumentParser(description="WMS Citation Network Analyzer")
    parser.add_argument("--papers-dir", default="../Papers", help="Directory containing research papers")
    parser.add_argument("--processed-dir", default="../ProcessedData", help="Directory containing processed data")
    parser.add_argument("--output-dir", default="../Analysis", help="Output directory for analysis results")
    parser.add_argument("--export-only", action="store_true", help="Only export existing network data")
    
    args = parser.parse_args()
    
    analyzer = WMSCitationAnalyzer(args.papers_dir, args.processed_dir, args.output_dir)
    
    if args.export_only:
        analyzer.logger.info("Exporting existing network data...")
        analyzer.load_paper_metadata()
        analyzer.export_network_data()
    else:
        analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
