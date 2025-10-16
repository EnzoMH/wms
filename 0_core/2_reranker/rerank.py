#!/usr/bin/env python3
"""
    
============================

BGE (BAAI General Embedding)  
      .

: 
: 2025 10 11
: 1.0.0
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import argparse

#  
try:
    from sentence_transformers import CrossEncoder
    import torch
    
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError as e:
    print(f"  . : {e}")
    print(": pip install sentence-transformers torch")
    exit(1)


class ChunkReranker:
    """       """
    
    def __init__(self, 
                 chunks_dir: str = "../../1_data/1_chunks",
                 output_dir: str = "../../1_data/1_chunks_filtered",
                 model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
         .
        
        Args:
            chunks_dir:     
            output_dir:    
            model_name:   
        """
        self.chunks_dir = Path(chunks_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        self.setup_logging()
        self.setup_directories()
        self.load_reranker_model()
        
        #      (  )
        self.reference_queries = [
            "AGV   ",
            "   ",
            "  ",
            "  ",
            "  ",
            "RTV   ",
            "  ",
            "  ",
            "  ",
            "IoT   "
        ]
    
    def setup_logging(self):
        """ ."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('reranker.log', encoding='utf-8', errors='ignore'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """  ."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  : {self.output_dir}")
    
    def load_reranker_model(self):
        """BGE   ."""
        self.logger.info(f"   : {self.model_name}")
        
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        self.logger.info(f" : {device}")
        
        try:
            self.reranker = CrossEncoder(
                self.model_name,
                max_length=512,
                device=device
            )
            
            self.logger.info(f"[SUCCESS]    ")
            self.logger.info(f"   : {self.model_name}")
            self.logger.info(f"   : {device}")
            
        except Exception as e:
            self.logger.error(f"[ERROR]    : {e}")
            raise
    
    def calculate_chunk_importance(self, chunk_text: str) -> float:
        """
          .
        
        Args:
            chunk_text:  
            
        Returns:
              (0~1)
        """
        #       
        pairs = [[query, chunk_text] for query in self.reference_queries]
        
        try:
            scores = self.reranker.predict(pairs)
            
            #     ( )
            normalized_scores = 1 / (1 + np.exp(-np.array(scores)))
            avg_score = float(np.mean(normalized_scores))
            
            return avg_score
            
        except Exception as e:
            self.logger.warning(f"  : {e}")
            return 0.0
    
    def filter_chunks_by_importance(self, 
                                    chunks: List[Dict], 
                                    threshold: float = 0.5,
                                    top_k_percent: float = 0.7) -> Tuple[List[Dict], Dict]:
        """
           .
        
        Args:
            chunks:  
            threshold:   
            top_k_percent:     (0.7 =  70%)
            
        Returns:
            (  ,  )
        """
        self.logger.info(f"   : {len(chunks)} ")
        
        #    
        scored_chunks = []
        for i, chunk in enumerate(chunks):
            if (i + 1) % 10 == 0:
                self.logger.info(f"  : {i + 1}/{len(chunks)}")
            
            importance_score = self.calculate_chunk_importance(chunk['content'])
            
            chunk_with_score = {
                **chunk,
                'importance_score': importance_score
            }
            scored_chunks.append(chunk_with_score)
        
        #   
        scored_chunks.sort(key=lambda x: x['importance_score'], reverse=True)
        
        #  :  OR  k%
        threshold_filtered = [c for c in scored_chunks if c['importance_score'] >= threshold]
        top_k_count = max(int(len(chunks) * top_k_percent), 1)
        top_k_filtered = scored_chunks[:top_k_count]
        
        #        
        filtered_chunks = threshold_filtered if len(threshold_filtered) >= len(top_k_filtered) else top_k_filtered
        
        #  (chunk id) 
        filtered_chunks.sort(key=lambda x: x['id'])
        
        #  
        stats = {
            'total_chunks': len(chunks),
            'filtered_chunks': len(filtered_chunks),
            'filter_ratio': len(filtered_chunks) / len(chunks) if chunks else 0,
            'avg_score_original': float(np.mean([c['importance_score'] for c in scored_chunks])),
            'avg_score_filtered': float(np.mean([c['importance_score'] for c in filtered_chunks])) if filtered_chunks else 0,
            'min_score': float(min([c['importance_score'] for c in filtered_chunks])) if filtered_chunks else 0,
            'max_score': float(max([c['importance_score'] for c in filtered_chunks])) if filtered_chunks else 0
        }
        
        self.logger.info(f"[SUCCESS]   :")
        self.logger.info(f"   : {stats['total_chunks']}")
        self.logger.info(f"    : {stats['filtered_chunks']} ({stats['filter_ratio']*100:.1f}%)")
        self.logger.info(f"    : {stats['avg_score_original']:.3f} → {stats['avg_score_filtered']:.3f}")
        
        return filtered_chunks, stats
    
    def process_all_chunk_files(self, threshold: float = 0.5, top_k_percent: float = 0.7):
        """    ."""
        self.logger.info("=" * 60)
        self.logger.info("     ")
        self.logger.info("=" * 60)
        
        # 논문 청크 파일 수집
        chunk_files = list(self.chunks_dir.glob("chunks_*.json"))
        self.logger.info(f"  : {len(chunk_files)}")
        
        # 뉴스 청크 파일 수집 (News/ 하위 디렉토리)
        news_dir = self.chunks_dir / "news"
        news_files = list(news_dir.glob("*_chunks.json")) if news_dir.exists() else []
        self.logger.info(f"   : {len(news_files)}")
        
        # 전체 파일 목록 통합
        all_files = chunk_files + news_files
        
        if not all_files:
            self.logger.error("[ERROR]    !")
            return
        
        all_stats = []
        
        for i, chunk_file in enumerate(all_files, 1):
            is_news = chunk_file.parent.name == "news"
            file_type = "" if not is_news else ""
            
            self.logger.info(f"\n[{i}/{len(all_files)}] {file_type}  : {chunk_file.name}")
            
            try:
                #   
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 뉴스와 논문의 구조 차이 처리
                if is_news:
                    # 뉴스: 리스트 형태 [{'content': ..., 'metadata': ...}, ...]
                    if isinstance(data, list):
                        chunks = data
                    else:
                        chunks = data.get('chunks', [])
                    
                    # 뉴스 청크에 id 추가 (없는 경우)
                    for idx, chunk in enumerate(chunks):
                        if 'id' not in chunk:
                            chunk['id'] = chunk.get('metadata', {}).get('chunk_index', idx)
                else:
                    # 논문: 딕셔너리 형태 {'chunks': [...], 'source': ..., ...}
                    chunks = data.get('chunks', [])
                
                if not chunks:
                    self.logger.warning(f"   : {chunk_file.name}")
                    continue
                
                #  
                filtered_chunks, stats = self.filter_chunks_by_importance(
                    chunks, 
                    threshold=threshold,
                    top_k_percent=top_k_percent
                )
                
                #   
                if is_news:
                    # 뉴스는 리스트 형태 그대로 저장
                    filtered_data = filtered_chunks
                else:
                    # 논문은 메타데이터 포함
                    filtered_data = {
                        'source': data.get('source'),
                        'filename': data.get('filename'),
                        'total_chars': data.get('total_chars'),
                        'original_chunks': len(chunks),
                        'filtered_chunks': len(filtered_chunks),
                        'filter_ratio': stats['filter_ratio'],
                        'reranking_stats': stats,
                        'chunks': filtered_chunks
                    }
                
                # 저장 경로 결정
                if is_news:
                    news_output_dir = self.output_dir / "news"
                    news_output_dir.mkdir(parents=True, exist_ok=True)
                    output_file = news_output_dir / chunk_file.name
                else:
                    output_file = self.output_dir / chunk_file.name
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"  [SUCCESS]  : {output_file.name}")
                
                all_stats.append({
                    'filename': chunk_file.name,
                    'type': 'news' if is_news else 'paper',
                    **stats
                })
                
            except Exception as e:
                self.logger.error(f"[ERROR] {chunk_file.name}  : {e}")
                continue
        
        #   
        self.save_summary_stats(all_stats)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("[DONE]   !")
        self.logger.info("=" * 60)
    
    def save_summary_stats(self, stats_list: List[Dict]):
        """  ."""
        if not stats_list:
            return
        
        summary = {
            'generation_time': datetime.now().isoformat(),
            'total_papers': len(stats_list),
            'total_original_chunks': sum(s['total_chunks'] for s in stats_list),
            'total_filtered_chunks': sum(s['filtered_chunks'] for s in stats_list),
            'overall_filter_ratio': sum(s['filtered_chunks'] for s in stats_list) / sum(s['total_chunks'] for s in stats_list),
            'avg_importance_score': np.mean([s['avg_score_filtered'] for s in stats_list]),
            'per_paper_stats': stats_list
        }
        
        summary_file = self.output_dir / "reranking_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"\n[STATS]  :")
        self.logger.info(f"   : {summary['total_papers']}")
        self.logger.info(f"   : {summary['total_original_chunks']}")
        self.logger.info(f"   : {summary['total_filtered_chunks']}")
        self.logger.info(f"   : {summary['overall_filter_ratio']*100:.1f}%")
        self.logger.info(f"   : {summary['avg_importance_score']:.3f}")
        self.logger.info(f"\n[SAVE]  : {summary_file}")


def main():
    """   """
    parser = argparse.ArgumentParser(description="    ")
    parser.add_argument("--chunks-dir", default="../../1_data/1_chunks",
                       help="  ")
    parser.add_argument("--output-dir", default="../../1_data/1_chunks_filtered",
                       help="   ")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="   (0~1)")
    parser.add_argument("--top-k-percent", type=float, default=0.7,
                       help="    (0~1)")
    
    args = parser.parse_args()
    
    #    
    reranker = ChunkReranker(
        chunks_dir=args.chunks_dir,
        output_dir=args.output_dir
    )
    
    reranker.process_all_chunk_files(
        threshold=args.threshold,
        top_k_percent=args.top_k_percent
    )


if __name__ == "__main__":
    main()

