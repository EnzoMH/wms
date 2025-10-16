#!/usr/bin/env python3
"""
Advanced RAG 청크 최적화 스크립트
논문과 뉴스 청크를 모두 처리
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_chunk(chunk, enable_graph=False, enable_linking=False, enable_grouping=False):
    """
    청크 최적화 수행 (현재는 기본 메타데이터만 추가)
    향후 그래프 강화, 엔터티 링킹, 의미적 그룹화 구현 예정
    """
    optimized = chunk.copy()
    
    # 최적화 플래그 메타데이터 추가
    if 'metadata' not in optimized:
        optimized['metadata'] = {}
    
    optimized['metadata']['optimized'] = True
    optimized['metadata']['graph_enhanced'] = enable_graph
    optimized['metadata']['entity_linked'] = enable_linking
    optimized['metadata']['semantic_grouped'] = enable_grouping
    
    return optimized

def process_paper_chunks(input_file, output_file, args):
    """논문 청크 파일 처리"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        
        # 청크 최적화
        optimized_chunks = [
            optimize_chunk(
                chunk, 
                enable_graph=args.graph_enhancement,
                enable_linking=args.entity_linking,
                enable_grouping=args.semantic_grouping
            ) 
            for chunk in chunks
        ]
        
        # 데이터 구조 유지
        data['chunks'] = optimized_chunks
        data['optimization_applied'] = True
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  [OK] 논문: {input_file.name} -> {len(optimized_chunks)}개 청크")
        return True
        
    except Exception as e:
        logger.error(f"  [X] 논문 처리 실패 ({input_file.name}): {e}")
        return False

def process_news_chunks(input_file, output_file, args):
    """뉴스 청크 파일 처리"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 뉴스 청크는 리스트 형태
        if isinstance(data, list):
            chunks = data
        else:
            chunks = data.get('chunks', [])
        
        # 청크 최적화
        optimized_chunks = [
            optimize_chunk(
                chunk, 
                enable_graph=args.graph_enhancement,
                enable_linking=args.entity_linking,
                enable_grouping=args.semantic_grouping
            ) 
            for chunk in chunks
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(optimized_chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  [OK] 뉴스: {input_file.name} -> {len(optimized_chunks)}개 청크")
        return True
        
    except Exception as e:
        logger.error(f"  [X] 뉴스 처리 실패 ({input_file.name}): {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="청크 최적화 (논문 + 뉴스)")
    parser.add_argument("--input-dir", required=True, help="입력 청크 디렉토리")
    parser.add_argument("--output-dir", required=True, help="출력 청크 디렉토리")
    parser.add_argument("--graph-enhancement", action="store_true", help="그래프 구조 정보 추가")
    parser.add_argument("--entity-linking", action="store_true", help="엔터티 연결")
    parser.add_argument("--semantic-grouping", action="store_true", help="의미적 그룹화")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("청크 최적화 시작")
    logger.info("="*60)
    logger.info(f"입력: {input_dir}")
    logger.info(f"출력: {output_dir}")
    logger.info(f"그래프 강화: {args.graph_enhancement}")
    logger.info(f"엔터티 링킹: {args.entity_linking}")
    logger.info(f"의미적 그룹화: {args.semantic_grouping}")
    logger.info("")
    
    paper_success = 0
    paper_fail = 0
    news_success = 0
    news_fail = 0
    
    # 1. 논문 청크 파일 처리 (chunks_*.json)
    if input_dir.exists():
        paper_files = list(input_dir.glob("chunks_*.json"))
        logger.info(f"[1] 논문 청크 파일: {len(paper_files)}개")
        
        for paper_file in paper_files:
            output_file = output_dir / paper_file.name
            if process_paper_chunks(paper_file, output_file, args):
                paper_success += 1
            else:
                paper_fail += 1
        
        # 2. 뉴스 청크 파일 처리 (News/*_chunks.json)
        news_dir = input_dir / "news"
        if news_dir.exists():
            news_files = list(news_dir.glob("*_chunks.json"))
            logger.info(f"\n[2] 뉴스 청크 파일: {len(news_files)}개")
            
            # 출력 뉴스 디렉토리 생성
            news_output_dir = output_dir / "news"
            news_output_dir.mkdir(parents=True, exist_ok=True)
            
            for news_file in news_files:
                output_file = news_output_dir / news_file.name
                if process_news_chunks(news_file, output_file, args):
                    news_success += 1
                else:
                    news_fail += 1
        else:
            logger.info("\n[2] 뉴스 청크 파일: 없음")
    
    # 결과 요약
    logger.info("")
    logger.info("="*60)
    logger.info("청크 최적화 완료")
    logger.info("="*60)
    logger.info(f"논문: {paper_success}개 성공, {paper_fail}개 실패")
    logger.info(f"뉴스: {news_success}개 성공, {news_fail}개 실패")
    logger.info(f"총: {paper_success + news_success}개 처리 완료")
    logger.info("")
            
if __name__ == "__main__":
    main()
