#!/usr/bin/env python3
"""
벡터DB 업데이트 스크립트
"""
import subprocess
import sys
from pathlib import Path

# 프로젝트 루트로 이동
project_root = Path(__file__).parent.parent.parent
print(f"프로젝트 루트: {project_root}")

print("="*60)
print(" Faiss 벡터DB 업데이트")
print("="*60)
print("\n선택하세요:")
print("  1. 증분 업데이트 (새 데이터만 추가, 기존 유지) - 권장")
print("  2. 전체 재구축 (기존 데이터 삭제)")
print("  3. 취소")

choice = input("\n선택 (1/2/3): ").strip()

if choice == '1':
    # 증분 업데이트
    print("\n [STAGE 1] 1단계: 뉴스 청킹...")
    result = subprocess.run([
        sys.executable, 
        str(project_root / "0_core/1_extractors/news_to_chunks.py"),
        "--news-dir", str(project_root / "1_data/0_news"),
        "--output-dir", str(project_root / "1_data/1_chunks")
    ], cwd=str(project_root))
    
    if result.returncode == 0:
        print("\n [SUCCESS] 뉴스 청킹 완료")
        
        print("\n [STAGE 2] 2단계: 벡터DB 증분 업데이트...")
        result = subprocess.run([
            sys.executable,
            str(project_root / "0_core/vectorDB/faiss_builder.py"),
            "--action", "update",
            "--processed-data", "1_data/1_chunks",
            "--vector-db", "2_vecdb"
        ], cwd=str(project_root))
        
        if result.returncode == 0:
            print("\n [SUCCESS] 벡터DB 증분 업데이트 완료!")
            print("\n [TEST] RAG 테스트:")
            print("   python test_rag_faiss.py")
        else:
            print("\n [ERROR] 벡터DB 업데이트 실패")
    else:
        print("\n [ERROR] 뉴스 청킹 실패")

elif choice == '2':
    # 전체 재구축
    confirm = input("\n [WARNING]  기존 벡터DB를 삭제합니다. 계속하시겠습니까? (y/N): ")
    
    if confirm.lower() == 'y':
        print("\n [STAGE 1] 1단계: 뉴스 청킹...")
        result = subprocess.run([
            sys.executable, 
            str(project_root / "0_core/1_extractors/news_to_chunks.py"),
            "--news-dir", str(project_root / "1_data/0_news"),
            "--output-dir", str(project_root / "1_data/1_chunks")
        ], cwd=str(project_root))
        
        if result.returncode == 0:
            print("\n [SUCCESS] 뉴스 청킹 완료")
            
            print("\n [STAGE 2] 2단계: 벡터DB 전체 재구축...")
            result = subprocess.run([
                sys.executable,
                str(project_root / "0_core/vectorDB/faiss_builder.py"),
                "--action", "build",
                "--processed-data", "1_data/1_chunks",
                "--vector-db", "2_vecdb"
            ], cwd=str(project_root))
            
            if result.returncode == 0:
                print("\n [SUCCESS] 벡터DB 재구축 완료!")
                print("\n [TEST] RAG 테스트:")
                print("   python test_rag_faiss.py")
            else:
                print("\n [ERROR] 벡터DB 재구축 실패")
        else:
            print("\n [ERROR] 뉴스 청킹 실패")
    else:
        print("\n[CANCEL] 취소됨")

else:
    print("\n[CANCEL] 취소됨")

