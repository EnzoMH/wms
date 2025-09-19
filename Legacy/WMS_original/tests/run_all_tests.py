#!/usr/bin/env python3
"""
WMS 전체 테스트 실행기
=====================

WMS 프로젝트의 모든 테스트를 실행하는 메인 테스트 런너입니다.

작성자: 신명호
날짜: 2025년 9월 3일
"""

import os
import sys
import unittest
import time
from pathlib import Path

def discover_and_run_tests():
    """테스트를 찾아서 실행합니다."""
    
    print("=" * 60)
    print("WMS 프로젝트 전체 테스트 실행")
    print("=" * 60)
    print()
    
    # 현재 디렉토리를 테스트 디렉토리로 설정
    test_dir = Path(__file__).parent
    os.chdir(test_dir)
    
    # 테스트 발견 및 실행
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    
    result = runner.run(suite)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 결과 요약
    print()
    print("=" * 60)
    print("테스트 실행 결과 요약")
    print("=" * 60)
    print(f"실행 시간: {execution_time:.2f}초")
    print(f"총 테스트 수: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"에러: {len(result.errors)}")
    print(f"성공률: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "N/A")
    
    # 실패 및 에러 상세 정보
    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}")
            print(f"  {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print("\n에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(f"  {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    print()
    if result.wasSuccessful():
        print("✅ 모든 테스트가 성공적으로 완료되었습니다!")
        return 0
    else:
        print("❌ 일부 테스트가 실패했습니다. 위 내용을 확인해주세요.")
        return 1


def run_individual_test(test_module):
    """개별 테스트 모듈을 실행합니다."""
    
    print(f"=" * 60)
    print(f"{test_module} 테스트 실행")
    print(f"=" * 60)
    
    try:
        # 모듈 임포트
        module = __import__(test_module)
        
        # 테스트 로더 생성
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        
        # 테스트 실행
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"테스트 모듈을 임포트할 수 없습니다: {e}")
        return False
    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")
        return False


def check_dependencies():
    """테스트 실행에 필요한 의존성을 확인합니다."""
    
    print("의존성 확인 중...")
    
    required_packages = [
        'unittest',
        'pandas',
        'numpy',
        'networkx'
    ]
    
    optional_packages = [
        'matplotlib',
        'seaborn', 
        'plotly',
        'nltk',
        'scikit-learn',
        'PyPDF2',
        'arxiv',
        'scholarly'
    ]
    
    missing_required = []
    missing_optional = []
    
    # 필수 패키지 확인
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(package)
    
    # 선택적 패키지 확인
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    # 결과 출력
    if missing_required:
        print(f"❌ 필수 패키지가 누락되었습니다: {', '.join(missing_required)}")
        print("설치 명령: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"⚠️  일부 선택적 패키지가 누락되었습니다: {', '.join(missing_optional)}")
        print("일부 테스트가 건너뛰어질 수 있습니다.")
        print("전체 설치 명령: pip install " + " ".join(missing_optional))
    
    print("✅ 기본 의존성 확인 완료")
    return True


def print_help():
    """도움말을 출력합니다."""
    
    print("""
WMS 테스트 실행기 사용법
======================

기본 사용법:
    python run_all_tests.py                 # 모든 테스트 실행
    python run_all_tests.py --help          # 도움말 출력
    python run_all_tests.py --check-deps    # 의존성 확인
    python run_all_tests.py --single MODULE # 개별 테스트 실행

개별 테스트 모듈:
    test_paper_scraper      # 논문 수집기 테스트
    test_text_extractor     # 텍스트 추출기 테스트  
    test_citation_analyzer  # 인용 분석기 테스트
    test_trend_visualizer   # 트렌드 시각화기 테스트

예제:
    python run_all_tests.py --single test_paper_scraper
    python run_all_tests.py --check-deps

참고사항:
- 일부 테스트는 인터넷 연결이나 외부 API가 필요할 수 있습니다
- 시각화 관련 테스트는 GUI가 없는 환경에서 건너뛰어질 수 있습니다
- 성능 테스트는 시스템 사양에 따라 결과가 달라질 수 있습니다
""")


def main():
    """메인 함수"""
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg in ['--help', '-h']:
            print_help()
            return 0
            
        elif arg == '--check-deps':
            if check_dependencies():
                return 0
            else:
                return 1
                
        elif arg == '--single' and len(sys.argv) > 2:
            test_module = sys.argv[2]
            if run_individual_test(test_module):
                return 0
            else:
                return 1
        else:
            print(f"알 수 없는 인수: {arg}")
            print("--help를 사용하여 도움말을 확인하세요.")
            return 1
    
    # 기본 동작: 모든 테스트 실행
    else:
        # 의존성 확인
        if not check_dependencies():
            print("의존성 문제로 인해 테스트를 실행할 수 없습니다.")
            return 1
        
        print()
        return discover_and_run_tests()


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
