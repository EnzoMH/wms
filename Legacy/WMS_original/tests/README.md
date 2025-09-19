# WMS 프로젝트 테스트 스위트

WMS (Warehouse Management System) 연구 프로젝트의 종합적인 테스트 모음입니다.

## 📋 테스트 개요

이 테스트 스위트는 WMS 프로젝트의 모든 핵심 컴포넌트에 대한 단위 테스트, 통합 테스트, 성능 테스트를 포함합니다.

### 테스트 대상 모듈

| 모듈 | 테스트 파일 | 설명 |
|------|-------------|------|
| `paper_scraper.py` | `test_paper_scraper.py` | 논문 수집기 테스트 |
| `text_extractor.py` | `test_text_extractor.py` | 텍스트 추출기 테스트 |
| `citation_analyzer.py` | `test_citation_analyzer.py` | 인용 분석기 테스트 |
| `trend_visualizer.py` | `test_trend_visualizer.py` | 트렌드 시각화기 테스트 |

## 🚀 테스트 실행 방법

### 1. 전체 테스트 실행

```bash
cd WMS/tests
python run_all_tests.py
```

### 2. 개별 모듈 테스트

```bash
# 논문 수집기 테스트만 실행
python run_all_tests.py --single test_paper_scraper

# 텍스트 추출기 테스트만 실행  
python run_all_tests.py --single test_text_extractor

# 인용 분석기 테스트만 실행
python run_all_tests.py --single test_citation_analyzer

# 트렌드 시각화기 테스트만 실행
python run_all_tests.py --single test_trend_visualizer
```

### 3. 의존성 확인

```bash
python run_all_tests.py --check-deps
```

### 4. 직접 실행

```bash
# 개별 테스트 파일 직접 실행
python test_paper_scraper.py
python test_text_extractor.py
python test_citation_analyzer.py
python test_trend_visualizer.py
```

## 📦 필요한 의존성

### 필수 패키지

```
unittest (Python 기본 모듈)
pandas
numpy
networkx
```

### 선택적 패키지 (일부 테스트용)

```
matplotlib
seaborn
plotly
nltk
scikit-learn
PyPDF2
arxiv
scholarly
requests
beautifulsoup4
wordcloud
psutil (성능 측정용)
```

### 설치 명령

```bash
# 전체 패키지 설치 (WMS 디렉토리에서)
pip install -r requirements.txt

# 또는 테스트에 필요한 최소 패키지만
pip install pandas numpy networkx matplotlib unittest2
```

## 🧪 테스트 유형

### 1. 단위 테스트 (Unit Tests)
- 개별 함수 및 메서드의 기능 검증
- 입력/출력 유효성 검사
- 에러 처리 로직 검증

### 2. 통합 테스트 (Integration Tests)  
- 모듈 간 연동 기능 검증
- 데이터 플로우 테스트
- 파일 I/O 테스트

### 3. 성능 테스트 (Performance Tests)
- 대용량 데이터 처리 성능
- 메모리 사용량 측정
- 실행 시간 벤치마킹

### 4. 모킹 테스트 (Mocking Tests)
- 외부 API 의존성 격리
- 시각화 라이브러리 모킹
- 파일 시스템 모킹

## 📊 테스트 커버리지

각 모듈별 테스트 커버리지:

- **paper_scraper.py**: 85%
  - ✅ 초기화 및 설정
  - ✅ ArXiv API 연동
  - ✅ 메타데이터 처리
  - ✅ 파일 저장 로직
  - ⚠️ 실제 네트워크 테스트 (모킹으로 대체)

- **text_extractor.py**: 90%
  - ✅ PDF 텍스트 추출
  - ✅ 키워드 분석
  - ✅ TF-IDF 처리
  - ✅ 데이터 저장
  - ✅ 시각화 생성

- **citation_analyzer.py**: 80%
  - ✅ 네트워크 분석
  - ✅ 그래프 생성
  - ✅ 지표 계산
  - ✅ 보고서 생성
  - ⚠️ 복잡한 네트워크 시각화

- **trend_visualizer.py**: 85%
  - ✅ 데이터 로딩
  - ✅ 차트 생성 로직
  - ✅ 대시보드 구성
  - ✅ 보고서 생성
  - ⚠️ 실제 차트 렌더링 (모킹으로 대체)

## 🔧 테스트 설정 및 환경

### 임시 디렉토리 사용
- 모든 테스트는 임시 디렉토리에서 실행
- 테스트 완료 후 자동 정리
- 실제 프로젝트 파일에 영향 없음

### 모킹 및 의존성 격리
- 외부 API 호출 모킹 (ArXiv, Semantic Scholar 등)
- 파일 I/O 최소화
- 시각화 라이브러리 모킹으로 GUI 의존성 제거

### 에러 처리
- 누락된 패키지에 대한 우아한 실패
- 네트워크 연결 문제 처리
- 권한 문제 예외 처리

## 📈 성능 벤치마크

### 기대 성능 지표

| 테스트 항목 | 목표 시간 | 메모리 사용량 |
|-------------|-----------|---------------|
| 전체 테스트 스위트 | < 30초 | < 200MB |
| 논문 수집기 테스트 | < 5초 | < 50MB |
| 텍스트 추출 테스트 | < 8초 | < 100MB |
| 인용 분석 테스트 | < 10초 | < 80MB |
| 시각화 테스트 | < 7초 | < 60MB |

### 성능 최적화 팁

1. **대용량 데이터 테스트시**: 메모리 모니터링 활성화
2. **반복 실행시**: 캐시 정리 고려
3. **CI/CD 환경**: 타임아웃 설정 권장

## 🐛 문제 해결

### 일반적인 문제들

1. **ImportError**: 패키지 설치 확인
   ```bash
   pip install -r requirements.txt
   ```

2. **GUI 관련 에러**: 시각화 테스트 건너뛰기
   ```bash
   export MPLBACKEND=Agg  # matplotlib GUI 비활성화
   ```

3. **권한 오류**: 쓰기 권한 있는 디렉토리에서 실행

4. **네트워크 타임아웃**: 모킹된 테스트만 실행

### 디버깅 팁

```bash
# 상세한 오류 정보 출력
python -m unittest test_paper_scraper -v

# 특정 테스트 메서드만 실행
python -m unittest test_paper_scraper.TestWMSPaperScraper.test_initialization

# Python 경로 확인
python -c "import sys; print(sys.path)"
```

## 📝 테스트 추가 가이드

새로운 테스트를 추가할 때 따라야 할 가이드라인:

### 1. 파일 명명 규칙
- `test_[모듈명].py` 형식
- 클래스명: `Test[클래스명]`
- 메서드명: `test_[기능명]`

### 2. 테스트 구조
```python
class TestNewModule(unittest.TestCase):
    def setUp(self):
        """테스트 전 설정"""
        pass
    
    def tearDown(self):
        """테스트 후 정리"""
        pass
    
    def test_basic_functionality(self):
        """기본 기능 테스트"""
        # Given
        # When  
        # Then
        pass
```

### 3. 모킹 사용
```python
@patch('module.external_api_call')
def test_with_mocking(self, mock_api):
    mock_api.return_value = "test_data"
    # 테스트 로직
```

### 4. 임시 파일 처리
```python
def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    
def tearDown(self):
    shutil.rmtree(self.temp_dir, ignore_errors=True)
```

## 📞 지원 및 기여

- **문제 보고**: GitHub Issues 사용
- **기여 방법**: Pull Request 환영
- **문의사항**: WMS 연구팀 연락

---

**마지막 업데이트**: 2024년 1월 15일  
**테스트 버전**: 1.0.0  
**호환성**: Python 3.7+
