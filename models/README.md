# Ollama Modelfiles

WMS RAG 시스템용 Ollama 모델 설정 파일들

## 사용 가능한 모델

### Modelfile.exaone-1.2b

**EXAONE 4.0 1.2B** - 일반 대화 + RAG 최적화

- **크기:** 1.6GB (Q4_K_M)
- **용도:** 일반 대화, RAG, 가벼운 추론
- **GPU 메모리:** 2-4GB
- **속도:** 빠름

**생성:**
```bash
ollama create exaone-rag -f models/Modelfile.exaone-1.2b
```

**실행:**
```bash
ollama run exaone-rag
```

**WMS API에서 사용:**
```bash
$env:USE_OLLAMA="true"
$env:OLLAMA_MODEL="exaone-rag"
python main.py
```

## 새 모델 추가

### 1. Modelfile 생성

예: `Modelfile.exaone-32b`

```dockerfile
FROM hf.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF:Q4_K_M

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM """당신은 VisionSpace에서 훈련시킨 창고 자동화 전문가입니다."""
```

### 2. 모델 생성

```bash
ollama create exaone-32b -f models/Modelfile.exaone-32b
```

### 3. 사용

```bash
ollama run exaone-32b
```

## 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| temperature | 0.1 | 창의성 (0=결정적, 2=랜덤) |
| top_p | 0.9 | 누적 확률 샘플링 |
| top_k | 40 | 상위 K개 토큰만 고려 |
| num_ctx | 4096 | 컨텍스트 윈도우 크기 |
| repeat_penalty | 1.1 | 반복 방지 (1.0=없음) |

## 모델 관리

```bash
# 모델 목록
ollama list

# 모델 삭제
ollama rm exaone-rag

# 모델 정보
ollama show exaone-rag

# 모델 업데이트
ollama create exaone-rag -f models/Modelfile.exaone-1.2b
```

## 참고

- 자세한 설정은 `OLLAMA_SETUP.md` 참조
- Langchain 통합은 `LANGCHAIN_GUIDE.md` 참조

