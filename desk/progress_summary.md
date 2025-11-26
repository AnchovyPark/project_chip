
# LLM 커널 지연시간 프로파일링 프로젝트 - 진행 상황 요약

이 문서는 LLM 추론 시간 예측을 위한 다양한 커널 프로파일링 작업의 진행 상황을 요약합니다.

## 목표
GPU 연산의 성능 특성을 측정하고 분석하여 대규모 언어 모델(LLM) 추론을 예측하는 것이 주요 목표입니다. 프로젝트는 두 가지 주요 구성 요소로 이루어져 있습니다:

### 1. 커널 프로파일링
일반적인 딥러닝 연산의 실행 시간(지연시간) 측정:
1.  완전 연결 계층 (Fully Connected Layers)
2.  합성곱 계층 (Convolutional Layers)
3.  순환 계층 (Recurrent Layers)
4.  메모리 제한 연산 (Memory-Limited Operations)

### 2. Roofline 모델 분석
GPU 하드웨어 한계를 특성화하여 연산 성능 병목 현상 파악:
1.  Peak FLOPS (연산 처리량)
2.  Peak Memory Bandwidth (메모리 대역폭)
3.  Ridge Point (연산 제한 vs 메모리 제한 경계)

이 측정값들을 사용하여 PyTorch를 활용하는 vLLM의 추론 시간을 예측하는 것이 목표입니다.

## 현재 상태 (업데이트됨)

### ✅ 완료된 작업

1. **프로젝트 구조:**
   - `src/kernel_profiling/` 디렉토리 생성
   - `src/roofline_profiling/` 디렉토리 생성
   - 모든 프로파일링 스크립트가 카테고리별로 정리됨

2. **커널 프로파일링 - 4개 스크립트 구현 완료:**
   - ✅ `profile_fc_cuda.py` - 완전 연결(Linear) 계층
   - ✅ `profile_conv_cuda.py` - 2D 합성곱 계층
   - ✅ `profile_rnn_cuda.py` - LSTM 계층
   - ✅ `profile_mem_cuda.py` - 메모리 제한 연산 (LayerNorm + GELU)

3. **Roofline 프로파일링 - 3개 스크립트 구현 완료:**
   - ✅ `measure_peak_flops.py` - FP16 GEMM을 사용한 Peak FLOPS 측정
   - ✅ `measure_peak_bandwidth.py` - Peak 메모리 대역폭 측정
   - ✅ `calculate_ridge_point.py` - Ridge point 계산 및 분석

4. **프로파일링 방법론:**
   - 정확한 GPU 측정을 위해 `torch.cuda.Event` 사용
   - GPU 상태 안정화를 위한 warm-up 실행 (커널: 10회, roofline: 5회)
   - 평균을 위한 여러 번의 프로파일링 실행 (커널: 100회, roofline: 20회)
   - `torch.cuda.synchronize()`를 통한 적절한 동기화

### 📊 코드 평가 결과

#### 강점:
- ✅ **정확한 타이밍 방법론**: CUDA Events를 사용하여 CPU-GPU 동기화 오버헤드 방지
- ✅ **적절한 GPU warm-up**: 주파수 스케일링 및 캐시 효과 처리
- ✅ **통계적 평균화**: 100회 (커널) / 20회 (roofline) 실행으로 노이즈 감소
- ✅ **일관된 구조**: 모든 스크립트가 동일한 패턴을 따름
- ✅ **깔끔한 코드**: 명확한 인터페이스와 함께 잘 문서화됨
- ✅ **Roofline 분석**: 병목 현상 식별을 위한 하드웨어 성능 한계 제공
- ✅ **FP16 최적화**: Roofline 테스트에서 FP16 사용 (최신 GPU 추론에 적합)

#### 확인된 한계점:
1. **통계 지표**: 평균만 측정하고, 표준편차/중앙값/최소값/최대값 누락
2. **고정된 파라미터**: 다양한 크기에 대한 파라미터 스윕 없음
3. **LLM 관련성**:
   - ✅ Linear, LayerNorm, GELU: 매우 관련성 높음
   - ⚠️ Conv2d: LLM에서 거의 사용되지 않음 (Vision 모델용)
   - ⚠️ LSTM: 구식 (최신 LLM은 Transformer 사용)
   - ❌ **누락된 핵심 연산**: Self-Attention / Multi-Head Attention
4. **결과 지속성 없음**: 결과가 출력만 되고 CSV/JSON으로 저장되지 않음
5. **메모리 추적 없음**: GPU 메모리 사용량이 측정되지 않음
6. **커널 레벨 프로파일링**: 상세한 분석을 위한 `torch.profiler` 미사용

### 🎯 개선 권장사항

#### 높은 우선순위 (vLLM 예측용):
1. **Attention 프로파일링 추가** - 가장 중요한 누락 구성 요소:
   ```python
   # torch.nn.MultiheadAttention
   # Scaled Dot-Product Attention
   # Flash Attention (사용하는 경우)
   ```

2. **파라미터 스윕 구현**:
   - 배치 크기: [1, 2, 4, 8, 16, 32]
   - 시퀀스 길이: [128, 256, 512, 1024, 2048]
   - Hidden 차원: [768, 1024, 2048, 4096, 8192]

3. **구조화된 형식으로 결과 저장**:
   ```python
   # CSV 컬럼: operation, batch_size, seq_len, hidden_dim,
   #           avg_latency_ms, std_latency_ms, gpu_memory_mb
   ```

4. **커널의 연산 강도 계산**:
   - 각 연산에 대한 FLOP 수와 메모리 트래픽 측정
   - Roofline 모델에 커널 성능 플롯
   - 어떤 커널이 연산 제한인지 메모리 제한인지 식별

#### 중간 우선순위:
5. **향상된 통계**: 표준편차, 중앙값, 백분위수 계산
6. **메모리 프로파일링**: `torch.cuda.max_memory_allocated()` 추가
7. **추가 LLM 연산**:
   - Softmax
   - Embedding lookups
   - 행렬 곱셈 변형 (GEMM)

#### 낮은 우선순위 (고급):
8. **PyTorch Profiler 통합**: 커널 레벨 분석
9. **vLLM 특정 연산**: PagedAttention, KV cache ops
10. **혼합 정밀도 분석**: FP16, FP32, INT8 성능 비교

## 환경 참고사항
- **개발 환경**: MacBook (CUDA 지원 없음)
- **대상 실행 환경**: NVIDIA GPU가 있는 AWS EC2
- **프레임워크**: PyTorch with CUDA 12.1+
- **Python 버전**: Python 3.x

## 평가: 현재 접근 방식으로 vLLM 지연시간을 예측할 수 있는가?

**짧은 답변:** 개선된 기반이지만, 여전히 더 많은 작업이 필요합니다.

**현재 접근 방식으로 가능한 것:**
- ✅ 개별 연산 비용 이해
- ✅ **신규**: Roofline 모델을 통한 연산 제한 vs 메모리 제한 연산 식별
- ✅ **신규**: GPU 하드웨어 한계 파악 (peak FLOPS, peak bandwidth, ridge point)
- ✅ 다양한 파라미터 구성 비교
- ✅ **신규**: 최적화 목표를 위한 이론적 성능 한계

**현재 접근 방식으로 부족한 것:**
- ❌ Attention 프로파일링 없이 정확한 end-to-end vLLM 예측 불가
- ❌ 연산 융합 및 커널 오버랩 모델링 불가
- ❌ vLLM 특정 최적화 캡처 불가 (continuous batching, PagedAttention)
- ❌ 혼합 정밀도 및 Tensor Core 활용 효과 반영 안 됨
- ❌ Roofline 모델에 실제 커널 성능 플롯 불가 (연산 강도 필요)

**Roofline 추가의 영향:**
- ✅ 이제 커널이 하드웨어 한계에 도달했는지 판단 가능
- ✅ 최적화 기회 식별 가능 (예: 메모리 제한이면 데이터 이동 감소에 집중)
- ✅ GPU가 효율적으로 활용되고 있는지 검증 가능
- ⚠️ Roofline에 플롯하려면 각 커널의 연산 강도(FLOP/Byte) 계산 필요

**정확한 vLLM 예측을 위한 권장 다음 단계:**
1. Attention 연산 프로파일링 추가 (최우선순위)
2. 각 커널의 연산 강도 계산 및 Roofline 모델에 플롯
3. 실제 LLM 모델 크기로 프로파일링 (예: Llama 7B/13B 차원)
4. 실제 vLLM 추론의 end-to-end 프로파일링 고려
5. 연산 프로파일을 추론 시간에 매핑하는 회귀 모델 구축

## 파일 개요

### 커널 프로파일링 스크립트
- `src/kernel_profiling/profile_fc_cuda.py` (66줄) - Linear 계층
- `src/kernel_profiling/profile_conv_cuda.py` (74줄) - Conv2D 계층
- `src/kernel_profiling/profile_rnn_cuda.py` (72줄) - LSTM 계층
- `src/kernel_profiling/profile_mem_cuda.py` (65줄) - LayerNorm + GELU

### Roofline 프로파일링 스크립트
- `src/roofline_profiling/measure_peak_flops.py` (109줄) - Peak FLOPS 측정
- `src/roofline_profiling/measure_peak_bandwidth.py` (109줄) - Peak bandwidth 측정
- `src/roofline_profiling/calculate_ridge_point.py` (87줄) - Ridge point 계산

### 문서
- `README.md` - 프로젝트 개요 및 사용 지침
- `desk/progress_summary.md` - 이 파일

## 📋 할 일 목록

### 높은 우선순위
1. **메모리 대역폭의 Write-to-Read Delay 반영**
   - 현재 대역폭 측정은 이상적인 순차 접근을 가정함
   - Write-to-Read 전환 페널티를 고려해야 함
   - 현실적인 읽기/쓰기 패턴 프로파일링 추가 (단순 복사가 아닌)
   - Write-to-Read 지연이 실제 대역폭에 미치는 영향 측정
   - 다양한 접근 패턴 고려: 순차, 스트라이드, 랜덤

2. **GPU 아키텍처를 반영한 현실적인 TFLOPS 측정 필요**
   - 현재 측정은 이상적인 대형 행렬 곱셈만 사용
   - 실제 워크로드에서 발생하는 다양한 행렬 크기 고려 필요
   - Tensor Core 활용률 측정 및 분석
   - 혼합 정밀도 연산의 실제 처리량 측정
   - 작은 배치 크기나 불규칙한 크기에서의 FLOPS 저하 파악

### 중간 우선순위
3. **Attention 메커니즘 프로파일링 추가**
4. **모든 커널에 대한 파라미터 스윕 구현**
5. **연산 강도 계산 및 Roofline 모델에 플롯**

### 낮은 우선순위
6. **결과 지속성 추가 (CSV/JSON 출력)**
7. **향상된 통계 (표준편차, 중앙값, 백분위수)**

## 마지막 업데이트
2025-11-20
