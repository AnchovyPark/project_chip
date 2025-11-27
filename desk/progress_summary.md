
# LLM 커널 지연시간 프로파일링 프로젝트 - 진행 상황 요약

이 문서는 LLM 추론 시간 예측을 위한 다양한 커널 프로파일링 작업의 진행 상황을 요약합니다.

## 목표
GPU 연산의 성능 특성을 측정하고 분석하여 대규모 언어 모델(LLM) 추론을 예측하는 것이 주요 목표입니다. 프로젝트는 두 가지 주요 구성 요소로 이루어져 있습니다:

### 1. 커널 프로파일링(GPU에 따라 달라지는 커널)
일반적인 딥러닝 연산의 실행 시간(지연시간) 측정:
1.  완전 연결 계층 (Fully Connected Layers)
2.  합성곱 계층 (Convolutional Layers)
3.  순환 계층 (Recurrent Layers)
4.  메모리 제한 연산 (Memory-Limited Operations)

### 2. Roofline 모델 분석(계층적 Roofline을 반영해야 하는데...)
GPU 하드웨어 한계를 특성화하여 연산 성능 병목 현상 파악:
1.  Peak FLOPS (연산 처리량)
2.  Peak Memory Bandwidth (메모리 대역폭)
3.  Ridge Point (연산 제한 vs 메모리 제한 경계)

이 측정값들을 사용하여 PyTorch를 활용하는 vLLM의 추론 시간을 예측하는 것이 목표입니다.

## 환경 참고사항
- **개발 환경**: MacBook (CUDA 지원 없음)
- **대상 실행 환경**: NVIDIA GPU가 있는 AWS EC2
- **프레임워크**: PyTorch with CUDA 12.1+
- **Python 버전**: Python 3.x

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

1. **메모리 대역폭의 Write-to-Read Delay 반영**
   - 현재 대역폭 측정은 이상적인 순차 접근을 가정함
   - Write-to-Read 전환 페널티를 고려해야 함
   - 현실적인 읽기/쓰기 패턴 프로파일링 추가 (단순 복사가 아닌)
   - Write-to-Read 지연이 실제 대역폭에 미치는 영향 측정
   - 다양한 접근 패턴 고려: 순차, 스트라이드, 랜덤

2. **SM당 처리량을 반영하도록 바꿔야 됨**
   - 현재 측정은 이상적인 대형 행렬 곱셈만 사용
   - 실제 워크로드에서 발생하는 다양한 행렬 크기 고려 필요
      - NeuSight 처럼 계단식으로 Latency가 증가함을 반영할 수 있어야 함
      - Delay Factor 측정했던 모양이, 살짝의 진동이 있었는데 그걸 반영해서 만들어 볼까?
   - Tensor Core 활용률 측정 및 분석
   - 혼합 정밀도 연산의 실제 처리량 측정
   - 작은 배치 크기나 불규칙한 크기에서의 FLOPS 저하 파악(위의 내용과 연게됨)
   - [실험환경 구성 필수] cuDNN라이브러리를 사용하는 환경 (버전도 중요함)
   - [실험환경 구성 필수] cudaToolkit 버전
   - [실험환경 구성 필수] SM 의 capacity에 맞게 여러 shape가 올라가서 실행되는 실험이 진행되어야 함

3. **메모리 계층적인 Roofline 반영**
   - 이건 아직 감이 안잡힘.
   - 구체화 되어야 하는데 우째야 하노

4. **vLLM 사용하는 예측 환경을 써야 됨**
   - 암튼 그럼 그러함.


## 마지막 업데이트
2025-11-26

-----

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