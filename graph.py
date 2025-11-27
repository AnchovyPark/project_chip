import matplotlib.pyplot as plt
import numpy as np

# 데이터 추정 (이미지로부터 근사값 추출)
# X축이 2^5 부터 2^13 까지의 데이터 포인트

# 1. 상단 그래프 (Fixed K=4096, Varying M=N)
# TFLOPS (좌측 그래프 Y값)
top_tflops = [1, 3, 8, 30, 85, 160, 230, 260, 262]
# Duration (우측 그래프 Y값, log2 scale을 ms로 변환)
# 2^-6 ~ 0.015, 2^-1 ~ 0.5, 2^1 ~ 2.0
top_duration_log2 = [-5.8, -5.8, -5.85, -5.8, -5.3, -4.2, -2.8, -0.9, 1.1]
top_duration_ms = [2**x for x in top_duration_log2]

# 2. 하단 그래프 (Fixed M=N=4096, Varying K)
# TFLOPS (좌측 그래프 Y값)
bottom_tflops = [30, 52, 100, 155, 205, 235, 250, 260, 255]
# Duration (우측 그래프 Y값)
bottom_duration_log2 = [-4.9, -4.7, -4.6, -4.2, -3.6, -2.8, -1.9, -0.9, 0.1]
bottom_duration_ms = [2**x for x in bottom_duration_log2]

# 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 첫 번째 그래프 (상단 데이터)
ax1.plot(top_tflops, top_duration_ms, 'o-', color='green', label='Fixed K=4096 (Var M=N)')
ax1.set_title('Re-plotted Figure (Top Row)\nTFLOPS vs Duration')
ax1.set_xlabel('TFLOPS (Performance)')
ax1.set_ylabel('Duration (ms)')
ax1.grid(True, which="both", ls="--")
ax1.set_yscale('log') # 원본의 특성을 살리기 위해 Y축 로그 스케일 적용

# 두 번째 그래프 (하단 데이터)
ax2.plot(bottom_tflops, bottom_duration_ms, 'o-', color='darkgreen', label='Fixed M=N=4096 (Var K)')
ax2.set_title('Re-plotted Figure (Bottom Row)\nTFLOPS vs Duration')
ax2.set_xlabel('TFLOPS (Performance)')
ax2.set_ylabel('Duration (ms)')
ax2.grid(True, which="both", ls="--")
ax2.set_yscale('log') # 원본의 특성을 살리기 위해 Y축 로그 스케일 적용

plt.tight_layout()
plt.show()