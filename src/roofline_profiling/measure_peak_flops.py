#!/usr/bin/env python
"""
Measures the achieved peak FLOPS of the GPU using FP16 matrix multiplication.

This script performs large matrix multiplications to saturate the GPU and
measure the maximum achievable FLOPS (floating point operations per second).
"""

import torch


def measure_peak_flops_fp16(
    matrix_sizes=None,
    num_warmup_runs: int = 5,
    num_profiling_runs: int = 20
):
    """
    Measures the achieved peak FLOPS using FP16 GEMM operations.

    Args:
        matrix_sizes (list): List of (M, N, K) tuples to test. If None, uses default sizes.
        num_warmup_runs (int): Number of warm-up iterations.
        num_profiling_runs (int): Number of profiling iterations per size.

    Returns:
        float: Peak achieved FLOPS in TFLOPS (TeraFLOPS).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return None

    device = torch.device("cuda")

    # Default matrix sizes to test (optimized for GPU saturation)
    if matrix_sizes is None:
        matrix_sizes = [
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 16384, 16384),
        ]

    print("=" * 60)
    print("Measuring Peak FLOPS (FP16)")
    print("=" * 60)

    max_flops = 0.0
    best_config = None

    for M, N, K in matrix_sizes:
        print(f"\nTesting matrix size: A({M}x{K}) @ B({K}x{N})")

        # Create FP16 matrices
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)

        # CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up
        for _ in range(num_warmup_runs):
            _ = torch.matmul(A, B)
        torch.cuda.synchronize()

        # Profiling
        total_time_ms = 0.0
        for _ in range(num_profiling_runs):
            start_event.record()
            C = torch.matmul(A, B)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)

        avg_time_ms = total_time_ms / num_profiling_runs
        avg_time_s = avg_time_ms / 1000.0

        # Calculate FLOPS
        # For C = A @ B, FLOPS = 2 * M * N * K (multiply-add operations)
        flops_per_matmul = 2 * M * N * K
        achieved_flops = flops_per_matmul / avg_time_s  # FLOPS
        achieved_tflops = achieved_flops / 1e12  # TeraFLOPS

        print(f"  Average time: {avg_time_ms:.3f} ms")
        print(f"  Achieved FLOPS: {achieved_tflops:.3f} TFLOPS")

        if achieved_tflops > max_flops:
            max_flops = achieved_tflops
            best_config = (M, N, K)

    print("\n" + "=" * 60)
    print(f"Peak Achieved FLOPS (FP16): {max_flops:.3f} TFLOPS")
    print(f"Best configuration: {best_config}")
    print("=" * 60)

    return max_flops


if __name__ == "__main__":
    peak_flops = measure_peak_flops_fp16()

    if peak_flops is not None:
        print(f"\n[Result] Peak FLOPS: {peak_flops:.3f} TFLOPS")
