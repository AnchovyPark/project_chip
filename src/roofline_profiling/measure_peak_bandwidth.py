#!/usr/bin/env python
"""
Measures the achieved peak memory bandwidth of the GPU.

This script performs large memory copy operations to saturate the memory
subsystem and measure the maximum achievable memory bandwidth.
"""

import torch


def measure_peak_bandwidth(
    tensor_sizes_mb=None,
    num_warmup_runs: int = 5,
    num_profiling_runs: int = 20
):
    """
    Measures the achieved peak memory bandwidth using device-to-device copies.

    Args:
        tensor_sizes_mb (list): List of tensor sizes in MB to test. If None, uses defaults.
        num_warmup_runs (int): Number of warm-up iterations.
        num_profiling_runs (int): Number of profiling iterations per size.

    Returns:
        float: Peak achieved bandwidth in GB/s (GigaBytes per second).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return None

    device = torch.device("cuda")

    # Default tensor sizes to test (in MB)
    if tensor_sizes_mb is None:
        tensor_sizes_mb = [
            256,    # 256 MB
            512,    # 512 MB
            1024,   # 1 GB
            2048,   # 2 GB
        ]

    print("=" * 60)
    print("Measuring Peak Memory Bandwidth")
    print("=" * 60)

    max_bandwidth = 0.0
    best_size = None

    for size_mb in tensor_sizes_mb:
        # Calculate number of float16 elements
        # 1 MB = 1024 * 1024 bytes, float16 = 2 bytes
        num_elements = (size_mb * 1024 * 1024) // 2
        total_bytes = num_elements * 2  # FP16 = 2 bytes per element

        print(f"\nTesting tensor size: {size_mb} MB ({num_elements:,} FP16 elements)")

        # Create source and destination tensors
        src_tensor = torch.randn(num_elements, dtype=torch.float16, device=device)
        dst_tensor = torch.empty(num_elements, dtype=torch.float16, device=device)

        # CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up
        for _ in range(num_warmup_runs):
            dst_tensor.copy_(src_tensor)
        torch.cuda.synchronize()

        # Profiling
        total_time_ms = 0.0
        for _ in range(num_profiling_runs):
            start_event.record()
            dst_tensor.copy_(src_tensor)
            end_event.record()
            torch.cuda.synchronize()
            total_time_ms += start_event.elapsed_time(end_event)

        avg_time_ms = total_time_ms / num_profiling_runs
        avg_time_s = avg_time_ms / 1000.0

        # Calculate bandwidth (read + write, so total_bytes * 2)
        # Memory bandwidth accounts for both read from src and write to dst
        effective_bytes = total_bytes * 2
        achieved_bandwidth_bps = effective_bytes / avg_time_s  # Bytes per second
        achieved_bandwidth_gbps = achieved_bandwidth_bps / 1e9  # GB/s

        print(f"  Average time: {avg_time_ms:.3f} ms")
        print(f"  Achieved bandwidth: {achieved_bandwidth_gbps:.2f} GB/s")

        if achieved_bandwidth_gbps > max_bandwidth:
            max_bandwidth = achieved_bandwidth_gbps
            best_size = size_mb

    print("\n" + "=" * 60)
    print(f"Peak Achieved Bandwidth: {max_bandwidth:.2f} GB/s")
    print(f"Best configuration: {best_size} MB tensor")
    print("=" * 60)

    return max_bandwidth


if __name__ == "__main__":
    peak_bandwidth = measure_peak_bandwidth()

    if peak_bandwidth is not None:
        print(f"\n[Result] Peak Bandwidth: {peak_bandwidth:.2f} GB/s")
