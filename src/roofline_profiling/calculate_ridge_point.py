#!/usr/bin/env python
"""
Calculates the Ridge Point for the GPU Roofline Model.

The Ridge Point represents the operational intensity (FLOP/Byte) at which
a kernel transitions from being memory-bound to compute-bound.

Ridge Point = Peak FLOPS / Peak Bandwidth

This script imports and runs the peak FLOPS and bandwidth measurement scripts,
then calculates and reports the ridge point.
"""

import sys
from pathlib import Path

# Add the roofline_profiling directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent))

from measure_peak_flops import measure_peak_flops_fp16
from measure_peak_bandwidth import measure_peak_bandwidth


def calculate_ridge_point():
    """
    Measures peak FLOPS and bandwidth, then calculates the ridge point.

    Returns:
        dict: Dictionary containing peak_flops, peak_bandwidth, and ridge_point.
    """
    print("\n" + "=" * 70)
    print(" GPU ROOFLINE MODEL - RIDGE POINT CALCULATION")
    print("=" * 70)

    # Step 1: Measure Peak FLOPS (FP16)
    print("\n[Step 1/3] Measuring Peak FLOPS...")
    peak_flops = measure_peak_flops_fp16()

    if peak_flops is None:
        print("Error: Could not measure peak FLOPS.")
        return None

    # Step 2: Measure Peak Bandwidth
    print("\n[Step 2/3] Measuring Peak Memory Bandwidth...")
    peak_bandwidth = measure_peak_bandwidth()

    if peak_bandwidth is None:
        print("Error: Could not measure peak bandwidth.")
        return None

    # Step 3: Calculate Ridge Point
    print("\n[Step 3/3] Calculating Ridge Point...")

    # Ridge Point = Peak FLOPS / Peak Bandwidth
    # Units: (TFLOPS) / (GB/s) = (10^12 FLOP/s) / (10^9 Byte/s) = 10^3 FLOP/Byte
    # So we multiply by 1000 to get FLOP/Byte
    ridge_point_flop_per_byte = (peak_flops / peak_bandwidth) * 1000

    # Display results
    print("\n" + "=" * 70)
    print(" ROOFLINE MODEL RESULTS")
    print("=" * 70)
    print(f"Peak FLOPS (FP16):       {peak_flops:.3f} TFLOPS")
    print(f"Peak Bandwidth:          {peak_bandwidth:.2f} GB/s")
    print(f"Ridge Point:             {ridge_point_flop_per_byte:.2f} FLOP/Byte")
    print("=" * 70)

    print("\nInterpretation:")
    print(f"  - Kernels with operational intensity < {ridge_point_flop_per_byte:.2f} FLOP/Byte are MEMORY-BOUND")
    print(f"  - Kernels with operational intensity > {ridge_point_flop_per_byte:.2f} FLOP/Byte are COMPUTE-BOUND")
    print(f"  - At the ridge point, performance is limited equally by compute and memory")

    return {
        'peak_flops_tflops': peak_flops,
        'peak_bandwidth_gbps': peak_bandwidth,
        'ridge_point_flop_per_byte': ridge_point_flop_per_byte
    }


if __name__ == "__main__":
    results = calculate_ridge_point()

    if results:
        print("\n" + "=" * 70)
        print("Benchmark completed successfully!")
        print("=" * 70)
