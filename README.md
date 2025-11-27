# CHIP: Calibrated Hardware Performance in Inference Prediction

**CHIP** is a framework for predicting Large Language Model (LLM) inference latency by calibrating and characterizing GPU hardware performance. Rather than simply profiling kernels in isolation, CHIP aims to build an accurate latency prediction model by understanding hardware limits and bottlenecks through systematic benchmarking and Roofline analysis.

The project measures GPU performance characteristics and operation latencies to predict the inference time of complex models like those run with vLLM.

## Project Goals

CHIP addresses the challenge of accurately predicting LLM inference latency by:

1. **Hardware Calibration**: Measuring actual GPU performance limits (peak FLOPS, memory bandwidth, ridge point) rather than relying on theoretical specifications
2. **Calibration Factor Development**: Creating correction factors that account for real-world performance degradation (write-to-read delays, SM utilization, workload diversity)
3. **Bottleneck Identification**: Using Roofline model analysis to determine whether operations are compute-bound or memory-bound
4. **Realistic Profiling**: Applying calibration factors to hardware performance metrics for accurate modeling
5. **Latency Prediction**: Building a prediction model based on calibrated hardware characteristics and operation profiles

## Target Environment

These scripts are intended to be run in an environment with an **NVIDIA GPU** and the **PyTorch** library installed. All profiling is performed on the CUDA device. The primary development and execution environment is assumed to be an AWS EC2 instance with GPU capabilities.

## Directory Structure

The project is organized as follows:

```
.
├── desk/
│   └── progress_summary.md
├── README.md
└── src/
    ├── kernel_profiling/
    │   ├── profile_fc_cuda.py      # Fully connected (Linear) layer
    │   ├── profile_conv_cuda.py    # 2D Convolutional layer
    │   ├── profile_rnn_cuda.py     # LSTM layer
    │   └── profile_mem_cuda.py     # Memory-bound ops (LayerNorm + GELU)
    └── roofline_profiling/
        ├── measure_peak_flops.py   # GPU peak FLOPS measurement
        ├── measure_peak_bandwidth.py  # GPU peak bandwidth measurement
        └── calculate_ridge_point.py   # Roofline model ridge point
```

-   **`src/kernel_profiling/`**: Contains Python scripts for profiling specific operations.
    -   **`profile_fc_cuda.py`**: Profiles `torch.nn.Linear` layers (fully connected).
    -   **`profile_conv_cuda.py`**: Profiles `torch.nn.Conv2d` layers (2D convolution).
    -   **`profile_rnn_cuda.py`**: Profiles `torch.nn.LSTM` layers (recurrent).
    -   **`profile_mem_cuda.py`**: Profiles memory-bound operations (`LayerNorm` + `GELU`).
-   **`src/roofline_profiling/`**: Contains scripts for GPU Roofline model analysis.
    -   **`measure_peak_flops.py`**: Measures GPU peak FLOPS using FP16 matrix multiplication.
    -   **`measure_peak_bandwidth.py`**: Measures GPU peak memory bandwidth.
    -   **`calculate_ridge_point.py`**: Calculates the ridge point for the Roofline model.
-   **`desk/`**: Contains markdown files for project documentation and progress tracking.

## Components

CHIP consists of two main components that work together to enable accurate latency prediction:

### 1. Kernel Profiling
Measures the actual latency of individual deep learning operations on the target GPU. Each kernel profiling script uses:
- **CUDA Events** (`torch.cuda.Event`) for precise GPU-side timing
- **Warm-up runs** to stabilize GPU state (frequency, cache)
- **Multiple profiling runs** (default: 100) for statistical averaging
- **Synchronization** (`torch.cuda.synchronize()`) to ensure accurate measurements

#### Current Operation Coverage

| Script | Operation | Relevance to LLM |
|--------|-----------|------------------|
| `profile_fc_cuda.py` | Linear (Fully Connected) | ✅ High - Core LLM operation |
| `profile_mem_cuda.py` | LayerNorm + GELU | ✅ High - Common in Transformers |
| `profile_conv_cuda.py` | Conv2d | ⚠️ Low - Mainly for Vision models |
| `profile_rnn_cuda.py` | LSTM | ⚠️ Low - Modern LLMs use Transformers |

**Note:** For accurate vLLM inference prediction, consider adding:
- Self-Attention / Multi-Head Attention (most critical)
- Softmax operations
- Various GEMM sizes
- Embedding operations

### 2. Hardware Calibration (Roofline Profiling)
Characterizes the GPU's actual performance limits to calibrate the prediction model. The Roofline model identifies whether operations are compute-bound or memory-bound:

- **`measure_peak_flops.py`**: Measures the GPU's achieved peak computational throughput (TFLOPS) using FP16 matrix multiplications
- **`measure_peak_bandwidth.py`**: Measures the GPU's achieved peak memory bandwidth (GB/s) using large memory copy operations
- **`calculate_ridge_point.py`**: Calculates the ridge point (FLOP/Byte) that determines the boundary between compute-bound and memory-bound operations

**Ridge Point Formula**: `Ridge Point = Peak FLOPS / Peak Bandwidth`

### Calibration Factor Methodology

CHIP introduces **Calibration Factors** to bridge the gap between ideal measurements and real-world performance:

**Why Calibration Factors?**
- Theoretical GPU specs (e.g., 312 TFLOPS for A100) rarely reflect actual achievable performance
- Ideal benchmarks (large matrix multiplications, sequential memory access) don't represent diverse workloads
- Real workloads suffer from: write-to-read delays, suboptimal SM utilization, irregular matrix sizes, memory hierarchy effects

**How Calibration Works:**
1. **Measure Ideal Performance**: Peak FLOPS and bandwidth under optimal conditions
2. **Measure Realistic Performance**: Performance under diverse workload conditions (various sizes, access patterns)
3. **Calculate Calibration Factor**: `CF = Realistic Performance / Ideal Performance`
4. **Apply to Predictions**: `Predicted Latency = f(Calibrated_FLOPS, Calibrated_BW, Operation_Profile)`

**Example Calibration Factors:**
- **Memory Bandwidth CF**: Accounts for write-to-read turnaround penalties (~0.7-0.9x ideal)
- **FLOPS CF per SM**: Reflects staircase latency behavior for varying matrix sizes (~0.6-0.95x ideal)
- **Tensor Core Utilization CF**: Actual utilization vs theoretical peak (~0.5-0.85x depending on workload)

This approach ensures predictions are grounded in **achievable** hardware performance, not theoretical maximums.

## How to Run

1.  **Prerequisites**: Ensure you have Python 3 and PyTorch with CUDA support installed in your environment.
    ```bash
    # Example installation for PyTorch with CUDA 12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Execute a Profiling Script**: Navigate to the project root directory and run the desired script using `python3`.

    **Kernel Profiling:**
    -   **Fully Connected:**
        ```bash
        python3 src/kernel_profiling/profile_fc_cuda.py
        ```
    -   **Convolutional:**
        ```bash
        python3 src/kernel_profiling/profile_conv_cuda.py
        ```
    -   **Recurrent:**
        ```bash
        python3 src/kernel_profiling/profile_rnn_cuda.py
        ```
    -   **Memory-Limited:**
        ```bash
        python3 src/kernel_profiling/profile_mem_cuda.py
        ```

    Each script will output the average latency in milliseconds (ms) after performing a number of warm-up and profiling runs.

    **Roofline Profiling:**
    -   **Peak FLOPS:**
        ```bash
        python3 src/roofline_profiling/measure_peak_flops.py
        ```
    -   **Peak Bandwidth:**
        ```bash
        python3 src/roofline_profiling/measure_peak_bandwidth.py
        ```
    -   **Ridge Point (runs both above tests):**
        ```bash
        python3 src/roofline_profiling/calculate_ridge_point.py
        ```

    The Roofline scripts will output peak performance metrics and the ridge point for classifying operations as compute-bound or memory-bound.

## Current Status & Roadmap

### ✅ Completed
- Hardware calibration framework (Roofline model)
- Basic kernel profiling for common operations
- Ridge point calculation for bottleneck identification

### 🚧 In Progress
See [progress_summary.md](desk/progress_summary.md) for detailed status and evaluation.

### 🎯 Key Next Steps for Accurate Latency Prediction
1. **Calibration Factor Measurement**:
   - Measure write-to-read delay effects on memory bandwidth → derive Memory BW Calibration Factor
   - Profile FLOPS across diverse matrix sizes to capture SM utilization patterns → derive FLOPS Calibration Factor
   - Measure Tensor Core utilization under various workloads → derive Tensor Core CF
   - Create hierarchical Roofline model accounting for cache/memory hierarchy

2. **Critical Missing Operations**:
   - Self-Attention / Multi-Head Attention profiling (highest priority for LLM)
   - Softmax, Embedding lookups

3. **Prediction Model Development**:
   - Calculate operational intensity for each kernel
   - Apply calibration factors to hardware performance metrics
   - Build regression model mapping calibrated profiles to actual inference time
   - Validate against real vLLM workloads

### Design Philosophy
CHIP's calibration-based approach differs from simple kernel profiling by:
- **Measuring actual hardware limits**, not theoretical specs
- **Deriving calibration factors** that quantify the gap between ideal and realistic performance
- **Identifying performance bottlenecks** (compute vs memory bound) through Roofline analysis
- **Accounting for real-world factors** (write-to-read delays, SM utilization, diverse workloads, memory hierarchy)
- **Building predictive models** grounded in calibrated (not theoretical) hardware characteristics

**Key Insight**: By measuring how much real workloads deviate from ideal conditions (via calibration factors), CHIP can accurately predict latency for unseen workloads without exhaustive profiling of every possible configuration.
