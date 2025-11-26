# LLM Kernel Latency Profiling Project

This project contains a set of Python scripts designed to measure the execution latency of common deep learning operations (kernels) that are fundamental to Large Language Model (LLM) inference. The goal is to collect performance data on these individual operations to better understand and predict the overall inference time of complex models like those run with vLLM.

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

## Profiling Scripts

### Kernel Profiling
Each kernel profiling script measures the GPU latency of a specific operation using:
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

### Roofline Profiling
The Roofline model helps identify whether operations are compute-bound or memory-bound:

- **`measure_peak_flops.py`**: Measures the GPU's peak computational throughput (TFLOPS) using FP16 matrix multiplications
- **`measure_peak_bandwidth.py`**: Measures the GPU's peak memory bandwidth (GB/s) using large memory copy operations
- **`calculate_ridge_point.py`**: Calculates the ridge point (FLOP/Byte) that determines the boundary between compute-bound and memory-bound operations

The ridge point is calculated as: `Ridge Point = Peak FLOPS / Peak Bandwidth`

Kernels with operational intensity (FLOP/Byte) below the ridge point are memory-bound, while those above are compute-bound.

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

## Evaluation & Recommendations

### Strengths
- ✅ Correct GPU timing methodology using CUDA Events
- ✅ Proper warm-up and synchronization
- ✅ Consistent code structure across all scripts

### Areas for Improvement
1. **Statistical metrics**: Add standard deviation, median, min/max (not just average)
2. **Parameter sweeps**: Test multiple batch sizes, sequence lengths, hidden dimensions
3. **LLM-relevant operations**: Add Attention mechanisms (most important for LLM inference)
4. **Result persistence**: Save results to CSV/JSON for analysis
5. **Memory profiling**: Track GPU memory usage alongside latency
6. **Advanced profiling**: Consider using `torch.profiler` for kernel-level analysis

### Usage for vLLM Prediction
The current approach measures individual operations in isolation. For accurate vLLM inference time prediction:
- Consider operation fusion and overlap effects
- Account for vLLM-specific optimizations (PagedAttention, continuous batching)
- Profile with actual model architectures when possible
