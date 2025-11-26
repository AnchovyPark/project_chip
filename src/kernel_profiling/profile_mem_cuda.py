
import torch

def profile_memory_limited_ops(
    batch_size: int = 16,
    sequence_length: int = 128,
    features: int = 512,
    num_warmup_runs: int = 10,
    num_profiling_runs: int = 100
):
    """
    Profiles the latency of memory-bandwidth-bound operations (LayerNorm + GELU)
    on the GPU using PyTorch.

    Args:
        batch_size (int): The batch size for the input tensor.
        sequence_length (int): The length of the input sequences.
        features (int): The number of features in the input tensor.
        num_warmup_runs (int): Number of runs to warm up the GPU before profiling.
        num_profiling_runs (int): Number of runs to average for profiling latency.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")

    # Define a sequence of memory-bound operations
    memory_limited_ops = torch.nn.Sequential(
        torch.nn.LayerNorm(features),
        torch.nn.GELU()
    ).to(device)

    # Create a dummy input tensor and move it to the GPU
    input_tensor = torch.randn(batch_size, sequence_length, features).to(device)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Profiling Memory-Limited Operations (LayerNorm + GELU):")
    print(f"  Input Shape: ({batch_size}, {sequence_length}, {features})")

    # --- Warm-up runs ---
    print(f"Performing {num_warmup_runs} warm-up runs...")
    for _ in range(num_warmup_runs):
        _ = memory_limited_ops(input_tensor)
    torch.cuda.synchronize()

    # --- Profiling runs ---
    print(f"Performing {num_profiling_runs} profiling runs...")
    total_latency_ms = 0.0
    for _ in range(num_profiling_runs):
        start_event.record()
        _ = memory_limited_ops(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        total_latency_ms += start_event.elapsed_time(end_event)

    average_latency_ms = total_latency_ms / num_profiling_runs
    print(f"Average Memory-Limited Ops Latency: {average_latency_ms:.3f} ms")

if __name__ == "__main__":
    profile_memory_limited_ops()
