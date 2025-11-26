
#!/usr/bin/env python
import torch
import time

def profile_fully_connected_layer(
    batch_size: int = 16,
    input_features: int = 1024,
    output_features: int = 512,
    num_warmup_runs: int = 10,
    num_profiling_runs: int = 100
):
    """
    Profiles the latency of a fully connected (Linear) layer on the GPU using PyTorch.

    Args:
        batch_size (int): The batch size for the input tensor.
        input_features (int): The number of input features for the linear layer.
        output_features (int): The number of output features for the linear layer.
        num_warmup_runs (int): Number of runs to warm up the GPU before profiling.
        num_profiling_runs (int): Number of runs to average for profiling latency.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")

    # Create a linear layer and move it to the GPU
    linear_layer = torch.nn.Linear(input_features, output_features).to(device)

    # Create a dummy input tensor and move it to the GPU
    input_tensor = torch.randn(batch_size, input_features).to(device)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Profiling Fully Connected Layer:")
    print(f"  Input Shape: ({batch_size}, {input_features})")
    print(f"  Output Shape: ({batch_size}, {output_features})")
    print(f"  Weights Shape: ({output_features}, {input_features})")

    # --- Warm-up runs ---
    print(f"Performing {num_warmup_runs} warm-up runs...")
    for _ in range(num_warmup_runs):
        _ = linear_layer(input_tensor)
    torch.cuda.synchronize() # Ensure all warm-up operations are complete

    # --- Profiling runs ---
    print(f"Performing {num_profiling_runs} profiling runs...")
    total_latency_ms = 0.0
    for _ in range(num_profiling_runs):
        start_event.record()
        _ = linear_layer(input_tensor)
        end_event.record()
        torch.cuda.synchronize() # Wait for the GPU to finish the operation
        total_latency_ms += start_event.elapsed_time(end_event) # milliseconds

    average_latency_ms = total_latency_ms / num_profiling_runs
    print(f"Average Fully Connected Layer Latency: {average_latency_ms:.3f} ms")

if __name__ == "__main__":
    profile_fully_connected_layer()

