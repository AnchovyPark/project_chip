
import torch

def profile_convolutional_layer(
    batch_size: int = 16,
    in_channels: int = 3,
    out_channels: int = 64,
    height: int = 224,
    width: int = 224,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    num_warmup_runs: int = 10,
    num_profiling_runs: int = 100
):
    """
    Profiles the latency of a 2D convolutional layer on the GPU using PyTorch.

    Args:
        batch_size (int): The batch size for the input tensor.
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        height (int): The height of the input image.
        width (int): The width of the input image.
        kernel_size (int): The size of the convolutional kernel.
        stride (int): The stride of the convolution.
        padding (int): The padding for the input.
        num_warmup_runs (int): Number of runs to warm up the GPU before profiling.
        num_profiling_runs (int): Number of runs to average for profiling latency.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")

    # Create a 2D convolutional layer and move it to the GPU
    conv_layer = torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    ).to(device)

    # Create a dummy input tensor and move it to the GPU
    input_tensor = torch.randn(batch_size, in_channels, height, width).to(device)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Profiling Convolutional Layer:")
    print(f"  Input Shape: ({batch_size}, {in_channels}, {height}, {width})")
    print(f"  Out Channels: {out_channels}, Kernel: {kernel_size}, Stride: {stride}, Padding: {padding}")

    # --- Warm-up runs ---
    print(f"Performing {num_warmup_runs} warm-up runs...")
    for _ in range(num_warmup_runs):
        _ = conv_layer(input_tensor)
    torch.cuda.synchronize()

    # --- Profiling runs ---
    print(f"Performing {num_profiling_runs} profiling runs...")
    total_latency_ms = 0.0
    for _ in range(num_profiling_runs):
        start_event.record()
        _ = conv_layer(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        total_latency_ms += start_event.elapsed_time(end_event)

    average_latency_ms = total_latency_ms / num_profiling_runs
    print(f"Average Convolutional Layer Latency: {average_latency_ms:.3f} ms")

if __name__ == "__main__":
    profile_convolutional_layer()
