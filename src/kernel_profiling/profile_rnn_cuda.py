
import torch

def profile_recurrent_layer(
    batch_size: int = 16,
    sequence_length: int = 128,
    input_features: int = 512,
    hidden_features: int = 1024,
    num_layers: int = 2,
    num_warmup_runs: int = 10,
    num_profiling_runs: int = 100
):
    """
    Profiles the latency of a recurrent (LSTM) layer on the GPU using PyTorch.

    Args:
        batch_size (int): The batch size for the input tensor.
        sequence_length (int): The length of the input sequences.
        input_features (int): The number of input features.
        hidden_features (int): The number of hidden features in the LSTM.
        num_layers (int): The number of recurrent layers.
        num_warmup_runs (int): Number of runs to warm up the GPU before profiling.
        num_profiling_runs (int): Number of runs to average for profiling latency.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        return

    device = torch.device("cuda")

    # Create an LSTM layer and move it to the GPU
    rnn_layer = torch.nn.LSTM(
        input_size=input_features,
        hidden_size=hidden_features,
        num_layers=num_layers
    ).to(device)

    # Create a dummy input tensor and initial hidden/cell states
    input_tensor = torch.randn(sequence_length, batch_size, input_features).to(device)
    h0 = torch.randn(num_layers, batch_size, hidden_features).to(device)
    c0 = torch.randn(num_layers, batch_size, hidden_features).to(device)

    # Create CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    print(f"Profiling Recurrent (LSTM) Layer:")
    print(f"  Input Shape: ({sequence_length}, {batch_size}, {input_features})")
    print(f"  Hidden Features: {hidden_features}, Num Layers: {num_layers}")

    # --- Warm-up runs ---
    print(f"Performing {num_warmup_runs} warm-up runs...")
    for _ in range(num_warmup_runs):
        _, _ = rnn_layer(input_tensor, (h0, c0))
    torch.cuda.synchronize()

    # --- Profiling runs ---
    print(f"Performing {num_profiling_runs} profiling runs...")
    total_latency_ms = 0.0
    for _ in range(num_profiling_runs):
        start_event.record()
        _, _ = rnn_layer(input_tensor, (h0, c0))
        end_event.record()
        torch.cuda.synchronize()
        total_latency_ms += start_event.elapsed_time(end_event)

    average_latency_ms = total_latency_ms / num_profiling_runs
    print(f"Average Recurrent (LSTM) Layer Latency: {average_latency_ms:.3f} ms")

if __name__ == "__main__":
    profile_recurrent_layer()
