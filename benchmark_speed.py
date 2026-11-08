import time
import torch
from train_cnn_mamba import HybridCNNMamba
from train_cnn_lstm import HybridCNNLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
mamba_model = HybridCNNMamba(in_channels=18, d_model=128).to(DEVICE).eval()
lstm_model = HybridCNNLSTM(in_channels=18, d_model=128).to(DEVICE).eval()

# Create a mock single 4-second EEG window (Batch Size = 1)
dummy_window = torch.randn(1, 18, 1024, device=DEVICE)

# Warm-up GPU
for _ in range(50):
    _ = mamba_model(dummy_window)
    _ = lstm_model(dummy_window)

# Benchmark Mamba Latency
torch.cuda.synchronize()
start_time = time.perf_counter()
for _ in range(1000):
    _ = mamba_model(dummy_window)
torch.cuda.synchronize()
mamba_latency = (time.perf_counter() - start_time) / 1000 * 1000  # in ms

# Benchmark LSTM Latency
torch.cuda.synchronize()
start_time = time.perf_counter()
for _ in range(1000):
    _ = lstm_model(dummy_window)
torch.cuda.synchronize()
lstm_latency = (time.perf_counter() - start_time) / 1000 * 1000  # in ms

# Calculate Parameter Count
mamba_params = sum(p.numel() for p in mamba_model.parameters() if p.requires_grad)
lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)

print("=" * 55)
print(f"⚡ HARDWARE: {torch.cuda.get_device_name(0)}")
print("=" * 55)
print(f"🧠 CNN-Mamba Parameters: {mamba_params:,}")
print(f"⏱️ CNN-Mamba Latency per Window: {mamba_latency:.3f} ms")
print("-" * 55)
print(f"🔄 CNN-LSTM Parameters: {lstm_params:,}")
print(f"⏱️ CNN-LSTM Latency per Window: {lstm_latency:.3f} ms")
print("=" * 55)
