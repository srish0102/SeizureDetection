import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# --- CONFIGURATION ---
# We use the file we created yesterday
DATA_PATH = "chb01_processed.npy" 

def get_device():
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("⚠️ GPU NOT found. Using CPU (Slow).")
        return torch.device("cpu")

# --- SIMPLE MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=23, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

# --- TRAINING LOOP ---
def train():
    device = get_device()
    
    print("📂 Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found. Did you run the download script?")
        # Create dummy data if file is missing, just to test GPU
        print("⚠️ Creating DUMMY data to test GPU anyway...")
        full_data = np.random.randn(23, 100000).astype(np.float32)
    else:
        full_data = np.load(DATA_PATH)

    # Prepare data slice
    n_samples = 100
    n_channels = 23
    time_points = 1000
    
    # Reshape logic
    if full_data.shape[1] >= n_samples * time_points:
        real_data_slice = full_data[:, :n_samples * time_points]
        real_data_slice = real_data_slice.reshape(n_samples, n_channels, time_points)
    else:
        print("⚠️ Data too small, using random noise for test.")
        real_data_slice = np.random.randn(n_samples, n_channels, time_points).astype(np.float32)
    
    tensor_x = torch.Tensor(real_data_slice).to(device)
    tensor_y = torch.randint(0, 2, (n_samples,)).to(device)

    # Model & Training
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("🚀 Starting Training Run...")
    model.train()
    for epoch in range(1, 11):
        optimizer.zero_grad()
        output = model(tensor_x)
        loss = criterion(output, tensor_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}/10 | Loss: {loss.item():.4f}")

    print("\n🎉 TEST COMPLETE: Your GPU is working!")

if __name__ == "__main__":
    train()
