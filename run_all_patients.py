import os
import re
import mne
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
from mamba import MambaBlock  # Uses your custom Mamba block

# --- 1. CONFIGURATION ---
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"
SAVE_DIR = "/mnt/d/Seizure_Project_Fast/full_study_data"
ALL_PATIENTS = [f"chb{i:02d}" for i in range(1, 24)] # chb01 to chb23
BATCH_SIZE = 64
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 2. DATA PIPELINE ---
def download_file(url, local_path):
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return 
    print(f"   ⬇️ Downloading {url.split('/')[-1]}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"   ⚠️ Failed to download: {e}")

def get_patient_data(patient_id):
    x_path = os.path.join(SAVE_DIR, f"X_{patient_id}.npy")
    y_path = os.path.join(SAVE_DIR, f"y_{patient_id}.npy")
    
    if os.path.exists(x_path):
        return np.load(x_path), np.load(y_path)

    print(f"\n⚙️ Processing {patient_id}...")
    summary_url = f"{BASE_URL}{patient_id}/{patient_id}-summary.txt"
    summary_path = os.path.join(SAVE_DIR, f"{patient_id}-summary.txt")
    download_file(summary_url, summary_path)
    
    seizures = []
    if not os.path.exists(summary_path): return None, None
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        current_file = ""
        for line in lines:
            if "File Name" in line:
                current_file = line.split(": ")[1].strip()
            if "Seizure" in line and "Start Time" in line:
                start = int(re.search(r'\d+', line).group())
            if "Seizure" in line and "End Time" in line:
                end = int(re.search(r'\d+', line).group())
                seizures.append((current_file, start, end))

    if not seizures:
        print(f"   ❌ No seizures listed for {patient_id}.")
        return None, None

    all_X, all_y = [], []
    processed_files = set()
    
    for filename, start, end in seizures:
        if filename in processed_files: continue
        
        file_url = f"{BASE_URL}{patient_id}/{filename}"
        local_edf = os.path.join(SAVE_DIR, filename)
        download_file(file_url, local_edf)
        
        try:
            raw = mne.io.read_raw_edf(local_edf, preload=True, verbose=False)
            raw.filter(0.5, 40.0, verbose=False)
            data = raw.get_data() * 1e6
            fs = int(raw.info['sfreq'])
            
            s_start, s_end = start * fs, end * fs
            n_start = max(0, s_start - (s_end - s_start)) 
            win = 4 * fs
            
            # --- QUALITY CONTROL UPDATE HERE ---
            for i in range(s_start, s_end - win, win):
                slice_data = data[:, i:i+win]
                if slice_data.shape[1] == win:  # Only accept perfect 4-second blocks
                    all_X.append(slice_data)
                    all_y.append(1)
                    
            for i in range(n_start, s_start - win, win):
                slice_data = data[:, i:i+win]
                if slice_data.shape[1] == win:  # Only accept perfect 4-second blocks
                    all_X.append(slice_data)
                    all_y.append(0)
            # -----------------------------------
                    
            processed_files.add(filename)
        except Exception as e:
            print(f"   ⚠️ Error processing {filename}: {e}")
        finally:
            # Delete the heavy .edf file immediately to save your disk space
            if os.path.exists(local_edf):
                os.remove(local_edf)

    if len(all_X) == 0: return None, None
    
    # This matrix build will now succeed perfectly
    X_final, y_final = np.array(all_X), np.array(all_y)
    np.save(x_path, X_final)
    np.save(y_path, y_final)
    return X_final, y_final

# --- 3. MODEL ARCHITECTURE ---
class HybridCNNMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(23, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4)
        )
        self.mamba = MambaBlock(d_model=128, d_state=16, d_conv=4, expand=2)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.mean(dim=1)
        return self.sigmoid(self.fc(x))

# --- 4. EXECUTION LOOP ---
results = []
print(f"🚀 STARTING FULL SCALE RESEARCH ON {len(ALL_PATIENTS)} PATIENTS")
print(f"💻 Using Device: {DEVICE}")

for patient in ALL_PATIENTS:
    X, y = get_patient_data(patient)
    if X is None: continue
    
    print(f"   🥊 Training Mamba on {len(X)} samples...", end="", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).unsqueeze(1)), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test).unsqueeze(1)), batch_size=BATCH_SIZE)
    
    model = HybridCNNMamba().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Train
    model.train()
    for epoch in range(EPOCHS):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            
    # Test
    model.eval()
    y_true, y_pred = [] , []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            preds = (model(inputs) > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    print(f" Done! [Acc: {acc*100:.1f}% | Prec: {prec:.2f}]")
    
    results.append([patient, acc, prec])

# --- 5. FINAL REPORT ---
df = pd.DataFrame(results, columns=["Patient", "Accuracy", "Precision"])
print("\n🏆 FINAL RESEARCH RESULTS")
print(df)
df.to_csv("all_patients_results.csv", index=False)
print("✅ Saved to 'all_patients_results.csv'")