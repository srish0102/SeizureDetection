import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
MODEL_SAVE_PATH = "cnn_lstm_seizure_model.pth"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. HARDWARE
print("=" * 60)
print(f"🚀 Benchmarking CNN-LSTM on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

# 2. LOAD DATA
X = np.load(os.path.join(DATA_DIR, "X_train.npy"))
y = np.load(os.path.join(DATA_DIR, "y_train.npy"))

# Identical random_state for exact same split as Mamba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Identical Recall-Weighted Loss
num_pos = np.sum(y_train == 1)
num_neg = np.sum(y_train == 0)
pos_weight_val = num_neg / max(num_pos, 1)
pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=False
)

# 3. HYBRID CNN-LSTM ARCHITECTURE
class HybridCNNLSTM(nn.Module):
    def __init__(self, in_channels=18, d_model=128, lstm_hidden=64):
        super(HybridCNNLSTM, self).__init__()
        
        # Identical CNN feature extractor to ensure fairness
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),

            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2)
        )
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Classifier (matching linear head)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: [Batch, 18, 1024]
        x = self.cnn(x)              # [Batch, 128, 64]
        x = x.permute(0, 2, 1)       # [Batch, 64, 128] (Batch, Time, Features)
        
        lstm_out, _ = self.lstm(x)   # [Batch, 64, 128]
        x = lstm_out.mean(dim=1)     # Global average pooling across time
        logits = self.classifier(x)  # [Batch, 1]
        return logits

# Initialize Model, Loss, Optimizer
model = HybridCNNLSTM(in_channels=18, d_model=128, lstm_hidden=64).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 4. TRAINING & EVALUATION
best_f1 = 0.0

print("\n🔥 Starting Hybrid CNN-LSTM Training...")
print("-" * 85)
print(f"{'Epoch':<8} | {'Loss':<10} | {'Sensitivity (Recall)':<22} | {'Precision':<12} | {'F1-Score':<10} | {'ROC-AUC':<10}")
print("-" * 85)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        
    avg_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            y_true.extend(labels.cpu().numpy().flatten())
            y_pred.extend(preds.cpu().numpy().flatten())
            y_prob.extend(probs.cpu().numpy().flatten())
            
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    
    print(f"{epoch:02d}/{EPOCHS:02d}   | {avg_loss:<10.4f} | {sensitivity*100:<21.2f}% | {precision*100:<11.2f}% | {f1*100:<9.2f}% | {auc:<10.4f}")
    
    if f1 >= best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("-" * 85)
print(f"\n✅ Training Complete! Best Validation F1-Score: {best_f1*100:.2f}%")
print(f"💾 Best LSTM model weights saved to: {MODEL_SAVE_PATH}")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
print("\n📋 Final Epoch Confusion Matrix (CNN-LSTM):")
print(f"   True Normal (TN): {tn:<6} | False Alarms (FP): {fp}")
print(f"   Missed Seizures (FN): {fn:<2} | Caught Seizures (TP): {tp}")
