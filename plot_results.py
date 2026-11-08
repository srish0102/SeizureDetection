import matplotlib.pyplot as plt
import numpy as np

# Data from your actual console training runs
epochs = np.arange(1, 16)
mamba_loss = [1.0479, 0.7815, 0.2893, 0.1692, 0.0161, 0.0323, 0.0565, 0.0866, 0.0040, 0.2408, 0.0362, 0.0380, 0.0833, 0.0155, 0.0632]
lstm_loss = [1.1284, 0.7394, 0.7290, 0.2659, 0.4565, 0.1695, 0.1195, 0.2841, 0.0698, 0.1996, 0.0513, 0.1032, 0.0066, 0.2013, 0.0019]

plt.figure(figsize=(10, 5))
plt.plot(epochs, mamba_loss, label="Hybrid CNN-Mamba (Stable)", color="crimson", linewidth=2.5, marker='o')
plt.plot(epochs, lstm_loss, label="Hybrid CNN-LSTM (Unstable)", color="royalblue", linewidth=2, linestyle='--', marker='s')

plt.title("Optimization Loss Convergence: CNN-Mamba vs. CNN-LSTM", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Recall-Weighted Loss", fontsize=12)
plt.xticks(epochs)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(fontsize=11, loc="upper right")
plt.tight_layout()

# Save as portfolio image
plt.savefig("loss_convergence.png", dpi=300)
print("✅ Loss convergence plot saved as 'loss_convergence.png'!")
