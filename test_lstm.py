# ======================================
# EMG → CyberGlove LSTM (Corrected)
# ======================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

# --------------------------------------
# 1️⃣ Load MATLAB data
# --------------------------------------
mat = loadmat("S1_A1_E1.mat")

emg = mat["emg"].astype(np.float32)        # (T, 10)
glove = mat["glove"].astype(np.float32)    # (T, 22)
repetition = mat["repetition"].squeeze()   # (T,)

print("EMG:", emg.shape)
print("Glove:", glove.shape)

# --------------------------------------
# 2️⃣ Normalize (using TRAIN stats only)
# --------------------------------------
# Repetition-based split (critical)
train_mask = repetition <= 4
test_mask  = repetition > 4

emg_train = emg[train_mask]
glove_train = glove[train_mask]

emg_mean, emg_std = emg_train.mean(axis=0), emg_train.std(axis=0)
glove_mean, glove_std = glove_train.mean(axis=0), glove_train.std(axis=0)

emg = (emg - emg_mean) / emg_std
glove = (glove - glove_mean) / glove_std

# --------------------------------------
# 3️⃣ Dataset class
# --------------------------------------
class EMGGloveDataset(Dataset):
    def __init__(self, emg, glove, seq_len):
        self.emg = emg
        self.glove = glove
        self.seq_len = seq_len

    def __len__(self):
        return len(self.emg) - self.seq_len

    def __getitem__(self, idx):
        X = self.emg[idx:idx + self.seq_len]
        y = self.glove[idx + self.seq_len]
        return torch.tensor(X), torch.tensor(y)

seq_len = 20

train_dataset = EMGGloveDataset(
    emg[train_mask],
    glove[train_mask],
    seq_len
)

test_dataset = EMGGloveDataset(
    emg[test_mask],
    glove[test_mask],
    seq_len
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------------------
# 4️⃣ Model (smaller + dropout)
# --------------------------------------
class EMGtoGloveLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 22)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMGtoGloveLSTM().to(device)

# --------------------------------------
# 5️⃣ Training setup
# --------------------------------------
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# --------------------------------------
# 6️⃣ Training loop with early stopping
# --------------------------------------
num_epochs = 100
patience = 8
best_val = float("inf")
counter = 0

for epoch in range(num_epochs):
    # ---- train ----
    model.train()
    train_loss = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            val_loss += criterion(model(X), y).item() * X.size(0)

    val_loss /= len(test_loader.dataset)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train: {train_loss:.5f} | "
        f"Val: {val_loss:.5f}"
    )

    # ---- early stopping ----
    if val_loss < best_val:
        best_val = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_emg_glove_model.pt")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# --------------------------------------
# 7️⃣ Test prediction (denormalized)
# --------------------------------------
model.load_state_dict(torch.load("best_emg_glove_model.pt"))
model.eval()

with torch.no_grad():
    X_test = torch.tensor(
        emg[test_mask][-seq_len:],
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    y_pred_norm = model(X_test).cpu().numpy()
    y_pred = y_pred_norm * glove_std + glove_mean

print("\nPredicted glove joint values:")
print(y_pred)
