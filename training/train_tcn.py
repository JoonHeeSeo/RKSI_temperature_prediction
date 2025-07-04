import os
import random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.tcn import TCNModel

# 0. 설정
SEED = 40
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Running on: {device}")

# 1. 데이터 로드
DATA_DIR  = os.path.join(os.getcwd(), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "rksi_weather.csv")
TEST_CSV  = os.path.join(DATA_DIR, "rksi_weather_2024.csv")

features = ["tmin", "tmax", "prcp", "wspd", "pres"]
target   = "tavg"

train_df = pd.read_csv(TRAIN_CSV, parse_dates=["time"]).dropna(subset=features+[target])
test_df  = pd.read_csv(TEST_CSV,  parse_dates=["time"]).dropna(subset=features+[target])

# 2. 정규화
os.makedirs("checkpoints", exist_ok=True)
scaler_X = StandardScaler().fit(train_df[features])
scaler_y = StandardScaler().fit(train_df[[target]])
joblib.dump(scaler_X, "checkpoints/scaler_X.pkl")
joblib.dump(scaler_y, "checkpoints/scaler_y.pkl")

X_train = scaler_X.transform(train_df[features])
y_train = scaler_y.transform(train_df[[target]]).flatten()
X_test  = scaler_X.transform(test_df[features])
y_test  = test_df[target].values

# 3. 시퀀스 생성
SEQ_LEN = 16

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(xs), np.array(ys)

X_tr, y_tr = create_sequences(X_train, y_train, SEQ_LEN)
X_te, y_te = create_sequences(X_test,  y_test,  SEQ_LEN)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_te_t = torch.tensor(X_te, dtype=torch.float32)

# 4. 시계열 분할
VAL_RATIO = 0.1
N = len(X_tr_t)
val_size = int(N * VAL_RATIO)
train_X, train_y = X_tr_t[:-val_size], y_tr_t[:-val_size]
val_X,   val_y   = X_tr_t[-val_size:], y_tr_t[-val_size:]

train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_X,   val_y),   batch_size=32, shuffle=False)

# 5. 모델/옵티마이저
model = TCNModel(input_size=len(features), seq_len=SEQ_LEN).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 6. 학습
best_val = float('inf')
for epoch in range(1, 51):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model_state': model.state_dict()}, "checkpoints/best_tcn.pth")

    if epoch <= 5 or epoch % 5 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"[{epoch:02d}/50] train={train_loss:.4f}, val={val_loss:.4f}, lr={lr:.1e}")

# 7. 평가
ckpt = torch.load("checkpoints/best_tcn.pth", map_location=device)
model.load_state_dict(ckpt['model_state'])
model.eval()
with torch.no_grad():
    preds_s = model(X_te_t.to(device)).cpu().numpy()

preds = scaler_y.inverse_transform(preds_s.reshape(-1,1)).flatten()
true  = y_te

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae  = mean_absolute_error(true, preds)
rmse = np.sqrt(mean_squared_error(true, preds))
print(f"▶ Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")