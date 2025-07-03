import os, random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.mlp import MLPModel

# -----------------------------
# 0. 공통 설정
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ Running on: {device}")

# -----------------------------
# 1. 데이터 로드
# -----------------------------
DATA_DIR  = os.path.join(os.getcwd(), "data")
TRAIN_CSV = os.path.join(DATA_DIR, "rksi_weather.csv")
TEST_CSV  = os.path.join(DATA_DIR, "rksi_weather_2024.csv")

features = ["tmin", "tmax", "prcp", "wspd", "pres"]
target   = "tavg"

train_df = pd.read_csv(TRAIN_CSV, parse_dates=["time"]).dropna(subset=features + [target])
test_df  = pd.read_csv(TEST_CSV,  parse_dates=["time"]).dropna(subset=features + [target])

# -----------------------------
# 2. 정규화 (train 기준)
# -----------------------------
scaler_X = StandardScaler().fit(train_df[features].values)
scaler_y = StandardScaler().fit(train_df[[target]].values)

X_train = scaler_X.transform(train_df[features].values)
y_train = scaler_y.transform(train_df[[target]].values)

X_test  = scaler_X.transform(test_df[features].values)
y_test  = test_df[[target]].values  # 실제값

# -----------------------------
# 3. 시퀀스 데이터셋 생성
# -----------------------------
SEQ_LEN = 7

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)

# Torch tensor
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)

# -----------------------------
# 4. 학습 & 검증 분할
# -----------------------------
BATCH     = 32
VAL_RATIO = 0.1
EPOCHS    = 50
LR        = 1e-3
CHECKPT   = "checkpoints/best_mlp.pth"

dataset   = TensorDataset(X_train_t, y_train_t)
val_size  = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size

g_loader = torch.Generator().manual_seed(SEED)
train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=g_loader)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

# -----------------------------
# 5. 모델 정의
# -----------------------------
input_dim = SEQ_LEN * len(features)
model = MLPModel(input_dim=input_dim, hidden_size=64, num_layers=2, dropout=0.1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 6. 학습 루프
# -----------------------------
best_val = float('inf')
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xv, yv in val_loader:
            xv, yv = xv.to(device), yv.to(device)
            val_loss += criterion(model(xv), yv).item() * xv.size(0)
    val_loss /= val_size

    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, CHECKPT)

    if epoch % 10 == 0 or epoch == 1:
        print(f"[{epoch}/{EPOCHS}] train={train_loss:.4f}, val={val_loss:.4f}")

# -----------------------------
# 7. 스케일러 저장
# -----------------------------
joblib.dump(scaler_X, "checkpoints/scaler_X.pkl")
joblib.dump(scaler_y, "checkpoints/scaler_y.pkl")

# -----------------------------
# 8. 추론 및 평가
# -----------------------------
ckpt = torch.load(CHECKPT)
model.load_state_dict(ckpt['model_state'])
model.eval()

# 스케일러 로드
scaler_X = joblib.load("checkpoints/scaler_X.pkl")
scaler_y = joblib.load("checkpoints/scaler_y.pkl")

with torch.no_grad():
    preds_scaled = model(X_test_t.to(device)).cpu().numpy()

# 역변환
preds = scaler_y.inverse_transform(preds_scaled)
true  = y_test_seq
mae   = mean_absolute_error(true, preds)
rmse  = np.sqrt(mean_squared_error(true, preds))
print(f"\n▶ MLP Test MAE: {mae:.2f} ℃, RMSE: {rmse:.2f} ℃")

# -----------------------------
# 9. 결과 요약
# -----------------------------
comp = pd.DataFrame({'true': true.flatten(), 'pred': preds.flatten()})
comp['abs_err'] = (comp.true - comp.pred).abs()
comp['pct_err'] = comp.abs_err / comp.true.abs() * 100
print(comp.head())
