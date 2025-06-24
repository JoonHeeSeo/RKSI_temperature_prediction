# rksi_daily_lstm.py
import os, random, json, datetime as dt
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 0. 공통 설정
# -----------------------------
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("▶ Running on:", device)

# -----------------------------
# 1. 데이터 로드
# -----------------------------
train_df = pd.read_csv("data/rksi_weather.csv",   parse_dates=["time"])
test_df  = pd.read_csv("data/rksi_weather_2024.csv", parse_dates=["time"])

features = ["tmin", "tmax", "prcp", "wspd", "pres"]
target   = "tavg"

train_df = train_df[features + [target]].dropna()
test_df  = test_df[features + [target]].dropna()

# -----------------------------
# 2. 정규화 (train 기준)
# -----------------------------
scaler_X = StandardScaler().fit(train_df[features])
scaler_y = StandardScaler().fit(train_df[[target]])

X_train = scaler_X.transform(train_df[features].values)
y_train = scaler_y.transform(train_df[[target]].values)

X_test  = scaler_X.transform(test_df[features].values)
y_test  = test_df[[target]].values          # 실측값(역변환 안 함)

# -----------------------------
# 3. 시퀀스 데이터셋 생성
# -----------------------------
SEQ_LEN = 7  # 지난 7일

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test, SEQ_LEN)  # y_test_seq는 실측 스케일 그대로

# Torch 텐서
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)

# -----------------------------
# 4. 학습 & 검증 분할
# -----------------------------
BATCH = 32
VAL_RATIO = 0.1

dataset = TensorDataset(X_train_t, y_train_t)
val_size = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

# -----------------------------
# 5. LSTM 모델 정의
# -----------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 마지막 스텝

model = LSTMRegressor(len(features)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# 6. 학습 루프
# -----------------------------
EPOCHS = 50
best_val = float("inf")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)
    train_loss = running / train_size

    # ---- 검증 ----
    model.eval()
    with torch.no_grad():
        v_loss = sum(
            criterion(model(xv.to(device)), yv.to(device)).item() * xv.size(0)
            for xv, yv in val_loader
        ) / val_size

    if v_loss < best_val:
        best_val = v_loss
        torch.save(model.state_dict(), "best_lstm.pth")

    if epoch % 10 == 0 or epoch == 1:
        print(f"[{epoch:3d}/{EPOCHS}]  train={train_loss:.4f}  val={v_loss:.4f}")

# -----------------------------
# 7. 추론 (2024)
# -----------------------------
model.load_state_dict(torch.load("best_lstm.pth"))
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test_t.to(device)).cpu().numpy()

# 역변환 → ℃
preds = scaler_y.inverse_transform(preds_scaled)
true  = y_test_seq  # (실측)

# -----------------------------
# 8. 평가 지표
# -----------------------------
mae  = mean_absolute_error(true, preds)
rmse = np.sqrt(mean_squared_error(true, preds))
print("\n📊 2024년 평가")
print(f"MAE  = {mae:.2f} ℃")
print(f"RMSE = {rmse:.2f} ℃")

# -----------------------------
# 9. 결과 요약 & 저장(선택)
# -----------------------------
comp = pd.DataFrame({
    "tavg_true": true.flatten(),
    "tavg_pred": preds.flatten()
})
comp["abs_err"] = (comp.tavg_true - comp.tavg_pred).abs()
comp["pct_err"] = comp.abs_err / comp.tavg_true.abs() * 100
print("\n〈상위 5개 샘플〉")
print(comp.head())

# comp.to_csv("rksi_2024_pred_vs_true.csv", index=False)
