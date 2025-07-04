import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.linear import LinearModel

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
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
train_csv = os.path.join(BASE_DIR, "data", "rksi_weather.csv")
test_csv  = os.path.join(BASE_DIR, "data", "rksi_weather_2024.csv")

features = ["tmin", "tmax", "prcp", "wspd", "pres"]
target   = "tavg"

train_df = pd.read_csv(train_csv, parse_dates=["time"]).dropna(subset=features + [target])
test_df  = pd.read_csv(test_csv,  parse_dates=["time"]).dropna(subset=features + [target])

# -----------------------------
# 2. 정규화 (train 기준)
# -----------------------------
scaler_X = StandardScaler().fit(train_df[features].values)
scaler_y = StandardScaler().fit(train_df[[target]].values)

X_train = scaler_X.transform(train_df[features].values)
y_train = scaler_y.transform(train_df[[target]].values)

X_test  = scaler_X.transform(test_df[features].values)
y_test  = test_df[[target]].values

# -----------------------------
# 3. 시퀀스 데이터셋 생성
# -----------------------------
SEQ_LEN = 7  # 지난 7일 데이터 사용

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        # 윈도우 슬라이딩으로 시퀀스 생성
        window = X[i : i + seq_len]
        xs.append(window.flatten())       # flatten to 1D
        ys.append(y[i + seq_len])        # 다음 날 평균
    return np.array(xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)

# Torch tensor
X_t = torch.tensor(X_train_seq, dtype=torch.float32)
Y_t = torch.tensor(y_train_seq, dtype=torch.float32)

# -----------------------------
# 4. 학습 & 검증 분할
# -----------------------------
BATCH      = 32
VAL_RATIO  = 0.1

dataset = TensorDataset(X_t, Y_t)
val_size   = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

loader_train = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
loader_val   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

# -----------------------------
# 5. 모델 정의
# -----------------------------
# input_dim = seq_len * num_features
model = LinearModel(input_dim=SEQ_LEN * len(features)).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# -----------------------------
# 6. 학습 루프
# -----------------------------
EPOCHS = 200
best_val = float('inf')
for ep in range(1, EPOCHS+1):
    model.train()
    running = 0.0
    for xb, yb in loader_train:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * xb.size(0)

    train_loss = running / train_size
    model.eval()
    with torch.no_grad():
        val_loss = sum(
            criterion(model(xv.to(device)), yv.to(device)).item() * xv.size(0)
            for xv, yv in loader_val
        ) / val_size

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "checkpoints/best_linear.pth")
    if ep % 20 == 0 or ep == 1:
        print(f"[{ep}/{EPOCHS}] train={train_loss:.4f}, val={val_loss:.4f}")

# -----------------------------
# 7. 추론 및 평가
# -----------------------------
model.load_state_dict(torch.load("checkpoints/best_linear.pth"))
model.eval()
with torch.no_grad():
    preds_scaled = model(
        torch.tensor(X_test_seq, dtype=torch.float32).to(device)
    ).cpu().numpy()

# 역변환
preds = scaler_y.inverse_transform(preds_scaled)
true  = y_test_seq
mae   = mean_absolute_error(true, preds)
rmse  = np.sqrt(mean_squared_error(true, preds))
print(f"\n▶ Test MAE: {mae:.2f} ℃, RMSE: {rmse:.2f} ℃")

# -----------------------------
# 8. 결과 요약
# -----------------------------
comp = pd.DataFrame({
    "true": true.flatten(),
    "pred": preds.flatten()
})
comp["abs_err"] = (comp.true - comp.pred).abs()
comp["pct_err"] = comp.abs_err / comp.true.abs() * 100
print(comp.head())

# -----------------------------
# 9. 결과 저장
# -----------------------------
from utils.metrics_utils import write_metrics
write_metrics(model_name='linear', mae=mae, rmse=rmse)