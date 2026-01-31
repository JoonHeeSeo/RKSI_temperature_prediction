import os, random, json, datetime as dt
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.feature_utils import add_time_features, TIME_FEATURES

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
DATA_DIR   = "data"
TRAIN_CSV  = os.path.join(DATA_DIR, "rksi_weather.csv")
TEST_CSV   = os.path.join(DATA_DIR, "rksi_weather_2024.csv")

train_df = pd.read_csv(TRAIN_CSV, parse_dates=["time"])
test_df  = pd.read_csv(TEST_CSV,  parse_dates=["time"])

# Add time features
train_df = add_time_features(train_df)
test_df  = add_time_features(test_df)

features = ["tmin", "tmax", "prcp", "wspd", "pres"] + TIME_FEATURES
target   = "tavg"

train_df = train_df[features + [target]].dropna()
test_df  = test_df[["time"] + features + [target]].dropna()

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
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)  # y_test_seq는 실측 스케일 그대로

# Torch 텐서
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_seq,  dtype=torch.float32)

# -----------------------------
# 4. 학습 & 검증 분할
# -----------------------------
BATCH      = 32
VAL_RATIO  = 0.1
EPOCHS     = 50
LR         = 1e-3
CHECKPOINT = "checkpoints/best_lstm.pth"

dataset = TensorDataset(X_train_t, y_train_t)
val_size   = int(len(dataset) * VAL_RATIO)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

# -----------------------------
# 5. 모델 선택 & 정의
# -----------------------------
from models.lstm import LSTMModel      # 필요하면 argparse로 동적으로 꺼내 써도 됨

model = LSTMModel(input_size=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# 6. 학습 루프
# -----------------------------
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
        torch.save(model.state_dict(), CHECKPOINT)

    if epoch % 10 == 0 or epoch == 1:
        print(f"[{epoch:3d}/{EPOCHS}]  train={train_loss:.4f}  val={v_loss:.4f}")

# -----------------------------
# 7. 추론 (2024)
# -----------------------------
model.load_state_dict(torch.load(CHECKPOINT))
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
print("\n[2024 Evaluation]")
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

# -----------------------------
# 9. 결과 저장
# -----------------------------
from utils.metrics_utils import write_metrics, write_predictions
write_metrics(model_name='lstm', mae=mae, rmse=rmse)
write_predictions(
    model_name='lstm',
    dates=test_df["time"].iloc[SEQ_LEN:].values,
    y_true=true,
    y_pred=preds
)