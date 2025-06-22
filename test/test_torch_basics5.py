import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

# ================================
# 1. 데이터 불러오기 및 분리
# ================================
df_train = pd.read_csv("data/rksi_weather_2023.csv", parse_dates=["time"])
df_test  = pd.read_csv("data/rksi_weather_2024.csv", parse_dates=["time"])

features = ["tmin", "tmax", "prcp", "wspd", "pres"]
target   = "tavg"

# 결측치 제거
df_train = df_train[features + [target]].dropna()
df_test  = df_test[features + [target]].dropna()

# 독립변수와 종속변수 분리
X_train = df_train[features].values
y_train = df_train[[target]].values  # 2D array

X_test  = df_test[features].values
y_test  = df_test[[target]].values

# ================================
# 2. 정규화 (Train 데이터 기준)
# ================================
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train_scaled = scaler_X.transform(X_train)
y_train_scaled = scaler_y.transform(y_train)

X_test_scaled  = scaler_X.transform(X_test)
# y_test는 평가 시 역변환 비교를 위해 스케일하지 않음

# ================================
# 3. 시퀀스 데이터 생성 함수
# ================================
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length])
    return np.array(xs), np.array(ys)

seq_length = 7  # 예: 지난 7일 데이터를 사용
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test,   seq_length)

# Tensor로 변환 및 DataLoader 준비
batch_size = 32
train_dataset = TensorDataset(
    torch.tensor(X_train_seq, dtype=torch.float32),
    torch.tensor(y_train_seq, dtype=torch.float32)
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ================================
# 4. LSTM 모델 정의
# ================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch_size, seq_length, input_size]
        out, _ = self.lstm(x)
        # 마지막 타임스텝의 출력만 사용
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMRegressor(input_size=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================================
# 5. 학습 루프
# ================================
epochs = 50
model.train()
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

# ================================
# 6. 예측 및 역변환
# ================================
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
    y_pred_test_scaled = model(X_test_tensor).numpy()

# y_test_seq는 원래 스케일(실제 값) 배열
# 예측값 역변환
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
y_true      = y_test_seq

# ================================
# 7. 평가 지표 계산
# ================================
mae = mean_absolute_error(y_true, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_true, y_pred_test))
print("\n📊 평가 결과:")
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")

# ================================
# 8. 결과 예시 출력
# ================================
comparison = pd.DataFrame({
    "tavg_true": y_true.flatten(),
    "tavg_pred": y_pred_test.flatten(),
    "abs_err":    np.abs(y_true.flatten() - y_pred_test.flatten()),
    "pct_err":    np.abs((y_true.flatten() - y_pred_test.flatten()) / y_true.flatten()) * 100
})
print("\n첫 5개 샘플 결과 비교:")
print(comparison.head())
