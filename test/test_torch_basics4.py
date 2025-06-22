import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

# ================================
# 1. 데이터 불러오기 및 분리
# ================================
# 학습용: 2023년
df_train = pd.read_csv("data/rksi_weather_2023.csv", parse_dates=["time"])
# 평가용: 2024년
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

# Torch 텐서로 변환
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)

# ================================
# 3. MLP 모델 정의
# ================================
model = nn.Sequential(
    nn.Linear(len(features), 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ================================
# 4. 학습 루프
# ================================
epochs = 200
for epoch in range(1, epochs+1):
    y_pred_t = model(X_train_t)
    loss = criterion(y_pred_t, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

# ================================
# 5. 2024 데이터 예측 및 역변환
# ================================
model.eval()
with torch.no_grad():
    y_pred_test_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

# 원래 스케일로 복원
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
y_true      = y_test  # already original scale

# ================================
# 6. 평가 지표 계산
# ================================
# MAPE: 평균절대백분율오차
mape = mean_absolute_percentage_error(y_true, y_pred_test) * 100
# RMSE: 평균제곱근오차
rmse = np.sqrt(mean_squared_error(y_true, y_pred_test))

print("\n📊 평가 결과:")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} °C")

# ================================
# 7. 결과 예시 출력 (원하는 만큼 출력)
# ================================
# 예: 처음 5개 샘플 비교
comparison = pd.DataFrame({
    "tavg_true": y_true.flatten(),
    "tavg_pred": y_pred_test.flatten(),
    "abs_err":    np.abs(y_true.flatten() - y_pred_test.flatten()),
    "pct_err":    np.abs((y_true.flatten() - y_pred_test.flatten()) / y_true.flatten()) * 100
})
print("\n첫 5개 샘플 결과 비교:")
print(comparison.head())
