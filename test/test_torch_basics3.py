import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# ================================
# 1. 데이터 불러오기 및 전처리
# ================================
df = pd.read_csv("data/rksi_weather_2023.csv", parse_dates=["time"])

# 사용할 변수들만 선택
features = ["tmin", "tmax", "prcp", "wspd", "pres"]
target = "tavg"

# 결측치 있는 행 제거
df = df[features + [target]].dropna()

# X, y 설정
X = df[features].values
y = df[target].values.reshape(-1, 1)

# ================================
# 2. 정규화 (평균 0, 표준편차 1)
# ================================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Torch 텐서 변환
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# ================================
# 3. 비선형 회귀 모델 (MLP) 정의
# ================================
model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
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
for epoch in range(epochs):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ================================
# 5. 결과 출력
# ================================
print("\n📈 학습된 파라미터:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy().flatten()}")
