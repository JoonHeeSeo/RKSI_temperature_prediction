import torch
import torch.nn as nn
import torch.optim as optim

# pip install torch torchvision torchaudio

# 1. PyTorch 버전과 CUDA 여부 확인
print("✅ PyTorch version:", torch.__version__)
print("🧠 CUDA available:", torch.cuda.is_available())

# 2. 예제 데이터 생성 (선형 관계 y = 2x + 1)
X = torch.linspace(0, 10, 100).unsqueeze(1)  # shape: (100, 1)
y = 2 * X + 1 + 0.5 * torch.randn(X.size())  # 약간의 노이즈 추가

# 3. 선형 회귀 모델 정의
model = nn.Linear(1, 1)  # 입력 1차원, 출력 1차원

# 4. 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. 학습 루프
epochs = 100
for epoch in range(epochs):
    # 순전파
    y_pred = model(X)

    # 손실 계산
    loss = criterion(y_pred, y)

    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 6. 결과 확인
print("\n📈 학습된 파라미터:")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.numpy().flatten()}")
