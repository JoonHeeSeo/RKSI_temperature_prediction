import torch.nn as nn
from .base import BaseModel

class GRUModel(BaseModel):
    """
    Gated Recurrent Unit(GRU) 기반 회귀 모델.

    Args:
        input_size (int): 입력 특성 수.
        hidden_size (int, optional): GRU 은닉 상태 크기. Defaults to 64.
        num_layers (int, optional): GRU 레이어 수. Defaults to 2.
        dropout (float, optional): 드롭아웃 비율. Defaults to 0.1.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # 마지막 time-step hidden state를 사용
        return self.fc(out[:, -1, :])
