import torch.nn as nn
from .base import BaseModel

class MLPModel(BaseModel):
    """
    다층 퍼셉트론(MLP) 기반 회귀 모델.

    Args:
        input_dim (int): 입력 차원 (sequence_length * num_features).
        hidden_size (int, optional): 은닉층 크기. Defaults to 64.
        num_layers (int, optional): 은닉층 개수. Defaults to 2.
        dropout (float, optional): 드롭아웃 비율. Defaults to 0.1.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        # 마지막 회귀 출력층
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, seq_len, num_features)
        # MLP에 입력하기 위해 flatten
        batch_size = x.size(0)
        flat = x.view(batch_size, -1)
        return self.network(flat)
