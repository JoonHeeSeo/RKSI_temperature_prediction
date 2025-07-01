from .base import BaseModel
import torch.nn as nn

class LinearModel(BaseModel):
    """BaseModel을 상속한 선형 회귀 모델"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch_size, feature_dim)
        return self.linear(x)