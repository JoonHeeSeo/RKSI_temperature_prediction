import torch
import torch.nn as nn
from .base import BaseModel

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(BaseModel):
    """
    Temporal Convolutional Network (TCN) 기반 회귀 모델.

    Args:
        input_size (int): 입력 특성 수 (num_features).
        seq_len (int): 시퀀스 길이.
        num_channels (List[int]): TCN 레이어별 채널 수 리스트.
        kernel_size (int): 커널 크기.
        dropout (float): 드롭아웃 비율.
    """
    def __init__(self, input_size: int, seq_len: int,
                 num_channels=[32, 64], kernel_size=2, dropout=0.1):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size,
                                   num_channels=num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> permute to (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        y = self.tcn(x)           # (batch, channel, seq_len)
        out = y[:, :, -1]         # 마지막 time-step, shape (batch, channel)
        # linear expects (batch, in_features)
        return self.linear(out).squeeze(-1)
