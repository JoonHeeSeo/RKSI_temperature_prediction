import math
import torch
import torch.nn as nn
from .base import BaseModel


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor: same shape as x with positional encoding added.
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x


class TransformerModel(BaseModel):
    """
    Vanilla Transformer Encoder for time series regression.

    Args:
        input_size (int): Number of input features per time step.
        seq_len (int): Length of the input sequence.
        d_model (int): Dimension of the Transformer embedding.
        nhead (int): Number of attention heads.
        num_layers (int): Number of TransformerEncoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout probability.
    """
    def __init__(
        self,
        input_size: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Input projection to match d_model
        self.input_proj = nn.Linear(input_size, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # Regression head
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
        Returns:
            Tensor of shape (batch_size,) with predicted values.
        """
        # (batch, seq_len, features) -> (seq_len, batch, features)
        x = x.permute(1, 0, 2)
        # Input projection and scaling
        x = self.input_proj(x) * math.sqrt(self.d_model)
        # Add positional encoding
        x = self.pos_encoder(x)
        # Transformer encoding
        output = self.transformer_encoder(x)
        # Take the last time-step's output
        last = output[-1, :, :]  # (batch_size, d_model)
        # Regression
        out = self.regressor(last)  # (batch_size, 1)
        return out.squeeze(-1)