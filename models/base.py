import abc
import torch.nn as nn

class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """모든 모델의 공통 인터페이스."""

    @abc.abstractmethod
    def forward(self, x):
        """순전파 로직."""
        raise NotImplementedError