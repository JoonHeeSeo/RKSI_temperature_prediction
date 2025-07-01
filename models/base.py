import abc
from torch import nn

class BaseModel(nn.Module, abc.ABC):
    """모든 모델이 상속받는 추상 베이스 클래스"""
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    @classmethod
    def add_model_specific_args(cls, parser):
        """
        해당 모델에 필요한 argparse 인자를 추가.
        예: hidden_size, num_layers 등
        """
        return parser