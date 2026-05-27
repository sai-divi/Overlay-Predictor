from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Any


class BaseModel(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Any:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str) -> "BaseModel":
        pass

    @abstractmethod
    def name(self) -> str:
        pass
