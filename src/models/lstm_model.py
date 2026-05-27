import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import List, Tuple, Any

from src.models.base import BaseModel


class LSTMModel(BaseModel):
    def __init__(
        self,
        units: List[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        sequence_length: int = 60,
        output_units: int = 1,
        output_activation: str = "linear",
        loss: str = "mse",
    ):
        self.units = units or [128, 64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.output_units = output_units
        self.output_activation = output_activation
        self.loss = loss
        self.model: Sequential = None

    def _build(self, input_shape: Tuple[int, int]):
        model = Sequential()
        for i, units in enumerate(self.units):
            return_seq = i < len(self.units) - 1
            model.add(LSTM(units, return_sequences=return_seq, input_shape=input_shape))
            model.add(Dropout(self.dropout))
        model.add(Dense(self.output_units, activation=self.output_activation))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=self.loss)
        self.model = model

    def train(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> Any:
        self._build((X_train.shape[1], X_train.shape[2]))
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0).flatten()

    def predict_with_confidence(self, X: np.ndarray, num_samples: int = 30) -> tuple:
        preds = np.array([self.model.predict(X, verbose=0).flatten() for _ in range(num_samples)])
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        return mean_pred, std_pred

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str) -> "LSTMModel":
        self.model = load_model(path)
        return self

    @property
    def name(self) -> str:
        return "lstm"


def lstm_for_classification(**kwargs):
    return LSTMModel(
        output_units=3,
        output_activation="softmax",
        loss="sparse_categorical_crossentropy",
        **kwargs,
    )
