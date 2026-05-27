import os
import random
import numpy as np
import tensorflow as tf
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(paths: list):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
