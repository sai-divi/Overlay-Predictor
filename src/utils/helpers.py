import os
import random
import numpy as np
from pathlib import Path


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def ensure_dirs(paths: list):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
