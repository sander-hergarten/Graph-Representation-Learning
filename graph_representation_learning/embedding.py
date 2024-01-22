from ..utils import config
import tensorflow as tf
import os
from pathlib import Path

SOURCE_DIR = os.environ["SOURCE_DIR"]
model_path = Path(SOURCE_DIR) / config["embedding_model"] / config["dataset"]


def from_observations(observations):
    model = _load_model_from_file()
    embeddings = model.embed(observations)
    return embeddings


def _load_model_from_file() -> tf.keras.Model:
    return tf.keras.models.load_model(model_path)
