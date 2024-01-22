from .utils import config
import tensorflow as tf
import os
from pathlib import Path

SOURCE_DIR = os.environ["EMBEDDING_MODELS_DIR"]
model_path = (
    Path(SOURCE_DIR)
    / config["embedding"]["embedding_model"]
    / config["dataset"]["name"]
)


def from_observations(observations, model_path: os.PathLike = model_path):
    model = tf.keras.models.load_model(model_path)
    embeddings = model(observations)
    return embeddings
