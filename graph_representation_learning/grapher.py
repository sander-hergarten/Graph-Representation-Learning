from .utils import config
import os
import tensorflow as tf
from pathlib import Path

SOURCE_DIR = os.environ["GRAPHER_MODELS_DIR"]
model_path = Path(SOURCE_DIR) / config["grapher"]["type"]

model = tf.keras.models.load_model(model_path)


def encode(embedding):
    numeric_graph_representation = model.encode(embedding)
    return model.to_graph(numeric_graph_representation)


def decode(graph):
    numeric_graph_representation = model.from_graph(graph)
    return model.decode(numeric_graph_representation)
