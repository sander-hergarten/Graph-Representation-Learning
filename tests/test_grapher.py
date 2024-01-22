import pytest
import tensorflow as tf
from tensorflow.keras import layers
from graph_representation_learning import grapher


class GraphModel(tf.keras.Model):
    def __init__(self):
        super(GraphModel, self).__init__()
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(10)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

    def encode(self, embedding):
        encoded = self.dense1(embedding)
        return encoded

    def to_graph(self, numeric_graph_representation):
        graph = self.dense2(numeric_graph_representation)
        return graph

    def decode(self, numeric_graph_representation):
        decoded = self.dense1(numeric_graph_representation)
        return decoded

    def from_graph(self, graph):
        numeric_graph_representation = self.dense2(graph)
        return numeric_graph_representation


def test_encode(monkeypatch):
    # Create a test model
    test_model = GraphModel()

    # Monkeypatch the model in the grapher module
    monkeypatch.setattr(grapher, "model", test_model)

    # Create a test embedding
    test_embedding = tf.random.uniform((32, 2))

    # Test the encode function
    encoded = grapher.encode(test_embedding)
    assert encoded.shape == (32, 10)


def test_decode(monkeypatch):
    # Create a test model
    test_model = GraphModel()

    # Monkeypatch the model in the grapher module
    monkeypatch.setattr(grapher, "model", test_model)

    # Create a test encoded representation
    test_encoded = tf.random.uniform((10, 10))

    # Test the decode function
    decoded = grapher.decode(test_encoded)
    assert decoded.shape == (10, 64)
