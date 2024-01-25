from tensorflow.keras import layers
import rustworkx as rx
import tensorflow as tf

from .layers import InnerProductDecoder
from spektral.layers import GeneralConv


class GAE(tf.keras.Model):
    def __init__(self, graph_node_count, decoder_hidden, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder_hidden = layers.Dense(embedding_size, activation="relu")
        self.encoding_layer = layers.Conv2DTranspose(
            graph_node_count, 128, activation="softmax"
        )
        self.decoding_layer = GeneralConv(aggregate="max")
        self.flatten = layers.Flatten()
        self.dense_layer = layers.Dense(decoder_hidden, activation="relu")

    def from_graph(self, graph: rx.PyDiGraph):
        return rx.digraph_adjacency_matrix(graph)

    def to_graph(self, adj_matrix):
        graph = rx.PyDiGraph()

        # Add nodes to the graph
        num_nodes = len(adj_matrix)
        for _ in range(num_nodes):
            graph.add_node(None)

        # Add edges based on adjacency matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] != 0:  # Assuming 0 means no edge
                    graph.add_edge(i, j, None)

        return graph

    def encode(self, embedding):
        x = self.encoder_hidden(embedding)
        tf.print(x.shape)
        x = tf.reshape(x, (-1, 1, 1, self.embedding_size))
        filters = self.encoding_layer(x)
        filters_sum = tf.reduce_sum(filters, axis=-1)
        return filters_sum

    def decode(self, adjacency_matrix):
        n_nodes = adjacency_matrix.shape[-1]  # Number of nodes in the graph
        node_features = tf.zeros((adjacency_matrix.shape[0], n_nodes, 1))
        conv_hidden = self.decoding_layer([node_features, adjacency_matrix])
        flattend = self.flatten(conv_hidden)
        result = self.dense_layer(flattend)
        return result
