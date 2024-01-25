import tensorflow as tf
from keras.layers import Layer, Conv2DTranspose


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""

    def __init__(
        self, max_number_of_nodes=100, dropout=0.1, act=tf.nn.sigmoid, **kwargs
    ):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.output_dim = max_number_of_nodes

    def call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        inputs_transpose = tf.transpose(inputs)
        x = tf.matmul(inputs, inputs_transpose)
        x = self.act(x)
        outputs = tf.reshape(x, [-1])
        return x
