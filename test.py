from GAE import GAE
import tensorflow as tf
import rustworkx.generators as rxg

embedding_size = 128

graph = rxg.directed_path_graph(200)
model = GAE(2, 100, 100)

embedding = tf.random.uniform((1, embedding_size))

encoding = model.encode(embedding)
print(encoding)
