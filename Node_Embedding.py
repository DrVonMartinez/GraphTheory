import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import keras
from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import Lambda


def average_node_embeddings():
    pass


def simple_node_similarity2(graph: nx.Graph):
    adjacency: pd.DataFrame = nx.to_pandas_adjacency(graph)
    tf_adjacency = tf.convert_to_tensor(adjacency, name=graph.name, dtype=tf.float32)
    z = tf.Variable(tf.random.uniform((1, tf_adjacency.shape[0])),  # 0, tf_adjacency.shape[0]),
                    trainable=True, dtype=tf.float32, name='Node Embedding')
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    for i in range(100):
        opt.minimize(
            lambda: tf.linalg.norm(tf.subtract(tf_adjacency, tf.matmul(z, z, transpose_a=True))),
            var_list=[z])
    return tf.transpose(z, name='Node Embedding')


if __name__ == '__main__':
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(['y', 'a', 'm'])
    g: nx.Graph = g.to_directed()
    g.add_edge('y', 'y')
    g.add_edge('y', 'a')
    g.add_edge('m', 'a')
    g.add_edge('a', 'y')
    g.add_edge('m', 'm')
    print(simple_node_similarity2(g))
