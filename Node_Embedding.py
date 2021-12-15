import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import node2vec as n2v


def average_node_embeddings(graph: nx.Graph, embedding_dimension=1):
    raise NotImplementedError("This Function is not finished")
    adjacency: pd.DataFrame = nx.to_pandas_adjacency(graph)
    tf_adjacency = tf.convert_to_tensor(adjacency, name=graph.name, dtype=tf.float32)
    z = tf.Variable(tf.random.uniform((tf_adjacency.shape[0], embedding_dimension)),
                    trainable=True, dtype=tf.float32, name='Node Embedding')
    opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    for i in range(1000):
        opt.minimize(
            lambda: tf.math.softmax(tf.subtract(tf_adjacency, tf.matmul(z, z, transpose_b=True))),
            var_list=[z])
    print(tf.math.softmax(tf.subtract(tf_adjacency, tf.matmul(z, z, transpose_b=True))))
    return z.value()


def simple_node_similarity3(graph: nx.Graph, embedding_dimension=1):
    adjacency: pd.DataFrame = nx.to_pandas_adjacency(graph)
    tf_adjacency = tf.convert_to_tensor(adjacency, name=graph.name, dtype=tf.float32)
    z = tf.Variable(tf.random.uniform((tf_adjacency.shape[0], embedding_dimension)),
                    trainable=True, dtype=tf.float32, name='Node Embedding')
    opt = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(1000):
        opt.minimize(
            lambda: tf.linalg.norm(tf.subtract(tf_adjacency, tf.matmul(z, z, transpose_b=True))),
            var_list=[z])
    print(tf.linalg.norm(tf.subtract(tf_adjacency, tf.matmul(z, z, transpose_b=True))))
    return z.value()


def node2vec(graph: nx.Graph, num_negative_samples=15):
    adjacency: pd.DataFrame = nx.to_pandas_adjacency(graph)
    vol_g = adjacency.sum()
    negative_node_sampling = [x[0] for x in graph.degree for _ in range(x[1])]
    if num_negative_samples > len(negative_node_sampling):
        num_negative_samples = len(negative_node_sampling)
    print(negative_node_sampling)
    np.random.seed(0)
    embedded_nodes = {x: i for x, i in zip(graph.nodes, range(len(graph.nodes)))}
    node_embedding_np = tf.random.normal((graph.size(), graph.size())).numpy()
    node_embedding_tf = tf.convert_to_tensor(node_embedding_np, name="Node Embedding")

    for u in graph.nodes:
        def negative_sampling(embedding: tf.Variable):
            total = tf.zeros((1, 1))
            for v in graph.neighbors(u):
                node_u = tf.reshape(embedding[embedded_nodes[u]], (1, graph.size()))
                node_v = tf.reshape(embedding[embedded_nodes[v]], (graph.size(), 1))
                a = tf.math.log(tf.math.sigmoid(tf.matmul(node_u, node_v)))
                b = 0
                for i in range(num_negative_samples):
                    node_i = tf.reshape(embedding[embedded_nodes[negative_node_sampling[i]]], (graph.size(), 1))
                    b += tf.math.log(tf.math.sigmoid(tf.matmul(node_u, node_i)))
                total += b - a
            return total.numpy().item()

        result = negative_sampling(node_embedding_tf)
        print(result)

        node_embedding_np[embedded_nodes[u]] = tf.subtract(node_embedding_tf[embedded_nodes[u]], 0.01 * result)
        node_embedding_tf = tf.convert_to_tensor(node_embedding_np, name="Node Embedding")
    print(node_embedding_tf)


if __name__ == '__main__':
    g: nx.Graph = nx.Graph()
    g.add_nodes_from(['1', '2', '3', '4'])
    g: nx.Graph = g.to_directed()
    g.add_edge('1', '2')
    g.add_edge('1', '4')
    g.add_edge('2', '4')
    g.add_edge('3', '4')
    # print(simple_node_similarity3(g, 1))
    # print(average_node_embeddings(g, 1))
    # node2vec(g, 5)
    result = n2v.node2vec.Node2Vec(g, num_walks=5)
    # print(n2v.node2vec.nx.algorithms.)
