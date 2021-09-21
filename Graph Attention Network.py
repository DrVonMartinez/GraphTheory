# import tensorflow as tf
import networkx
import numpy as np
import pickle

graph = networkx.Graph()

v = np.arange(1, 10)
pickled_v = pickle.dumps(v)
graph.add_node(pickled_v, name='Vertex v')
for node in graph:
    loaded = pickle.loads(node)
    print(loaded)
    loaded += 1
    print(loaded)
    # pickle
    pickled_v = pickle.dumps(loaded)
for node in graph:
    loaded = pickle.loads(node)
    print(loaded)
