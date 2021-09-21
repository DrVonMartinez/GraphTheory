import networkx as nx
import numpy as np


def common_neighbors(graph: nx.Graph, u, v) -> int:
    if not graph.has_node(u):
        raise ValueError('Node ' + str(u) + ' not in Graph ' + str(graph))
    elif not graph.has_node(v):
        raise ValueError('Node ' + str(v) + ' not in Graph ' + str(graph))
    neighbors_u = list(graph.neighbors(u))
    neighbors_v = list(graph.neighbors(v))
    return len(np.intersect1d(neighbors_u, neighbors_v))


def jaccard_coef(graph: nx.Graph, u, v) -> float:
    if not graph.has_node(u):
        raise ValueError('Node ' + str(u) + ' not in Graph ' + str(graph))
    elif not graph.has_node(v):
        raise ValueError('Node ' + str(v) + ' not in Graph ' + str(graph))
    neighbors_u = list(graph.neighbors(u))
    neighbors_v = list(graph.neighbors(v))
    return len(np.intersect1d(neighbors_u, neighbors_v)) / len(np.union1d(neighbors_u, neighbors_v))


def adamic_adar_index(graph: nx.Graph, u, v) -> float:
    if not graph.has_node(u):
        raise ValueError('Node ' + str(u) + ' not in Graph ' + str(graph))
    elif not graph.has_node(v):
        raise ValueError('Node ' + str(v) + ' not in Graph ' + str(graph))
    intersect = np.intersect1d(list(graph.neighbors(u)), list(graph.neighbors(v)))
    return sum([1 / np.log(graph.degree(x)) for x in intersect])


def katz_index(graph: nx.Graph, b: float) -> np.ndarray:
    return np.linalg.inv(np.identity(len(graph.nodes)) - b * nx.to_numpy_array(graph)) - np.identity(len(graph.nodes))
