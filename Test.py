import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import Node_Embedding
import Page_Rank
import Link_Prediction as lp
import Node_Feature as nf
import Graphlet as grflt

graph = nx.Graph()
graph.add_nodes_from(['y', 'a', 'm'])
graph: nx.Graph = graph.to_directed()
graph.add_edge('y', 'y')
graph.add_edge('y', 'a')
graph.add_edge('m', 'a')
graph.add_edge('a', 'y')
graph.add_edge('m', 'm')
# print(Page_Rank.page_rank(graph, beta=0.8))
print(Node_Embedding.simple_node_similarity(graph))
'''
print(nx.to_numpy_array(graph))
print(nf.graphlet_degree_vector(graph2, 'a'))
print(grflt.bag_of_node_degrees(graph3))
print(grflt.bag_of_node_degrees(graph4))
print(grflt.print_graphlets(grflt.graphlet2(3, True)))
print(grflt.print_graphlets(grflt.distinct_graphlets(3, True)))
print(len(grflt.distinct_graphlets(3, True)))
print(grflt.print_graphlets(grflt.graphlet2(3, False)))
print(grflt.print_graphlets(grflt.distinct_graphlets(3, False)))
print(len(grflt.distinct_graphlets(3, False)))
print(len(grflt.distinct_graphlets(4, False)))
print(len(grflt.distinct_graphlets(5, False)))
print(grflt.graphlet_kernel(graph5, 3))

print(lp.common_neighbors(graph, 'a', 'b'))
print(lp.jaccard_coef(graph, 'a', 'b'))
print(lp.adamic_adar_index(graph, 'a', 'b'))
print(lp.katz_index(nx.from_numpy_array(np.array([[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1], [1, 1, 1, 0]])), ))
'''
