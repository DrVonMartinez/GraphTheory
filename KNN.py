import networkx as nx
import matplotlib.pyplot as plt

knn = nx.Graph()
knn = knn.to_undirected()
cnn = nx.complete_graph(2)
knn.add_nodes_from(['a', 'b', 'c', 'd'])
knn.add_edge('a', 'b')
knn.add_edge('b', 'a')
knn.add_edge('a', 'c')
knn.add_edge('c', 'a')
knn.add_edge('c', 'd')
knn.add_edge('d', 'c')
# print(nx.k_nearest_neighbors(knn))
# print(nx.k_nearest_neighbors(cnn))

