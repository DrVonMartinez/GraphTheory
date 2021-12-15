import networkx as nx

from Graph_Algorithms.Isomorphism import tree_center
from Graph_Algorithms.tests import test_graph1, test_graph2, test_graph3


def is_dag(graph: nx.Graph):
    return graph.is_directed() and [x for x in nx.simple_cycles(graph)] == []


if __name__ == '__main__':
    g = test_graph1()
    g2 = test_graph2()
    g3 = test_graph3()
    print(is_dag(g))
    print(tree_center(g2))
    print(tree_center(g3))
