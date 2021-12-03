import itertools
import networkx as nx
import numpy as np
import math


def graphlet_degree_vector(graph: nx.Graph, source) -> tuple[np.ndarray, list]:
    def relabel(label):
        __graph_history = label.copy()
        count = 1
        for j in label.nodes:
            if j == source:
                continue
            __graph_history = nx.relabel_nodes(__graph_history, {j: 'Other_' + str(count)})
            count += 1
        return __graph_history

    graphlet_set = []
    for j in range(2, max(dict(graph.degree()).values()) + 1):
        graphlet_set += graphlet(j)
    counts = np.zeros(len(graphlet_set))
    for subgraph in graphlet_set:
        for sub_nodes in itertools.combinations(graph.nodes(), len(subgraph.nodes)):
            sub_graph: nx.Graph = graph.subgraph(sub_nodes)
            if not sub_graph.has_node(source) or not nx.is_connected(sub_graph):
                continue
            x_edges = sum([sub_graph.degree[e] for e in sub_graph.neighbors(source)])
            test_edges = sum([subgraph.degree[e] for e in subgraph.neighbors('Source')])
            x_neighbors = sorted(list(map(lambda y: y[:5], relabel(sub_graph).neighbors(source))))
            test_neighbors = sorted(list(map(lambda y: y[:5], subgraph.neighbors('Source'))))
            option_1 = x_neighbors == test_neighbors
            option_2 = x_edges == test_edges
            if nx.is_isomorphic(sub_graph, subgraph) and option_1 and option_2:
                counts[graphlet_set.index(subgraph)] += 1
    return counts, list(map(lambda x: x.degree, graphlet_set))


def graphlet(size: int) -> list[nx.Graph]:
    graphlet_set = []
    graph: nx.Graph = nx.complete_graph(['Source'] + ['Other_' + str(i) for i in range(size - 1)])
    graphlet_set.append(graph)
    for remove_num_edges in range(1, graph.size() + 1):
        for j in itertools.combinations(graph.edges(), remove_num_edges):
            test_graph = graph.copy()
            test_graph.remove_edges_from(j)
            if not nx.is_connected(test_graph):
                continue
            elif all([not nx.is_isomorphic(test_graph, x) for x in graphlet_set]):
                graphlet_set.append(test_graph)
            else:
                check = []
                for x in graphlet_set:
                    if nx.is_isomorphic(test_graph, x):
                        x_edges = sum([x.degree[e] for e in x.neighbors('Source')])
                        test_edges = sum([test_graph.degree[e] for e in test_graph.neighbors('Source')])
                        x_neighbors = sorted(list(map(lambda y: y[:5], x.neighbors('Source'))))
                        test_neighbors = sorted(list(map(lambda y: y[:5], test_graph.neighbors('Source'))))
                        option_1 = x_neighbors != test_neighbors
                        option_2 = x_edges != test_edges
                        check.append(option_1 or option_2)
                if all(check):
                    graphlet_set.append(test_graph)
    return sorted(graphlet_set, key=lambda x: x.degree['Source'], reverse=True)


if __name__ == '__main__':
    total = 0
    for num in range(6, 7):
        current = len(list(map(lambda x: x.nodes, graphlet(num))))
        print(current)
        total += current
    print(total)
