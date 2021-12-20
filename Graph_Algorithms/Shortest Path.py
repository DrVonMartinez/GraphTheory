import networkx as nx
import numpy as np
import pandas as pd

from Graph_Algorithms.Isomorphism import tree_center
from Graph_Algorithms.tests import *


def is_dag(graph: nx.Graph):
    return graph.is_directed() and [x for x in nx.simple_cycles(graph)] == []


def bfs(graph: nx.Graph, starting_node: str):
    """
    Pros:
        Shortest Path on graph with unweighted edges
    Neutral:
        All Pairs, Shortest Path Problem on unweighted graphs
    Cons:
        Shortest Path on graph with weighted edges
        Can't detect negative cycles

    Complexity: O(V + E)
    Graph Size: Large
    :param graph:
    :param starting_node:
    :return:
    """
    distance = {node: None for node in graph.nodes}
    distance[starting_node] = 0
    queue = [starting_node]
    while queue:
        node = queue.pop(0)
        adjacent_nodes = nx.neighbors(graph, node)
        for neighbor in adjacent_nodes:
            if distance[neighbor] is None:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def single_source_shortest_path(graph: nx.Graph, start: str):
    if start not in [n for n in graph.nodes]:
        raise KeyError(start + ' not in graph')
    if not is_dag(graph):
        raise ValueError('Graph must be a DAG')
    # distance = {node: np.inf for node in graph.nodes}
    distance = {node: None for node in graph.nodes}
    distance[start] = 0
    distance_keys = list(distance.keys())

    for i in range(len(graph)):
        node = distance_keys[i]
        if distance[node] is not None:
            adjacent_nodes = nx.neighbors(graph, node)
            for neighbor in adjacent_nodes:
                edge = graph.get_edge_data(node, neighbor)
                new_distance = distance[node] + edge['weight']
                if distance[neighbor] is None:
                    distance[neighbor] = new_distance
                else:
                    distance[neighbor] = min(distance[neighbor], new_distance)
    return distance


def dijkstra_algorithm(graph: nx.Graph, starting_node: str) -> tuple[dict, dict]:
    """
    Pros:
        Shortest Path on graph with weighted edges
    Neutral:
        Shortest Path on graph with unweighted edges
        All Pairs, Shortest Path Problem
    Cons:
        Can't detect negative cycles

    Complexity: O((V + E) logV)
    Graph Size: Large/Medium
    :param graph:
    :param starting_node:
    :return:
    """
    distance: dict[str, float] = {node: np.inf for node in graph.nodes}
    prev: dict[str, str] = {node: None for node in graph.nodes}
    priority_queue: list[tuple[str, float]] = [(starting_node, 0)]
    distance[starting_node] = 0
    while priority_queue:
        node, best_distance = priority_queue.pop(0)
        if distance[node] < best_distance:
            continue
        adjacent_nodes = nx.neighbors(graph, node)
        for neighbor in adjacent_nodes:
            edge = graph.get_edge_data(node, neighbor)
            if distance[neighbor] > distance[node] + edge['weight']:
                distance[neighbor] = distance[node] + edge['weight']
                prev[neighbor] = node
                priority_queue.append((neighbor, distance[neighbor]))
        priority_queue.sort(key=lambda x: x[-1])
    return distance, prev


def bellman_ford_alg(graph: nx.Graph, starting_node: str):
    """
    Pros:
        Can detect negative cycles
    Neutral:
        Shortest Path on graph with weighted edges
    Cons:
        Shortest Path on graph with unweighted edges
        Bad at All Pairs, Shortest Path Problem

    Complexity: O(VE)
    Graph Size: Medium/Small
    :param graph:
    :param starting_node:
    :return:
    """
    distance: dict[str, float] = {node: np.inf for node in graph.nodes}
    distance[starting_node] = 0
    for i in range(len(nx.nodes(graph))):
        for edge in nx.edges(graph):
            edge_from, edge_to = edge
            _edge_ = graph.get_edge_data(edge_from, edge_to)
            if distance[edge_from] + _edge_['weight'] < distance[edge_to]:
                distance[edge_to] = distance[edge_from] + _edge_['weight']

    for i in range(len(nx.nodes(graph))):
        for edge in nx.edges(graph):
            edge_from, edge_to = edge
            _edge_ = graph.get_edge_data(edge_from, edge_to)
            if distance[edge_from] + _edge_['weight'] < distance[edge_to]:
                distance[edge_to] -= np.inf
    return distance


def has_negative_cycles(graph: nx.Graph, starting_node: str):
    distance: dict[str, float] = bellman_ford_alg(graph, starting_node)
    return np.min(np.array([v for v in distance.values()])) == -np.inf


def floyd_warshall(graph: nx.Graph):
    """
    Pros:
        Good at All Pairs, Shortest Path Problem
        Can detect negative cycles
    Cons:
        Shortest Path on graph with unweighted/weighted edges

    Complexity: O(V^3)
    Graph Size: Small
    :param graph:
    :return:
    """
    adj_matrix: pd.DataFrame = nx.to_pandas_adjacency(graph, nonedge=np.inf)
    next_adj_matrix = pd.DataFrame(index=adj_matrix.index, columns=adj_matrix.columns)
    # print(next_adj_matrix)
    for node in graph.nodes:
        adj_matrix[node][node] = 0
    # print(adj_matrix)
    # dynamic_distance = np.zeros((len(graph), len(graph)))
    # nodes = [node for node in graph.nodes]
    index = {i: node for i, node in zip(range(len(graph)), graph.nodes)}
    for i in range(len(graph)):
        for j in range(len(graph)):
            for k in range(0, len(graph)):
                if adj_matrix[index[i]][index[k]] + adj_matrix[index[k]][index[j]] < adj_matrix[index[i]][index[j]]:
                    adj_matrix[index[i]][index[j]] = adj_matrix[index[i]][index[k]] + adj_matrix[index[k]][index[j]]
                    next_adj_matrix[index[i]][index[j]] = next_adj_matrix[index[i]][index[k]]
    for i in range(len(graph)):
        for j in range(len(graph)):
            for k in range(0, len(graph)):
                if adj_matrix[index[i]][index[k]] + adj_matrix[index[k]][index[j]] < adj_matrix[index[i]][index[j]]:
                    adj_matrix[index[i]][index[j]] = -np.inf
                    next_adj_matrix[index[i]][index[j]] = -1
    return adj_matrix,  # next_adj_matrix


if __name__ == '__main__':
    g1 = test_graph1()
    g2 = test_graph2()
    g3 = test_graph3()
    g4 = test_graph8()
    print("Is DAG")
    print(is_dag(g1))
    print("Tree Centering")
    print(tree_center(g2))
    print(tree_center(g3))
    print("BFS:")
    print(bfs(g3, '0'))
    # print('Single Source:')
    # print(single_source_shortest_path(g4, 'A'))
    print("Dijkstra's Algorithm:")
    print(dijkstra_algorithm(g3, '0'))
    print("Bellman Ford Algorithm:")
    print(bellman_ford_alg(test_negative_cycle(), '0'))
    print(bellman_ford_alg(test_negative_cycle3(), '0'))
    print("Has Negative Cycles:")
    print(has_negative_cycles(test_negative_cycle3(), '0'))
    print(has_negative_cycles(test_graph9(), '0'))
    print("Floyd Warshall:")
    print(floyd_warshall(test_dense_graph()))
    print(floyd_warshall(test_dense_graph2()))
    # print(floyd_warshall(g2))
