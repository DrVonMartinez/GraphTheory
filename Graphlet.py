import itertools
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def bag_of_node_degrees(graph: nx.Graph) -> np.ndarray:
    max_degree = len(graph.nodes)
    bag_of_nodes = np.zeros(max_degree)
    for node in graph.nodes:
        bag_of_nodes[graph.degree[node]] += 1
    return bag_of_nodes


def __color_refinement(graph: nx.Graph, size: int) -> np.ndarray:
    history = []
    max_color = -1
    prior = graph.copy()
    for node in prior.nodes:
        nx.set_node_attributes(prior, {node: {'Neighbor Colors': [], 'Color': 1}})
    history += list(map(lambda x: x[1], prior.nodes('Color')))
    for i in range(size):
        local = prior.copy()
        for node in local.nodes:
            local.nodes[node]['Neighbor Colors'] = []
            for neighbor in local.neighbors(node):
                local.nodes[node]['Neighbor Colors'].append(local.nodes[neighbor]['Color'])
        print(dict(local.nodes(data=True)))
        for node in local.nodes:
            local.nodes[node]['Color'] += sum(local.nodes[node]['Neighbor Colors'])
        color_set = list(map(lambda x: x[1], local.nodes('Color')))
        if max(color_set) > max_color:
            max_color = max(color_set)
        history += color_set
        prior = local.copy()
    counts = Counter(history)
    print(counts)
    wl_kernel = {key: counts[key + 1] for key in range(max_color)}
    return np.reshape(np.asarray(list(wl_kernel.values())), (1, max_color))


def color_refinement2(graph_1: nx.Graph, graph_2: nx.Graph, size: int) -> int:
    wl_kernel_1 = [0]
    wl_kernel_2 = [0]
    unused_color = 2
    color_hash: dict[str, int] = {'1': 1}
    option = nx.union(graph_1, graph_2, rename=('1_', '2_'))
    for node in option.nodes:
        nx.set_node_attributes(option, {node: {'Color': 1, 'Mapping': ''}})
    wl_kernel_1[0] = len(graph_1.nodes)
    wl_kernel_2[0] = len(graph_2.nodes)
    for i in range(size):
        local = option.copy()
        mapping_set = []
        for node in local.nodes:
            neighbor_colors = sorted([local.nodes[neighbor]['Color'] for neighbor in local.neighbors(node)])
            mapping = str(local.nodes[node]['Color']) + ':' + ''.join(map(str, neighbor_colors))
            local.nodes[node]['Mapping'] = mapping
            mapping_set.append(mapping)
        for mapping in sorted(mapping_set):
            if mapping not in color_hash.keys():
                color_hash[mapping] = unused_color
                wl_kernel_1.append(0)
                wl_kernel_2.append(0)
                unused_color += 1
        results = dict(option.nodes(data=True))
        print({key[2:]: results[key] for key in filter(lambda x: '1_' in x, results.keys())})
        print({key[2:]: results[key] for key in filter(lambda x: '2_' in x, results.keys())})
        for node in filter(lambda x: '1_' in x, local.nodes):
            local.nodes[node]['Color'] = int(color_hash[local.nodes[node]['Mapping']])
            wl_kernel_1[local.nodes[node]['Color'] - 1] += 1
        for node in filter(lambda x: '2_' in x, local.nodes):
            local.nodes[node]['Color'] = int(color_hash[local.nodes[node]['Mapping']])
            wl_kernel_2[local.nodes[node]['Color'] - 1] += 1
        option = local.copy()
    results = dict(option.nodes(data=True))
    print({key[2:]: results[key] for key in filter(lambda x: '1_' in x, results.keys())})
    print({key[2:]: results[key] for key in filter(lambda x: '2_' in x, results.keys())})
    print(color_hash)
    print(len(color_hash))
    # wl_kernel_1 = np.reshape(np.array(wl_kernel_1), (len(wl_kernel_1), 1))
    # wl_kernel_2 = np.reshape(np.array(wl_kernel_2), (len(wl_kernel_2), 1))
    return (np.array(wl_kernel_1).T @ np.array(wl_kernel_2)).item()


def color_refinement(graph_1: nx.Graph, graph_2: nx.Graph, size: int) -> int:
    wl_kernel_1 = __color_refinement(graph_1, size)
    wl_kernel_2 = __color_refinement(graph_2, size)
    if wl_kernel_1.size < wl_kernel_2.size:
        temp_wl_1 = np.zeros_like(wl_kernel_2)
        for i in range(wl_kernel_1.size):
            temp_wl_1[0, i] = wl_kernel_1[0, i]
        wl_kernel_1 = temp_wl_1
    elif wl_kernel_1.size > wl_kernel_2.size:
        temp_wl_2 = np.zeros_like(wl_kernel_1)
        for i in range(wl_kernel_2.size):
            temp_wl_2[0, i] = wl_kernel_2[0, i]
        wl_kernel_2 = temp_wl_2
    print(wl_kernel_1)
    print(wl_kernel_2)
    return (wl_kernel_1 @ wl_kernel_2.T).item()


def graphlet_kernel(graph_1: nx.Graph, graph_2: nx.Graph, size: int) -> np.ndarray:
    graphlet_options = distinct_graphlets(size, connected=False)
    print(print_graphlets(graphlet_options))
    bag_of_graphlet_kernels_1 = np.zeros(len(graphlet_options))
    bag_of_graphlet_kernels_2 = np.zeros(len(graphlet_options))
    for i in range(len(graphlet_options)):
        subgraph = graphlet_options[i]
        for sub_nodes in itertools.combinations(graph_1.nodes(), size):
            sub_graph = graph_1.subgraph(sub_nodes)
            if nx.is_isomorphic(sub_graph, subgraph):
                bag_of_graphlet_kernels_1[i] += 1
        for sub_nodes in itertools.combinations(graph_2.nodes(), size):
            sub_graph = graph_2.subgraph(sub_nodes)
            if nx.is_isomorphic(sub_graph, subgraph):
                bag_of_graphlet_kernels_2[i] += 1
    h_g_1 = bag_of_graphlet_kernels_1 / np.sum(bag_of_graphlet_kernels_1)
    h_g_2 = bag_of_graphlet_kernels_2 / np.sum(bag_of_graphlet_kernels_2)
    return h_g_1.transpose() @ h_g_2


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


def graphlet2(size: int, connected: bool) -> list[nx.Graph]:
    graphlet_set = []
    graph: nx.Graph = nx.complete_graph(['Source'] + ['Other_' + str(i) for i in range(size - 1)])
    graphlet_set.append(graph)
    for remove_num_edges in range(1, graph.size() + 1):
        for j in itertools.combinations(graph.edges(), remove_num_edges):
            test_graph = graph.copy()
            test_graph.remove_edges_from(j)
            if not nx.is_connected(test_graph) and connected:
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
    return sorted(graphlet_set, key=lambda x: [str(x.degree[deg]) for deg in x.nodes], reverse=True)


def print_graphlets(graphlet_set: list[nx.Graph]) -> list[str]:
    return list(map(lambda x: x.degree, graphlet_set))


def distinct_graphlets(size: int, connected: bool) -> list[nx.Graph]:
    graphlet_set = []
    graph: nx.Graph = nx.complete_graph(['Source'] + ['Other_' + str(i) for i in range(size - 1)])
    graphlet_set.append(graph)
    for remove_num_edges in range(1, graph.size() + 1):
        for j in itertools.combinations(graph.edges(), remove_num_edges):
            test_graph = graph.copy()
            test_graph.remove_edges_from(j)
            if not nx.is_connected(test_graph) and connected:
                continue
            elif all([not nx.is_isomorphic(test_graph, x) for x in graphlet_set]):
                graphlet_set.append(test_graph)
    return sorted(graphlet_set, key=lambda x: [str(x.degree[deg]) for deg in x.nodes], reverse=True)


if __name__ == '__main__':
    g = nx.Graph()
    g: nx.Graph = g.to_undirected()
    g.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])
    g.add_edge('a', 'b')
    g.add_edge('b', 'c')
    g.add_edge('b', 'e')
    g.add_edge('b', 'f')
    g.add_edge('c', 'd')
    g.add_edge('c', 'e')
    g.add_edge('d', 'e')
    g2 = nx.Graph()
    g2: nx.Graph = g2.to_undirected()
    g2.add_nodes_from(['a', 'b', 'c', 'd', 'e', 'f'])
    g2.add_edge('a', 'b')
    g2.add_edge('b', 'c')
    g2.add_edge('b', 'd')
    g2.add_edge('b', 'e')
    g2.add_edge('c', 'd')
    g2.add_edge('d', 'e')
    g2.add_edge('e', 'f')
    nx.draw(g, with_labels=True)
    plt.show()
    nx.draw(g2, with_labels=True)
    plt.show()
    print(color_refinement2(g, g2, 2))
    '''
    nx.draw(g)
    plt.show()
    cr_1 = __color_refinement(g, 2)
    cr_2 = __color_refinement(g2, 2)
    print(cr_1, cr_1.shape)
    print(cr_2, cr_2.shape)
    print(color_refinement(g, g2, 2))
    '''
