import numpy as np
import networkx as nx


# softmax
# sigmoid

def random_walk(graph: nx.Graph, starting_node, number_of_steps: int) -> list:
    path = [starting_node]
    for i in range(number_of_steps):
        neighbors = [n for n in graph.neighbors(path[-1])]
        next_index = np.random.randint(0, len(neighbors))
        next_node = neighbors[next_index]
        print(neighbors, next_node)
        path.append(next_node)
    return path


def random_walk_embedding(graph: nx.Graph, random_samples: int) -> np.ndarray:
    """
    Higher values of random_samples will lead to higher bias on negative events and
    provides a more robust estimate
    :param graph:
    :param random_samples: 5 - 20 typically
    :return:
    """
    embedded_nodes = {x: i for x, i in zip(graph.nodes, range(len(graph.nodes)))}
    np.random.seed(0)
    node_embedding_np = np.reshape(np.random.normal(size=graph.size() ** 2), (graph.size(), graph.size()))
    negative_node_sampling = [x[0] for x in graph.degree for _ in range(x[1])]
    for _ in range(100):
        for u in graph.nodes:
            def negative_sampling(embedding: np.ndarray):
                total = np.array([0])
                for v in graph.neighbors(u):
                    node_u = np.reshape(embedding[embedded_nodes[u]], (1, graph.size()))
                    node_v = np.reshape(embedding[embedded_nodes[v]], (graph.size(), 1))
                    a = np.log(sigmoid(np.dot(node_u, node_v)))
                    b = 0
                    for i in range(random_samples):
                        node_i = np.reshape(embedding[embedded_nodes[negative_node_sampling[i]]], (graph.size(), 1))
                        b += np.log(sigmoid(np.dot(node_u, node_i)))
                    total += b - a
                return total.item()

            result = negative_sampling(node_embedding_np)
            node_embedding_np[embedded_nodes[u]] = node_embedding_np[embedded_nodes[u]] - 0.001 * result
    return node_embedding_np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(x))
    return expo / expo_sum


if __name__ == '__main__':
    test_graph = nx.Graph()
    test_graph: nx.Graph = test_graph.to_undirected()
    test_graph.add_nodes_from([str(i) for i in range(1, 13)])
    test_graph.add_edge('1', '2')
    test_graph.add_edge('1', '3')
    test_graph.add_edge('1', '4')
    test_graph.add_edge('2', '3')
    test_graph.add_edge('2', '8')
    test_graph.add_edge('3', '4')
    test_graph.add_edge('4', '5')
    test_graph.add_edge('5', '6')
    test_graph.add_edge('5', '7')
    test_graph.add_edge('5', '8')
    test_graph.add_edge('6', '7')
    test_graph.add_edge('8', '9')
    test_graph.add_edge('8', '11')
    test_graph.add_edge('9', '10')
    test_graph.add_edge('10', '11')
    test_graph.add_edge('10', '12')
    test_graph.add_edge('11', '12')
    # print(random_walk(test_graph, '4', 5))
    print(random_walk_embedding(test_graph, 15))
