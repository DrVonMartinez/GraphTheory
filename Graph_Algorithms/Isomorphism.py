import networkx as nx
from Tree import root_tree, TreeNode
from tests import test_graph4, test_graph5, test_graph6, test_graph7


def tree_center(graph: nx.Graph, show_steps=False) -> list:
    if graph.is_directed():
        raise ValueError('Graph is Directed')
    temp_graph = graph.copy()
    if show_steps:
        print(dict(temp_graph.degree))
    while len(temp_graph.nodes) > 2:
        temp_graph_nodes = dict(temp_graph.degree)
        for node in temp_graph_nodes:
            if temp_graph_nodes[node] == 1:
                temp_graph.remove_node(node)
        if show_steps:
            print(dict(temp_graph.degree))
    return list(temp_graph.nodes)


def tree_isomorphism(graph1: nx.Graph, graph2: nx.Graph):
    center1 = tree_center(graph1)
    center2 = tree_center(graph2)

    print(center1)
    print(center2)
    rooted_tree1 = root_tree(graph1, center1[0])
    tree1_encoding = encode(rooted_tree1)
    print(tree1_encoding)

    for center in center2:
        rooted_tree2 = root_tree(graph2, center)
        tree2_encoding = encode(rooted_tree2)
        print(tree2_encoding)
        if tree1_encoding == tree2_encoding:
            return True
    return False


def encode(node: TreeNode) -> str:
    if not node:
        return ''
    labels = []
    for child in node.children:
        labels.append(encode(child))
    labels.sort()
    return '(' + ''.join(labels) + ')'


if __name__ == '__main__':
    g1 = test_graph4()
    g2 = test_graph5()
    print(tree_isomorphism(g1, g2))
    g3 = test_graph6()
    g4 = test_graph7()
    print(tree_isomorphism(g3, g4))
