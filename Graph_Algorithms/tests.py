import networkx as nx


def test_graph1():
    graph = nx.Graph()
    graph.add_nodes_from(['y', 'a', 'm'])
    graph: nx.Graph = graph.to_directed()
    graph.add_edge('y', 'a')
    graph.add_edge('m', 'a')
    return graph


def test_graph2():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(10)])
    graph.add_edge('0', '1')
    graph.add_edge('1', '2')
    graph.add_edge('2', '3')
    graph.add_edge('2', '6')
    graph.add_edge('2', '9')
    graph.add_edge('3', '4')
    graph.add_edge('3', '5')
    graph.add_edge('6', '7')
    graph.add_edge('6', '8')
    return graph


def test_graph3():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(10)])
    graph.add_edge('0', '1')
    graph.add_edge('1', '3')
    graph.add_edge('1', '4')
    graph.add_edge('2', '3')
    graph.add_edge('3', '6')
    graph.add_edge('3', '7')
    graph.add_edge('4', '5')
    graph.add_edge('4', '8')
    graph.add_edge('6', '9')
    return graph


def test_graph4():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(6)])
    graph.add_edge('0', '1')
    graph.add_edge('1', '2')
    graph.add_edge('1', '4')
    graph.add_edge('3', '4')
    graph.add_edge('3', '5')
    return graph


def test_graph5():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(6)])
    graph.add_edge('0', '1')
    graph.add_edge('1', '2')
    graph.add_edge('2', '4')
    graph.add_edge('3', '4')
    graph.add_edge('4', '5')
    return graph


def test_graph6():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(7)])
    graph.add_edge('0', '1')
    graph.add_edge('1', '2')
    graph.add_edge('2', '3')
    graph.add_edge('2', '5')
    graph.add_edge('4', '5')
    graph.add_edge('4', '6')
    return graph


def test_graph7():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(7)])
    graph.add_edge('0', '1')
    graph.add_edge('1', '2')
    graph.add_edge('2', '4')
    graph.add_edge('3', '4')
    graph.add_edge('4', '5')
    graph.add_edge('5', '6')
    return graph
