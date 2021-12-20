import networkx as nx


def test_graph1():
    graph = nx.Graph()
    graph.add_nodes_from(['y', 'a', 'm'])
    graph: nx.Graph = graph.to_directed()
    graph.add_edge('y', 'a', weight=1)
    graph.add_edge('m', 'a', weight=1)
    return graph


def test_graph2():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(10)])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('1', '2', weight=1)
    graph.add_edge('2', '3', weight=1)
    graph.add_edge('2', '6', weight=1)
    graph.add_edge('2', '9', weight=1)
    graph.add_edge('3', '4', weight=1)
    graph.add_edge('3', '5', weight=1)
    graph.add_edge('6', '7', weight=1)
    graph.add_edge('6', '8', weight=1)
    return graph


def test_graph3():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(10)])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('1', '3', weight=1)
    graph.add_edge('1', '4', weight=1)
    graph.add_edge('2', '3', weight=1)
    graph.add_edge('3', '6', weight=1)
    graph.add_edge('3', '7', weight=1)
    graph.add_edge('4', '5', weight=1)
    graph.add_edge('4', '8', weight=1)
    graph.add_edge('6', '9', weight=1)
    return graph


def test_graph4():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(6)])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('1', '2', weight=1)
    graph.add_edge('1', '4', weight=1)
    graph.add_edge('3', '4', weight=1)
    graph.add_edge('3', '5', weight=1)
    return graph


def test_graph5():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(6)])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('1', '2', weight=1)
    graph.add_edge('2', '4', weight=1)
    graph.add_edge('3', '4', weight=1)
    graph.add_edge('4', '5', weight=1)
    return graph


def test_graph6():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(7)])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('1', '2', weight=1)
    graph.add_edge('2', '3', weight=1)
    graph.add_edge('2', '5', weight=1)
    graph.add_edge('4', '5', weight=1)
    graph.add_edge('4', '6', weight=1)
    return graph


def test_graph7():
    graph = nx.Graph()
    graph.add_nodes_from([str(i) for i in range(7)])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('1', '2', weight=1)
    graph.add_edge('2', '4', weight=1)
    graph.add_edge('3', '4', weight=1)
    graph.add_edge('4', '5', weight=1)
    graph.add_edge('5', '6', weight=1)
    return graph


def test_graph8():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
    graph.add_edge('A', 'B', weight=3)
    graph.add_edge('A', 'C', weight=6)
    graph.add_edge('B', 'C', weight=4)
    graph.add_edge('B', 'D', weight=4)
    graph.add_edge('B', 'E', weight=11)
    graph.add_edge('C', 'D', weight=8)
    graph.add_edge('C', 'G', weight=11)
    graph.add_edge('D', 'E', weight=-4)
    graph.add_edge('D', 'F', weight=5)
    graph.add_edge('D', 'G', weight=2)
    graph.add_edge('E', 'H', weight=9)
    graph.add_edge('F', 'H', weight=1)
    graph.add_edge('G', 'H', weight=2)
    return graph


def test_graph9():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['0', '1', '2', '3', '4'])
    graph.add_edge('0', '1', weight=4)
    graph.add_edge('0', '2', weight=1)
    graph.add_edge('1', '3', weight=1)
    graph.add_edge('2', '1', weight=2)
    graph.add_edge('2', '3', weight=5)
    graph.add_edge('3', '4', weight=3)
    return graph


def test_negative_cycle():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['0', '1', '2', '3', '4', '5', '6'])
    graph.add_edge('0', '1', weight=4)
    graph.add_edge('0', '6', weight=2)
    graph.add_edge('1', '1', weight=-1)
    graph.add_edge('1', '2', weight=3)
    graph.add_edge('2', '4', weight=1)
    graph.add_edge('2', '3', weight=3)
    graph.add_edge('3', '5', weight=-2)
    graph.add_edge('4', '5', weight=2)
    graph.add_edge('6', '4', weight=2)
    return graph


def test_negative_cycle2():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['0', '1', '2', '3', '4', '5', '6'])
    graph.add_edge('0', '1', weight=1)
    graph.add_edge('0', '2', weight=1)
    graph.add_edge('1', '3', weight=4)
    graph.add_edge('2', '1', weight=1)
    graph.add_edge('3', '2', weight=-6)
    graph.add_edge('3', '4', weight=1)
    graph.add_edge('3', '5', weight=1)
    graph.add_edge('4', '5', weight=1)
    graph.add_edge('4', '6', weight=1)
    graph.add_edge('5', '6', weight=1)
    return graph


def test_negative_cycle3():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    graph.add_edge('0', '1', weight=5)
    graph.add_edge('1', '2', weight=20)
    graph.add_edge('1', '5', weight=30)
    graph.add_edge('1', '6', weight=60)
    graph.add_edge('2', '3', weight=10)
    graph.add_edge('2', '4', weight=75)
    graph.add_edge('3', '2', weight=-15)
    graph.add_edge('4', '9', weight=100)
    graph.add_edge('5', '4', weight=25)
    graph.add_edge('5', '6', weight=5)
    graph.add_edge('5', '8', weight=50)
    graph.add_edge('6', '7', weight=-50)
    graph.add_edge('7', '8', weight=-10)
    return graph


def test_dense_graph():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['A', 'B', 'C', 'D'])
    graph.add_edge('A', 'B', weight=4)
    graph.add_edge('A', 'C', weight=1)
    graph.add_edge('A', 'D', weight=9)
    graph.add_edge('B', 'A', weight=3)
    graph.add_edge('B', 'C', weight=6)
    graph.add_edge('B', 'D', weight=11)
    graph.add_edge('C', 'A', weight=4)
    graph.add_edge('C', 'B', weight=1)
    graph.add_edge('C', 'D', weight=2)
    graph.add_edge('D', 'A', weight=6)
    graph.add_edge('D', 'B', weight=5)
    graph.add_edge('D', 'C', weight=-4)
    return graph


def test_dense_graph2():
    graph = nx.Graph()
    graph: nx.Graph = graph.to_directed()
    graph.add_nodes_from(['A', 'B', 'C', 'D'])
    graph.add_edge('A', 'B', weight=4)
    graph.add_edge('A', 'C', weight=1)
    graph.add_edge('B', 'C', weight=6)
    graph.add_edge('C', 'A', weight=4)
    graph.add_edge('C', 'B', weight=1)
    graph.add_edge('C', 'D', weight=2)
    return graph
