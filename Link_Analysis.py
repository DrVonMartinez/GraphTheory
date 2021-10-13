import networkx as nx
import numpy as np


def page_rank(graph: nx.Graph):
    """
    Determine Node importance via Random Walk
    :return:
    """
    stochastic_matrix = nx.to_numpy_matrix(graph)
    stochastic_matrix /= np.sum(stochastic_matrix, axis=0)
    print(stochastic_matrix)
    # rank_vector = np.sum(stochastic_matrix, axis=1)
    # print(rank_vector)


def personalized_page_rank(graph: nx.Graph):
    pass


def random_walk(graph: nx.Graph):
    pass
