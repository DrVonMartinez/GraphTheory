from fractions import Fraction
from random import random

import networkx as nx
import numpy as np
import pandas as pd


def page_rank(graph: nx.Graph, beta: float, iterations: int = 50) -> pd.Series:
    """
    Determine Node importance via Random Walk
    :return:
    """
    if beta > 1 or beta < 0:
        raise ValueError("Beta value " + str(beta) + " must be between 0 and 1")
    stochastic_matrix: pd.DataFrame = nx.to_pandas_adjacency(graph)
    out_degree = stochastic_matrix.sum(axis=0)
    n = len(stochastic_matrix)
    importance = pd.Series(np.ones(n) / n, stochastic_matrix.index)
    power_matrix = beta * stochastic_matrix / out_degree + (1 - beta) * (np.ones((n, n)) / n)
    for i in range(iterations):
        importance = power_matrix @ importance
    return pd.Series(map(lambda x: Fraction.from_float(x).limit_denominator(), importance), index=importance.index)


def personalized_page_rank(graph: nx.Graph):
    """
    Ranks proximity of nodes to teleport nodes S
    :return:
    """


def random_walk(graph: nx.Graph):
    """
    Ranking with teleportation back to starting node S = {Q}
    :return:
    """


def pixie_random_walk(graph: nx.Graph, query_nodes: dict[str, float], steps: int, alpha: float = 0.5):
    """
    Ranking with teleportation back to starting node S = {Q}
    :return:
    """
    if alpha > 1 or alpha < 0:
        raise ValueError("Alpha value " + str(alpha) + " must be between 0 and 1")
    elif sum(query_nodes.values()) != 1:
        raise ValueError("Query node weight must add up to " + str(alpha) + " 1")

    stochastic_matrix: pd.DataFrame = nx.to_pandas_adjacency(graph)
    # out_degree = stochastic_matrix.sum(axis=0)
    # n = len(stochastic_matrix)
    # importance = pd.Series(np.ones(n) / n, stochastic_matrix.index)
    item_visit_count = {x: 0 for x in stochastic_matrix.index}

    def get_random_neighbor(node):
        neighbors = [x for x in nx.neighbors(graph, node)]
        return np.random.choice(neighbors, 1, replace=False)

    def sample_by_weight():
        return np.random.choice(query_nodes.keys(), len(query_nodes), replace=False, p=query_nodes.values())

    item = sample_by_weight()
    for i in range(steps):
        user = get_random_neighbor(item)
        item = get_random_neighbor(user)
        item_visit_count[item] += 1
        if random() < alpha:
            item = sample_by_weight()
