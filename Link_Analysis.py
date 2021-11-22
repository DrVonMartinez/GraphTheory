from fractions import Fraction
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
    pass


def random_walk(graph: nx.Graph):
    pass
