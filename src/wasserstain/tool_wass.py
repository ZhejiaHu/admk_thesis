"""
Tools to compute Wasserstein distance between two graphs
"""

import numpy as np  
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.spatial import distance

def get_graph_union(g1: nx.Graph, g2: nx.Graph, weight: str = 'length'):

    G_union = nx.Graph()
    numv_1 = len(g1.nodes())

    for node in g1.nodes():
        G_union.add_node(node)
        G_union.nodes[node]["pos"] = g1.nodes[node]["pos"]

    for node in g2.nodes():
        G_union.add_node(node + numv_1)
        G_union.nodes[node + numv_1]["pos"] = g2.nodes[node]["pos"]

    for edge in g1.edges():
        G_union.add_edge(edge[0], edge[1])

    for edge in g2.edges():
        G_union.add_edge(edge[0] + numv_1, edge[1] + numv_1)

    if weight == "unit":
        for edge in sorted(G_union.edges):
            G_union.edges[edge]["weight"] = 1

    elif weight == "length":
        for edge in sorted(G_union.edges):
            G_union.edges[edge]["weight"] = distance.euclidean(
                G_union.nodes[edge[0]]["pos"], G_union.nodes[edge[1]]["pos"]
            )

    return G_union


def get_rhs(g1: nx.Graph, g2: nx.Graph,g_union: nx.Graph = None,
            weight: str = 'length',normalize: bool = True):

    if g_union is None:
        g_union = get_graph_union(g1,g2,weight=weight)

    rhs = get_f(g_union, g1, g2, oriented=True, normalize=normalize)

    return rhs


def get_f(g_union: nx.Graph, g1: nx.Graph, g2: nx.Graph, oriented: bool = True, normalize: bool = True):
    """Takes in input the source, sink and union graphs and returns the rhs balanced and normalized, if normalize=True"""

    q1 = find_q(g_union, list(g1.edges()))
    q2 = find_q(g_union, list(g2.edges()))

    I = incidence_matrix(g_union, oriented=oriented)
    f0 = np.dot(I, q1 - q2)
    f0 = balance_forcing(f0, normalize=normalize)
    f = f0.tolist()
    return f[0]


def incidence_matrix(graph: nx.Graph, oriented: bool=True):
    """

    From Networkx: The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row’s node is incident on the column’s edge.
    For an oriented incidence matrix each edge is assigned an orientation (arbitrarily for undirected and
    aligning to direction for directed). A -1 appears for the source (tail) of an edge and 1 for the
    destination (head) of the edge. The elements are zero otherwise.

    Here, if oriented=False, then we have the case with no signed incidence matrix"""

    if oriented is not False:
        I = csr_matrix(nx.incidence_matrix(graph, oriented=True))
        new_inc = I.todense()
    else:
        I = csr_matrix(nx.incidence_matrix(graph, oriented=False))
        new_inc = I.todense()

    return new_inc


def balance_forcing(f0, normalize=False):
    f = np.copy(f0)

    if np.allclose(f.sum(), 0):

        if normalize == True:
            # n_sinks = float(np.count_nonzero(f < 0))
            # n_sources = float(np.count_nonzero(f > 0))
            f[f > 0] /= f[f > 0].sum()
            f[f < 0] /= abs(f[f < 0]).sum()

            assert np.allclose(f[f > 0].sum(), 1.0), f"f[f > 0].sum() is {f[f > 0].sum()}"
            assert np.allclose(f[f < 0].sum(), -1)

        return f

    else:

        sink_mass = abs(f[f < 0]).sum()
        source_mass = f[f > 0].sum()
        if source_mass > sink_mass:
            total_mass = source_mass
            delta_sink_mass = total_mass - sink_mass
            n_sinks = float(np.count_nonzero(f < 0))
            f[f < 0] -= delta_sink_mass / n_sinks
        else:
            total_mass = sink_mass
            delta_source_mass = total_mass - source_mass
            n_sources = float(np.count_nonzero(f > 0))
            f[f > 0] += delta_source_mass / n_sources

        assert np.allclose(f.sum(), 0)

        if normalize == True:
            f[f > 0] /= f[f > 0].sum()
            f[f < 0] /= abs(f[f < 0]).sum()

            assert np.allclose(f[f > 0].sum(), 1.0)
            assert np.allclose(f[f < 0].sum(), -1.0)

        return f


def find_q(G: nx.Graph, input_list: list):
    """Finds source/sink terms
    input list: source/sink lists"""

    q = {}
    for i in G.edges():
        if i in input_list:
            q[i] = 1
        else:
            q[i] = 0

    return np.array(list(q.values()))