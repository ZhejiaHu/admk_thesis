import numpy as np
import networkx as nx
from scipy.spatial import distance

from mcopt import *
from .dynamics import *


class WassersteinGraph:
    """Optimal Transport for Multilayer Wasserstein"""

    def __init__(self, union: nx.Graph, rhs, pflux: float = 1.0, verbose: bool = True, weight: str = 'length'):
        # graph topology
        self.g = union  # graph, which is the union between the two layers
        self.length = np.zeros(self.g.number_of_edges())  # length of union graph edges

        for i, edge in enumerate(self.g.edges()):
            self.length[i] = distance.euclidean(
                self.g.nodes[edge[0]]["pos"], self.g.nodes[edge[1]]["pos"]
            )
        self.U = union.copy()
        self.tdens = np.zeros(self.g.number_of_edges())
        self.pflux = pflux  # beta

        self.relax_linsys = 1.0e-8  # relaxation for Laplacian
        self.seed = 0  # seed init conductivities
        self.time_tol = 1e4
        self.tol = 1e-3

        self.forcing = rhs

        self.time_step = 0.5  # time step dynamical system
        self.tot_time = 10000  # upper bound on number of time steps

        # misc
        self.verbose = verbose
        self.B = nx.linalg.graphmatrix.incidence_matrix(self.g, oriented=True)

        if weight == "unit":

            for edge in self.g.edges:
                self.g.edges[edge]["weight"] = 1

        elif weight == "length":
            for edge in self.g.edges:
                self.g.edges[edge]["weight"] = distance.euclidean(
                    self.g.nodes[edge[0]]["pos"], self.g.nodes[edge[1]]["pos"]
                )

    def dmk_solve(self):
        return dmk_solve(self)