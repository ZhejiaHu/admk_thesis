"""
McOpt (Multicommodity Optimal Transport) -- https://github.com/aleable/McOpt

Contributors:
    Alessandro Lonardi
    Enrico Facca
    Caterina De Bacco
"""

import os
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import diags

from .initialization import *
from .dynamics import *
from .optimization import *
from typing import Dict, Tuple, List


path = os.getcwd()
input_path = os.getcwd() + "/../data/input/"


class McOpt:
    """Multicommodity Optimal Transport"""

    def __init__(self, method, coupling, museed, pflux, verbose, rho, relax_linsys, tau_cond_dyn, tau_cost_dyn,
                 time_step, tot_time, tau_cond_opt, tau_cost_opt, waxman_config: Dict[str, float]=None):

        # graph topology
        self.g = nx.Graph()                                     # graph
        self.length = np.zeros(self.g.number_of_edges())        # length of edges

        # dynamical system parameters
        self.method = method                                    # paris metro / synthetic network
        self.coupling = coupling                                # function to couple commodities, f(F_e)
        self.rho = rho                                          # aggregation coefficient for the mass
        self.pflux = pflux                                      # beta
        self.relax_linsys = relax_linsys                        # relaxation for Laplacian
        self.seed = museed                                      # seed init conductivities
        self.tdens = np.zeros(self.g.number_of_edges())         # conductivities
        self.forcing = np.zeros((self.g.number_of_edges(), self.g.number_of_edges()))   # right hand side

        self.time_step = time_step                              # time step dynamical system
        self.tot_time = tot_time                                # upper bound on number of time steps

        # convergence paramenters
        self.tau_cond_dyn = tau_cond_dyn                        # threshold convergence conductivities dynamics
        self.tau_cost_dyn = tau_cost_dyn                        # threshold convergence cost dynamics
        self.tau_cond_opt = tau_cond_opt                        # threshold convergence conductivities fixed-point
        self.tau_cost_opt = tau_cost_opt                        # threshold convergence cost fixed-point

        # misc
        self.verbose = verbose

        # variable at convergence
        self.opttends_dyn = np.zeros(self.g.number_of_edges())
        self.optpot_dyn = np.zeros((self.g.number_of_edges(), self.g.number_of_edges()))
        self.optflux_opt = np.zeros((self.g.number_of_edges(), self.g.number_of_edges()))
        self.cost_stack_dyn = np.zeros(self.g.number_of_edges())
        self.cost_stack_opt = np.zeros(self.g.number_of_edges())
        
        # Waxman Configuration 
        self.waxman_config = waxman_config
        self.is_grid = False



    def ot_setup(self):
        """Building graph topology"""

        print("* graph topology construction")

        if self.method == "synth":
            if self.waxman_config is not None:
                self.g, self.length, self.forcing = waxman_topology(self, self.waxman_config["n"], self.waxman_config["alpha"], self.waxman_config["beta"], self.waxman_config["L"])
            else:
                self.g, self.length, self.forcing = waxman_topology(self)
        if self.method == "paris":
            self.g, self.length, self.forcing = paris_topology(self, input_path)
            print(f"Graph: {self.g}")
            print(f"Forcing: {self.forcing}")


    def ot_reset(self, graph: nx.Graph, length, forcing):
        self.g, self.length, self.forcing = graph, length, forcing

        self.tdens = np.zeros(self.g.number_of_edges())
        self.opttends_dyn = np.zeros(self.g.number_of_edges())
        self.optpot_dyn = np.zeros((self.g.number_of_edges(), self.g.number_of_edges()))
        self.optflux_opt = np.zeros((self.g.number_of_edges(), self.g.number_of_edges()))
        self.cost_stack_dyn = np.zeros(self.g.number_of_edges())
        self.cost_stack_opt = np.zeros(self.g.number_of_edges())
        self.is_grid = True

    def ot_forcing_mask(self, mask: np.ndarray):
        num_comm = len(self.forcing)
        for comm in range(num_comm):
            if mask[comm] == 0:
                self.forcing[comm, :] = 0


    def export_setup(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        edge_lists = np.array([list(e) for e in self.g.edges()])
        return edge_lists, np.array(self.length), [self.forcing[:, i] for i in range(self.forcing.shape[1])]


    def dyn_exec(self):
        """Run dynamical system discretization"""

        self.tdens = tdensinit(self)                # conductivities initialization
        self.opttends_dyn, self.optpot_dyn, self.cost_stack_dyn = dyn(self)     # run dynamics

    def opt_exec(self):
        """Run fixed-point iteration routine"""

        self.tdens = tdensinit(self)   # conductivities initialization
        self.optflux_opt, self.cost_stack_opt = opt(self)   # run fixed point iteration

    def export_flux(self):
        """Export fluxes at convergence

        Returns:
            self.optflux_dyn: np.array, fluxes at convergence dynamics
            self.optflux_opt: np.array, fluxes at convergence fixed-point
            self.g: nx.Graph, graph topology
            self.length: np.array, edges lengths
            self.forcing: np. array, forcing matrix
            """

        print("* export fluxes")

        td_mat = diags(self.opttends_dyn, 0)
        inc_mat = csr_matrix(nx.incidence_matrix(self.g, nodelist=list(range(self.g.number_of_nodes())), oriented=True))
        inc_transpose = csr_matrix(inc_mat.transpose())
        inv_len_mat = diags(1 / self.length, 0)
        optflux_dyn = td_mat * inv_len_mat * inc_transpose * self.optpot_dyn

        return optflux_dyn, self.optflux_opt, self.g, self.length, self.forcing

    def export_cost(self):
        """Export cost array at different iterations

      Returns:
            self.cost_stack_dyn: np.array, cost vs iterations dynamics
            self.cost_stack_opt: np.array, cost vs iterations fixed-point
            """

        print("* export cost")

        return self.cost_stack_dyn, self.cost_stack_opt


    def export_tdens(self):
        """Export conductivities"""
        return self.tdens
