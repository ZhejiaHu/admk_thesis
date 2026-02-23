# import Solver
from copy import deepcopy
import sys
import os

import networkx as nx
import time

# Import Admk solver for graphs
sys.path.append('../src/')
from admk import Graph, MinNorm, AdmkControls, AdmkSolver, AdmkSolution
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


def create_forcing(n_nodes, num_commodity: int=2) -> List[np.ndarray]:
    forcing_1 = np.ones(n_nodes)  / (n_nodes - 1)
    forcing_1[0] = -sum(forcing_1[1:])
    forcing_2 = np.ones(n_nodes)
    forcing_2[-1] = -sum(forcing_2[:-1])
    if num_commodity == 1: return [forcing_1]
    else: return [forcing_1, forcing_2]


def set_control(beta:float=1, log_print=True, max_iter=1000):
    import inspect
    #print("init signature:", inspect.signature(AdmkControls.__init__))
    ctrl = AdmkControls(tol_optimization=1e-4, tol_constraint=1e-5, method='explicit_tdens', max_iter=max_iter,
                        max_restart=2, verbose=2, log=0, log_file='admk.log', beta=beta, log_print=log_print)

    # deltat controls
    ctrl.set_method_ctrl('deltat', {'control': 'adaptive2', 'initial': 1e-3, 'min': 1e-10, 'max': 1e-1, 'expansion': 1.2, 'contraction': 2})
    # linear solver
    # matrix is singualr. we need to relax it with + relax*identity
    ctrl.set_method_ctrl('relax_Laplacian', 1e-10)
    ctrl.set_method_ctrl(['ksp', 'type'], 'cg')
    ctrl.set_method_ctrl(['pc', 'type'], 'icc')
    ctrl.set_method_ctrl(['pc', 'factor_drop_tolerance', 'dt'], 1e-4)
    ctrl.set_method_ctrl(['pc', 'type'], 'hypre')
    return ctrl


def plot_result(graph_topo: np.ndarray, weights: np.ndarray, num_commodity: int, potentials: List[np.ndarray], conductivity: np.ndarray):
    assert len(potentials) == num_commodity and len(conductivity) == graph_topo.shape[0] and graph_topo.shape[0] == weights.shape[0]
    graph = nx.Graph()
    graph.add_edges_from(map(lambda i: (graph_topo[i][0], graph_topo[i][1], {"weight": weights[i], "conductivity": conductivity[i]}), range(len(graph_topo))))
    #print(graph.edges)
    #print(graph.nodes)
    fig, ax = plt.subplots(num_commodity, 1, figsize=(8, 8))
    pos = nx.spring_layout(graph)
    edge_labels, conductivity = nx.get_edge_attributes(graph, "weight"), nx.get_edge_attributes(graph, "conductivity")
    #print(conductivity)
    #print(f"Conductivity is {conductivity}")
    for i, potential in enumerate(potentials):
        nx.draw_networkx_edges(graph, pos, width=[conductivity.get(edge, 1.0) * 0.3 for edge in list(graph.edges())], edge_color="C0", style="solid", ax=ax[i] if num_commodity > 1 else ax)
        nx.draw_networkx_nodes(graph, pos=pos, node_shape='o', node_color='gray', node_size=np.abs(potential) * 50, linewidths=0.1, ax=ax[i] if num_commodity > 1 else ax)
        # nx.draw_networkx_labels(graph, pos=pos, labels={n: n for n in pos}, ax=ax[i] if num_commodity > 1 else ax)
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax[i] if num_commodity > 1 else ax)



def test_main(topol: np.ndarray, weight: np.ndarray, num_commodity: int, forcing_: List[np.ndarray], beta: float=1, max_iter: int=2000) -> Tuple[List[np.ndarray], np.ndarray]:

    # Init. graph problem, incidence matrix and its transpose
    graph = Graph(topol.transpose())
    incidence_matrix = graph.signed_incidence_matrix()
    incidence_matrix_transpose = incidence_matrix.transpose()
    forcing = np.concatenate(forcing_)
    
    problem = MinNorm(incidence_matrix_transpose, rhs_of_time=forcing, q_exponent=1.0, weight=weight)
    admk = AdmkSolver(problem, set_control(beta=beta, log_print=False, max_iter=max_iter))
    admk.ctrl.set_method_ctrl(['pc','type'],'hypre')
    sol0 = deepcopy(admk.solution) # first option
    sol0 = AdmkSolution(problem) # second option
    pot0, tdens0 = sol0.subfunctions()
    tdens0[:]=2.0
    admk.set_initial_guess(sol0)
    start = time.process_time()
    ierr, history_losses = admk.solve()
    running_time_admk = time.process_time() - start
    print(f"Elapsed (wall clock): {running_time_admk:.6f} s")
    def _plot(history_losses: List[float]):
        plt.plot(history_losses)
        plt.xlabel("Iteration")
        plt.ylabel("Losses (Energy)")
        plt.title(f"Losses During Training with Beta {beta}")

    _plot(history_losses)
    # print('ierr=',ierr,admk.ierr_dictionary(ierr))
    pot, tdens, vel = admk.solution.get_problem_solution()
    final_energy = admk._evaluate_loss(tdens)
    pots = [admk.solution.get_subpotential(i) for i in range(num_commodity)]
    # print(f"pots = {pots}")
    # print('tdens=',tdens)
    # print('vel=',vel)

    # check if convergence is achieved
    return pots, tdens, final_energy, running_time_admk


if __name__ == "__main__":
    topo = np.array([[0, 1], [0, 2], [0, 3], [2, 3], [1, 2], [3, 4], [0, 4]],)
    weights = np.array([1, 11, 10, 4, 5, 6, 1])
    potentials, conductivity = test_main(topo.copy(), weights, 2, create_forcing(5, num_commodity=2), beta=1)
    plot_result(topo, weights,  2, potentials, conductivity)
    potentials, conductivity = test_main(topo.copy(), weights, 2, create_forcing(5, num_commodity=2), beta=2)
    plot_result(topo, weights, 2, potentials, conductivity)
    plt.show()

