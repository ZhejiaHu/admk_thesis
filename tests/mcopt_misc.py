def plot_network(graph, forcing, tdens):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pickle as pkl
    import os

    path = os.getcwd()
    input_path = path + "/../data/input/"
    with open(input_path + "comm_list.pkl", 'rb') as comm_list_f:
        comm_list = pkl.load(comm_list_f)


    inflows = np.diag(forcing)
    inflows = inflows / np.sum(inflows)

    pos = {n[0]: n[1]["pos"] for n in graph.nodes(data=True)}

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    width_prop = 5 * 1e2
    node_prop = 20 * 1e2

    nx.draw_networkx_edges(graph,
                           pos=pos,
                           width=width_prop * tdens,
                           edge_color='C0',
                           style='solid',
                           ax=ax[0])
    nx.draw_networkx_nodes(graph,
                           nodelist=comm_list,
                           pos=pos,
                           node_shape='o',
                           node_color='gray',
                           node_size=inflows * node_prop,
                           linewidths=0.1,
                           ax=ax[0])

    ax[0].set_title("Dynamics", size=14)
    ax[1].set_title("Optimization (Fixed-point)", size=14)

    plt.show()


def run_mcopt(mcopt):
    mcopt.dyn_exec()
    flux_dyn, flux_opt, graph, length, forcing = mcopt.export_flux()
    cost_dyn, cost_opt = mcopt.export_cost()
    tdens =  mcopt.export_tdens()
    return forcing, tdens
