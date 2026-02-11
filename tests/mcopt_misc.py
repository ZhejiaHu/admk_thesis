import numpy as np

def plot_network(graph, forcing, tdens_mcopt_, tdens_admk_, mask):
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pickle as pkl
    import os

    path = os.getcwd()
    input_path = path + "/../data/input/"
    with open(input_path + "comm_list.pkl", 'rb') as comm_list_f:
        comm_list = pkl.load(comm_list_f)

    tdens_mcopt, tdens_admk = tdens_mcopt_ / np.linalg.norm(tdens_mcopt_), tdens_admk_ / np.linalg.norm(tdens_admk_)
    inflows = np.diag(forcing)
    inflows = inflows / np.sum(inflows)
    print(f"Inflows: {inflows}")

    pos = {n[0]: n[1]["pos"] for n in graph.nodes(data=True)}

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    print(f"tdens_admk_: {tdens_admk_}")
    node_prop = 100
    
    for idx, tdens in enumerate([tdens_mcopt, tdens_admk]):
        nx.draw_networkx_edges(graph,
                               pos=pos,
                               width=tdens * 15,
                               edge_color='C0',
                               style='solid',
                               ax=ax[idx])
        nx.draw_networkx_nodes(graph,
                               nodelist=comm_list,
                               pos=pos,
                               node_shape='o',
                               node_color='orange',
                               node_size=inflows * mask * node_prop,
                               linewidths=0.1,
                               ax=ax[idx])

        ax[idx].set_title(f"Result of  {'Mcopt' if idx == 0 else 'Admk'}", size=14)

    plt.show()


def run_mcopt(mcopt):
    mcopt.dyn_exec()
    flux_dyn, flux_opt, graph, length, forcing = mcopt.export_flux()
    cost_dyn, cost_opt = mcopt.export_cost()
    tdens =  mcopt.export_tdens()
    return forcing, tdens
