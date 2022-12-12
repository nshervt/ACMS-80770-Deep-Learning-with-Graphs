import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
import os
from typing import NamedTuple
import networkx 
import soln2cti
import networkx as nx

def dfs(G, source=None, depth_limit=None):

  # Depth first search algorithm to search the edges
  
    if source is None:
        # edges for all components
        nodes = G
    else:
        # edges for components with source
        nodes = [source]
    visited = set()  # creates list of visited species in the graph
    if depth_limit is None:
        depth_limit = len(G)  # total number of species
    for start in nodes:
        if start in visited:
            continue
        yield start, start, "forward"
        visited.add(start)
        stack = [(start, depth_limit, iter(G[start]))]  #iter functions shows the negighbours of the graph
        
        while stack:  ## Done till the depth is reached and then some nodes might have been visited which are saved. 
            parent, depth_now, children = stack[-1]
            try:
                child = next(children)   # Ch4-> H2-> H ...forward  ## If you dont get stack go to stack.pop (going behind)
                if child in visited:
                    yield parent, child, "nontree" ## labeling edges, if the edge is not visited but nodes have been
                else:
                    yield parent, child, "forward"
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, iter(G[child]))) ## Add neighbours of child in stack
            except StopIteration:
                stack.pop()
                if stack:
                    yield stack[-1][0], parent, "reverse" ## Decreases depth now
        yield start, start, "reverse"


def graph_search(graph, target_species):
    """Search nodal graph and generate list of species to remove

    Parameters
    ----------
    graph : networkx.DiGraph
        graph representing reaction system
    target_species : list
        List of target species to search from

    Returns
    -------
    reached_species : list of str
        List of species reachable from targets in graph

    """
    reached_species = []
    for target in target_species:
        edges = dfs(graph, target)
        v = (v for u, v, d in edges if d == "reverse")  #if we have reversed means we have seen the species
        reached_species += list(v)  # depth first search the nodes
    reached_species = list(set(reached_species))

    return reached_species


def trim_drg(matrix, species_names, species_targets, threshold):
    """

    Parameters
    ----------
    matrix : np.ndarray
        Adjacency matrix representing graph
    species_names : list of str
        List of all species names
    species_targets : list of str
        List of target species names
    threshold : float
        DRG threshold for trimming graph

    Returns
    ------
    species_reached : list of str
        Names of species reached in graph search

    """
    name_mapping = {i: sp for i, sp in enumerate(species_names)}
    
    # if the values of matrix is less than threshold than make it 0.0
    graph = networkx.DiGraph(np.where(matrix >= threshold, matrix, 0.0))  ## Removes edge connected to species from the graph
    networkx.relabel_nodes(graph, name_mapping, copy=False)

    ###### Print graphs
    
    # G = graph
    # color_map = ['red' if node == "CH4" else 'green' for node in G]
    # pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes()))*30, iterations=20)
    # nx.draw(G, pos=pos, node_color = color_map)
    # nx.draw_networkx_labels(G, pos=pos)
    # plt.show()

    species_reached = graph_search(graph, species_targets)

    return species_reached


def trim(initial_model_file, exclusion_list, new_model_file, phase_name=''):
    """Function to eliminate species and corresponding reactions from model

    Parameters
    ----------
    initial_model_file : str
        Filename for initial model to be reduced
    exclusion_list : list of str
        List of species names that will be removed
    new_model_file : str
        Name of new reduced model file
    phase_name : str, optional
        Optional name for phase to load from CTI file (e.g., 'gas'). 

    Returns
    -------
    new_solution : ct.Solution
        Model with species and associated reactions eliminated

    """
    solution = ct.Solution(initial_model_file, phase_name)

    # Remove species if in list to be removed
    final_species = [
        sp for sp in solution.species() if sp.name not in exclusion_list]
    final_species_names = [sp.name for sp in final_species]

    # Remove reactions that use eliminated species
    final_reactions = []
    for reaction in solution.reactions():
        # remove reactions with an explicit third body that has been removed
        if hasattr(reaction, 'efficiencies') and not getattr(reaction, 'default_efficiency', 1.0):
            if (len(reaction.efficiencies) == 1 and
                    list(reaction.efficiencies.keys())[0] in exclusion_list
                    ):
                continue

        reaction_species = list(reaction.products.keys()) + \
            list(reaction.reactants.keys())
        if all([sp in final_species_names for sp in reaction_species]):
            
            # remove any eliminated species from third-body efficiencies
            if hasattr(reaction, 'efficiencies'):
                reaction.efficiencies = {
                    sp: val for sp, val in reaction.efficiencies.items()
                    if sp in final_species_names
                }
            final_reactions.append(reaction)

    # Create new solution based on remaining species and reactions
    new_solution = ct.Solution(
        species=final_species, reactions=final_reactions,
        thermo='IdealGas', kinetics='GasKinetics'
    )
    new_solution.TP = solution.TP
    if phase_name:
        new_solution.name = phase_name
    else:
        new_solution.name = os.path.splitext(new_model_file)[0]

    return new_solution
#gas = ct.Solution('reduced_22.cti')


def create_matrix(model_file, state):

    solution = ct.Solution(model_file)
    solution.TPY = state[0], state[1], state[2:]

    # nu for all 53 species and there are 325 reactions
    net_stoich = solution.product_stoich_coeffs() - solution.reactant_stoich_coeffs()
    
    # check if the reaction has coresponding species
    flags = np.where(((solution.product_stoich_coeffs() != 0) |
                      (solution.reactant_stoich_coeffs() != 0)
                      ), 1, 0)

    # only consider contributions from reactions with nonzero net rates of progress
    valid_reactions = np.where(solution.net_rates_of_progress != 0)[0]
    
    #nu * omega
    if valid_reactions.size:
        base_rates = np.abs(
            net_stoich[:, valid_reactions] *
            solution.net_rates_of_progress[valid_reactions]
        )
        
        denominator = np.sum(base_rates, axis=1)[
            :, np.newaxis]  # sum over all reactions

        numerator = np.zeros((solution.n_species, solution.n_species))
        for sp_b in range(solution.n_species):
            numerator[:, sp_b] += np.sum(  # flags[sp_b, valid_reactions] gives species b for omega in reaction i so omega is counted only if 
                # species is present in that reaction.
                # flags give if species is present in that reaction
                # for each species B sp_b , we see its involvment in reactions
                base_rates[:, np.where(flags[sp_b, valid_reactions])[
                    0]], axis=1
                # species A is dimn 1
            )

        # May get divide by zero if an inert species is present, and denominator
        # entry is zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            adjacency_matrix = np.where(
                denominator != 0, numerator / denominator, 0)

    else:
        adjacency_matrix = np.zeros((solution.n_species, solution.n_species))

    # set diagonals to zero, to avoid self-directing graph edges
    (np.fill_diagonal(adjacency_matrix, 0.0))
    return adjacency_matrix


def calc_ign_delay(model_file, Temperature, EQ):

    # This is an auto-ignition simulation module by using a software "Cantera"
    
    gas = ct.Solution(model_file)
    gas.TP = Temperature, 1*ct.one_atm
    gas.set_equivalence_ratio(EQ, fuel='CH4:1', oxidizer='O2:1, N2:3.76')

    t_autoI = 0
    time = 0
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    states = ct.SolutionArray(gas, extra=['time_in_ms', 't'])
    
    # print('{:10s} {:10s} {:10s}'.format('t [s]', 'T [K]', 'P [Pa]'))

    for j in range(2900): # Hardcoded here. 
        # time += 1e-3
        sim.step()
        states.append(r.thermo.state, time_in_ms=sim.time*1e3, t=sim.time)
        # print('{:10.3e} {:10.3f} {:10.3f} '.format(sim.time, r.T, r.thermo.P))
    states1.append(states.T)

    for j in range(n):
        T_error = states.T[j] - Temperature - 400
        if T_error > 0:
           # T_autoI = states.T[j]
            t_autoI = states.time_in_ms[j]
            break
    t_autoI_T.append(t_autoI)

    delta = 0.05
    deltas = np.arange(delta, 1 + delta, delta)  # to create 20 samples per Temp and Equi_raio config
    sampled_data = np.zeros((len(deltas), 2 + states.Y.shape[1]))
    temperature_diff = states.T[-1] - states.T[0]
    idx = 0
    Q = 0
    ignition_flag = False

    for time, temp, pres, mass in zip(
        states.t, states.T, states.P, states.Y
    ):
        if temp >= Temperature + 400.0 and not ignition_flag:
            ignition_delay = time
            ignition_flag = True

        Q = Q+1
        # print(Q) 2460
        if temp >= Temperature + (deltas[idx] * temperature_diff):
            sampled_data[idx, 0:2] = [temp, pres]
            sampled_data[idx, 2:] = mass

            idx += 1
            if idx == 20:
                break

    # Passing 20 points sampled for T=1000 and phi = 1. We can do that for multiple condition to create a set
    return t_autoI, sampled_data

if __name__ == "__main__":

    model_file = 'gri30.cti'

    species_targets = ["CH4", "O2"] # We want the target species to be saved. 
    Temperature = [1000]     # in Kelvin
    Equivalance_Ratio = [1.0]
    time_start = 0.0
    time_end = 10.0
    time_step = 1e-3
    n = int((time_end - time_start)/time_step)
    t_autoI_T = []
    states1 = []
    sampled_data_gen = []
    ignition_delay_gen = []
    error_limit = 20.0  # Percent error acceptable for DRG (hyper parameter)
    error = 0.0  # Initializing the error
    threshold_increment = 0.01 # This is a hyper parameter
    threshold = 0.01
    matrices = []
    reduced_model_metrics = []
    error_list = []
    auto_I_list = []
    species_retained_list = []


    for idx, T in enumerate(Temperature):
        ignition_delay, sampled_data = calc_ign_delay(model_file, Temperature[idx], Equivalance_Ratio[idx])

        ignition_delay_gen.append(ignition_delay)

        if idx > 0:
            sampled_data = np.concatenate(
                (sampled_data_gen, sampled_data), axis=0)
        sampled_data_gen = sampled_data

    for state in sampled_data:
        # Based on equations create the matrix, 20 matrices for 20 data points with N_species x N_species shape
        matrices.append(create_matrix(model_file, state))

    while error < error_limit:

        reduced_model_metrics = []
        solution = ct.Solution(model_file, "")

        # Creating matrix

        species_retained = []

        for matrix in matrices:
            species_retained += trim_drg(matrix, solution.species_names,
                                        species_targets, threshold)  # Based of dfs and threshold we remove the species.

        # want to ensure retained species are the set of those reachable for each state
        species_retained = list(set(species_retained))  # Too many lists are merged into one using set, removing duplicates

        species_removed = [sp for sp in solution.species_names
                        if sp not in (species_retained)
                        ]

        # Cut the exclusion list from the model.
        reduced_model = trim(
            model_file, species_removed, f'reduced_drg_{model_file}', phase_name=""
        )

        reduced_model_file = soln2cti.write(
            reduced_model, f'reduced_drg_{reduced_model.n_species}.cti', path="."
        )

        for idx, T in enumerate(Temperature):
            ignition_delay_metrics, sampled_data_discard = calc_ign_delay(
                reduced_model_file, Temperature[idx], Equivalance_Ratio[idx])

            

            reduced_model_metrics.append(ignition_delay_metrics)

        error = np.max(
            abs((np.array(ignition_delay_gen)-np.array(reduced_model_metrics))/np.array(ignition_delay_gen)))*100

        threshold += threshold_increment
        error_list.append(error)
        species_retained_list.append(len(species_retained))

        print("threshold",threshold,"| number of species retained", len(species_retained),"| error",error,"| species removed",species_removed )

    name_mapping = {i: sp for i, sp in enumerate(solution.species_names)}
    
    # if the values of matrix is less than threshold than make it 0.0
    graph = networkx.DiGraph(matrix)  ## Removes edge connected to species from the graph
    networkx.relabel_nodes(graph, name_mapping, copy=False)

    ###### Print graphs
    color_map = []
    G = graph
    for node in G: 
   
        if node in species_removed:
            color_map.append('blue'); 
        else: 
            color_map.append('green')
            
    pos = nx.spring_layout(G, k=0.3*1/np.sqrt(len(G.nodes()))*30, iterations=20)
    nx.draw(G, pos=pos, node_color = color_map)
    nx.draw_networkx_labels(G, pos=pos)
    plt.show()  
