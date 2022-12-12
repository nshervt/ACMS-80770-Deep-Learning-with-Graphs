import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cantera as ct
import os
from typing import NamedTuple
import networkx
import soln2cti
from heapq import heappush, heappop
from itertools import count
import networkx as nx


def dijkstra(G, source, get_weight, pred=None, paths=None,
                 cutoff=None, target=None
                 ):

    G_succ = G.succ if G.is_directed() else G.adj  # Adjaceny list
    push = heappush  # creates a binary tree where parent > child
    pop = heappop  # child > parent
    dist = {}  # dictionary of final distances
    seen = {source: 0}
    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (0, next(c), source))

    while fringe:
        cont = 0
        (d, _, v) = pop(fringe)  # take species with min R as v| v is next node
        if v in dist and d < dist[v]:
            continue  # already searched this node.
        if v == source:
            d = 1
        dist[v] = d
        if v == target:
            break

        # For all adjancent edges, get weights and multiply them by current path taken.
        for u, e in G_succ[v].items():  # u around v
            cost = get_weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] * get_weight(v, u, e)
            if cutoff is not None:
                if vu_dist < cutoff:
                    continue

            # If this path to u is greater than any other path we've seen, push it to the heap to be added to dist.
            # If already seen and the distance*edge is greater, we do not add to fringe
            elif u not in seen or vu_dist > seen[u]:
                # if 1-> 4 has some value vu_dist and 3-> 4 has a greater vu_dist then greater is updated
                seen[u] = vu_dist
                # push assembles fringe in increasing order of weights (lowest to highest)
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    if paths is not None:
        return (dist, paths)
    if pred is not None:
        return (pred, dist)
    return dist


def ss_dijkstra_path_length_modified(G, source, cutoff=None, weight='weight'):

    if G.is_multigraph():
        get_weight = lambda u, v, data: min(
            eattr.get(weight, 1) for eattr in data.values())
    else:
        # get_weight is a function which searches for weights and returns corresponding entery in dict
        get_weight = lambda u, v, data: data.get(weight, 1)

    return dijkstra(G, source, get_weight, cutoff=cutoff)


def graph_search_drgep(graph, target_species):

    overall_coefficients = {}
    for target in target_species:
        coefficients = ss_dijkstra_path_length_modified(graph, target)
        # ensure target has importance of 1.0
        coefficients[target] = 1.0

        for sp in coefficients:
            overall_coefficients[sp] = max(
                overall_coefficients.get(sp, 0.0), coefficients[sp])

    return overall_coefficients


def get_importance_coeffs(species_names, target_species, matrices):

    importance_coefficients = {sp: 0.0 for sp in species_names}
    name_mapping = {i: sp for i, sp in enumerate(species_names)}
    for matrix in matrices:
        graph = networkx.DiGraph(matrix)
        networkx.relabel_nodes(graph, name_mapping, copy=False)
        coefficients = graph_search_drgep(graph, target_species)

        importance_coefficients = {
            sp: max(coefficients.get(sp, 0.0), importance_coefficients[sp])
            for sp in importance_coefficients
        }

    return importance_coefficients


def create_matrix(model_file, state):

    solution = ct.Solution(model_file, '')
    temp, pressure, mass_fractions = [state[0], state[1], state[2:]]
    solution.TPY = temp, pressure, mass_fractions

    net_stoich = solution.product_stoich_coeffs() - solution.reactant_stoich_coeffs()
    flags = np.where(((solution.product_stoich_coeffs() != 0) |
                        (solution.reactant_stoich_coeffs() != 0)
                        ), 1, 0)

    # only consider contributions from reactions with nonzero net rates of progress
    valid_reactions = np.where(solution.net_rates_of_progress != 0)[0]
    if valid_reactions.size:
        base_rates = (
            net_stoich[:, valid_reactions] *
            solution.net_rates_of_progress[valid_reactions]
            )

        denominator_dest = np.sum(np.maximum(0.0, -base_rates), axis=1)
        denominator_prod = np.sum(np.maximum(0.0, base_rates), axis=1)
        denominator = np.maximum(denominator_prod, denominator_dest)[
                                 :, np.newaxis]

        numerator = np.zeros((solution.n_species, solution.n_species))
        for sp_b in range(solution.n_species):
            numerator[:, sp_b] += np.sum(
                base_rates[:, np.where(flags[sp_b, valid_reactions])[
                                       0]], axis=1
                )
        numerator = np.abs(numerator)

        # May get divide by zero if an inert species is present, and denominator
        # entry is zero.
        with np.errstate(divide='ignore', invalid='ignore'):
            adjacency_matrix = np.where(
                denominator != 0, numerator / denominator, 0)

    else:
        adjacency_matrix = np.zeros((solution.n_species, solution.n_species))

    # set diagonals to zero, to avoid self-directing graph edges
    np.fill_diagonal(adjacency_matrix, 0.0)

    return adjacency_matrix


def trim(initial_model_file, exclusion_list, new_model_file, phase_name=''):

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


def calc_ign_delay(model_file, Temperature):

    gas = ct.Solution(model_file, '')
    gas.TP = Temperature, 1*ct.one_atm
    gas.set_equivalence_ratio(1.0, fuel='CH4:1', oxidizer='O2:1, N2:3.76')

    t_autoI = 0
    time = 0
    r = ct.IdealGasReactor(gas)
    sim = ct.ReactorNet([r])
    states = ct.SolutionArray(gas, extra=['time_in_ms', 't'])
   # print('{:10s} {:10s} {:10s}'.format('t [s]', 'T [K]', 'P [Pa]'))

    for j in range(2908):  # Need to add some sort of residual here for solution convergence
        # time += 1e-3
        sim.step()
        states.append(r.thermo.state, time_in_ms=sim.time*1e3, t=sim.time)
      #  print('{:10.3e} {:10.3f} {:10.3f} '.format(sim.time, r.T, r.thermo.P))
    states1.append(states.T)

    for j in range(n):
        T_error = states.T[j] - Temperature - 400
        if T_error > 0:
           # T_autoI = states.T[j]
            t_autoI = states.time_in_ms[j]
            break
    t_autoI_T.append(t_autoI)

    delta = 0.05
    deltas = np.arange(delta, 1 + delta, delta)
    sampled_data = np.zeros((len(deltas), 2 + states.Y.shape[1]))
    temperature_diff = states.T[-1] - states.T[0]
    idx = 0
    Q = 0
    ignition_flag = False
    for time, temp, pres, mass in zip(
        states.t, states.T, states.P, states.Y
    ):
        if temp >= 1000 + 400.0 and not ignition_flag:
            ignition_delay = time
            ignition_flag = True

        Q = Q+1

        if temp >= 1000 + (deltas[idx] * temperature_diff):
            sampled_data[idx, 0:2] = [temp, pres]
            sampled_data[idx, 2:] = mass

            idx += 1
            if idx == 20:
                break

    # Passing 20 points sampled for T=1000 and phi = 1. We can do that for multiple condition to create a set
    return t_autoI, sampled_data


#################################################################
model_file = 'gri30.cti'

species_targets = ["CH4", "O2"]
species_safe = ["N2"]
Temperature = 1000.0  # np.linspace(950,1450,11)
time_start = 0.0
time_end = 10.0
time_step = 1e-3
n = int((time_end - time_start)/time_step)
t_autoI_T = []
states1 = []


error_limit = 20.0  # Percent error acceptable for DRG
error = 0.0  # Initializing the error
threshold_increment = 0.01  # Nedd to see why it is incremented
threshold = 0.01


ignition_delay, sampled_data = calc_ign_delay(model_file, Temperature)
matrices = []

for state in sampled_data:
		# Based on equations create the matrix, 20 matrices for 20 data points with NsxNs shape
		matrices.append(create_matrix(model_file, state))

solution = ct.Solution(model_file)
importance_coeffs = get_importance_coeffs(
        # taking max coeff of all samples
        solution.species_names, species_targets, matrices
        )

while error < error_limit:
	solution = ct.Solution(model_file, "")

	# Creating matrix

    # shouldnt we create new matrix after each iterations? look into paper for algorithm

	species_retained = []

	species_removed = [sp for sp in solution.species_names
                       if importance_coeffs[sp] < threshold
                       and sp not in species_safe]

	reduced_model = trim(
        model_file, species_removed, f'reduced_drgep_{model_file}', phase_name='')

	reduced_model_file = soln2cti.write(
		reduced_model, f'reduced_drgep_{reduced_model.n_species}.cti', path=".")

	reduced_model_metrics, sampled_data = calc_ign_delay(
	    reduced_model_file, Temperature)
	# Creating matrix

	error = abs((ignition_delay-reduced_model_metrics)/ignition_delay)*100

	threshold += threshold_increment;print("threshold",threshold,"| error",error,"| species removed",species_removed )

name_mapping = {i: sp for i, sp in enumerate(solution.species_names)}
    
# if the values of matrix is less than threshold than make it 0.0
graph = networkx.DiGraph(matrices[0])  ## Removes edge connected to species from the graph
networkx.relabel_nodes(graph, name_mapping, copy=False)

###### Print graphs #############

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

# for node in species_removed: G.remove_node(node)