
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

rng = np.random.default_rng(38)

N = 200
T = 200
p_edge = 0.05
infect_0 = 5

def simulate(beta, gamma, rho):

    # Initializing the graph
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Initializing the infection state
    state = np.zeros(N, dtype = np.int8)
    initial_infected = rng.choice(N, size = infect_0, replace = False)
    state[initial_infected] = 1

    # Initializing output arrays
    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype = np.int64)
    infected_fraction[0] = np.count_nonzero(state == 1) / N

    all_nodes = set(range(N))

    for t in range(1, T + 1):
        
        new_infections = set()
        new_recoveries = set()
        infected_nodes = np.where(state == 1)[0]

        rewire_count = 0

        si_edges = []

        # End early when infection rate hits 0
        infected_nodes = np.where(state == 1)[0]

        if len(infected_nodes) == 0:
            degree_histogram = np.zeros(31, dtype=np.int64)
            for i in range(N):
                deg = min(len(neighbors[i]), 30)
                degree_histogram[deg] += 1
            return infected_fraction, rewire_counts, degree_histogram
        
        # Infect and Recover simultaneously
        for i in infected_nodes:
            for j in neighbors[i]:
                if state[j] == 0: 
                    if rng.random() < beta:
                        new_infections.add(j)
                    si_edges.append((j, i))
            if rng.random() < gamma:
                new_recoveries.add(i)
        
        # Rewire using initial state
        for s_node, i_node in si_edges:
            if rng.random() < rho:

                if i_node not in neighbors[s_node]:
                    continue

                neighbors[s_node].discard(i_node)
                neighbors[i_node].discard(s_node)

                candidates = []
                candidates = list(all_nodes - neighbors[s_node] - {s_node})
                if candidates:
                    new_partner = rng.choice(candidates)
                    neighbors[s_node].add(new_partner)
                    neighbors[new_partner].add(s_node)
                    rewire_count += 1

        # Update state after infection and recovery
        for i in new_infections:
            state[i] = 1
        
        for j in new_recoveries:
            state[j] = 2
        
        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count

    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = min(len(neighbors[i]), 30)
        degree_histogram[deg] += 1
    
    return infected_fraction, rewire_counts, degree_histogram    


def create_tables(N_sim):
    beta_prior = rng.uniform(0.05, 0.5, N_sim)
    gamma_prior = rng.uniform(0.02, 0.2, N_sim)
    rho_prior = rng.uniform(0, 0.8, N_sim)
    param_values = np.stack([beta_prior, gamma_prior, rho_prior], axis = 1)
    infected_sim = np.zeros((N_sim, 40, 201))
    rewire_sim = np.zeros((N_sim, 40, 201))
    degree_sim = np.zeros((N_sim, 40, 31))
    for idx, params in enumerate(tqdm(param_values, desc="Simulating")):
        for rep in range(40):
            inf, rew, deg = simulate(params[0], params[1], params[2])
            infected_sim[idx, rep] = inf
            rewire_sim[idx, rep] = rew
            degree_sim[idx, rep] = deg
    return param_values, infected_sim, rewire_sim, degree_sim




# p, i, r, d = create_tables(10000)

# np.savez('simulation_results_2.npz',
#          param_values = p,
#          infected_sim = i,
#          rewire_sim = r,
#          degree_sim = d)