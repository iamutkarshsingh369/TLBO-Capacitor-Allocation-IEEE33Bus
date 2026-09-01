"""
=============================================================================
 Capacitor Allocation and Sizing in Distribution Systems
 Using Genetic Algorithm (GA)
 Reference: IEEE 33-Bus Radial Distribution System
 Method: Holland (1975) - GA with tournament selection, crossover, mutation
=============================================================================
 HOW TO RUN:
   1. Make sure common.py is in the same folder
   2. Run: py -3.12 ga_capacitor.py
=============================================================================
"""

import numpy as np
from common import (bfs_load_flow, decode_solution, evaluate_fitness,
                     from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                     cap_sizes, n_bus, n_branch, Sbase, Vmin, Vmax,
                     P_total, Q_total)

print("\n=============================================================")
print("  IEEE 33-Bus System Loaded")
print(f"  Total Load : {P_total:.0f} kW  +  {Q_total:.0f} kVAR")
print("=============================================================")

# ── BASE CASE ────────────────────────────────────────────────────────────────
Qcap_base = np.zeros(n_bus + 1)
V_base, Ploss_base = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                                    Qcap_base, n_bus, n_branch)
Ploss_base_kW = Ploss_base * Sbase
print(f"\n  Base Case Power Loss : {Ploss_base_kW:.4f} kW")
print(f"  Base Case Min Voltage: {np.min(V_base):.4f} p.u.")

# ── GA PARAMETERS ─────────────────────────────────────────────────────────────
n_pop = 30
n_iter = 100
n_cap = 3
dim = 2 * n_cap

pc = 0.8            # crossover probability
pm = 0.1             # mutation probability
tournament_k = 3     # tournament selection size
elite_count = 2       # number of best individuals carried over unchanged

lb = np.zeros(dim)
ub = np.zeros(dim)
for i in range(n_cap):
    lb[2 * i] = 1
    lb[2 * i + 1] = 0
    ub[2 * i] = n_bus - 1
    ub[2 * i + 1] = len(cap_sizes) - 1

rng = np.random.default_rng(42)

pop = rng.random((n_pop, dim)) * (ub - lb) + lb
fit = np.array([evaluate_fitness(pop[i, :], cap_sizes, from_bus, to_bus, R_pu, X_pu,
                                  P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)
                for i in range(n_pop)])

best_idx = np.argmin(fit)
best_f = fit[best_idx]
best_X = pop[best_idx, :].copy()

best_hist = np.zeros(n_iter)
avg_hist = np.zeros(n_iter)


def tournament_select(pop, fit, k, rng):
    """Pick k random individuals, return the fittest one."""
    idx = rng.integers(0, len(pop), size=k)
    best = idx[np.argmin(fit[idx])]
    return pop[best, :].copy()


print("\n=============================================================")
print(f"  GA: {n_pop} population | {n_iter} generations | {n_cap} capacitors")
print("=============================================================")

for it in range(n_iter):
    # ── ELITISM: carry the best individuals forward unchanged ─────────────
    elite_idx = np.argsort(fit)[:elite_count]
    new_pop = [pop[i, :].copy() for i in elite_idx]

    # ── GENERATE REST OF NEW POPULATION ────────────────────────────────────
    while len(new_pop) < n_pop:
        parent1 = tournament_select(pop, fit, tournament_k, rng)
        parent2 = tournament_select(pop, fit, tournament_k, rng)

        # Crossover (arithmetic/blend crossover)
        if rng.random() < pc:
            alpha = rng.random(dim)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()

        # Mutation (random reset within bounds)
        for child in (child1, child2):
            mut_mask = rng.random(dim) < pm
            rand_vals = rng.random(dim) * (ub - lb) + lb
            child[mut_mask] = rand_vals[mut_mask]
            child[:] = np.clip(child, lb, ub)

        new_pop.append(child1)
        if len(new_pop) < n_pop:
            new_pop.append(child2)

    pop = np.array(new_pop[:n_pop])
    fit = np.array([evaluate_fitness(pop[i, :], cap_sizes, from_bus, to_bus, R_pu, X_pu,
                                      P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)
                    for i in range(n_pop)])

    cur_idx = np.argmin(fit)
    if fit[cur_idx] < best_f:
        best_f = fit[cur_idx]
        best_X = pop[cur_idx, :].copy()

    best_hist[it] = best_f
    avg_hist[it] = np.mean(fit)

    if (it + 1) % 20 == 0:
        print(f"  Gen {it + 1:4d}/{n_iter} | Best Fitness = {best_f:.4f}")

print(f"\n  Done! Best Fitness = {best_f:.4f}")

# ── DECODE BEST SOLUTION ─────────────────────────────────────────────────────
Qcap_opt, cap_bus, cap_kvar = decode_solution(best_X, cap_sizes, n_bus, n_cap)
V_opt, Ploss_opt = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                                  Qcap_opt, n_bus, n_branch)
Ploss_opt_kW = Ploss_opt * Sbase
reduction = 100 * (Ploss_base_kW - Ploss_opt_kW) / Ploss_base_kW

print("\n=============================================================")
print("  OPTIMAL CAPACITOR PLACEMENT RESULTS (GA)")
print("=============================================================")
print(f"  {'Bus':<6}  {'Size (kVAR)':<14}")
for i in range(n_cap):
    print(f"  {cap_bus[i]:<6d}  {cap_kvar[i]:<14.0f}")
print(f"\n  Total Capacitor : {np.sum(cap_kvar):.0f} kVAR")
print(f"\n  Base Case Power Loss  : {Ploss_base_kW:.4f} kW")
print(f"  GA Optimized Loss     : {Ploss_opt_kW:.4f} kW")
print(f"  Loss Reduction        : {reduction:.2f}%")
print(f"  Optimized Min Voltage : {np.min(V_opt):.4f} p.u.")
print("=============================================================")

np.savez('ga_results.npz', best_hist=best_hist, avg_hist=avg_hist,
          Ploss_opt_kW=Ploss_opt_kW, V_opt=V_opt, cap_bus=cap_bus, cap_kvar=cap_kvar,
          reduction=reduction)
print("\n  Results saved to 'ga_results.npz' (for comparison script)\n")
