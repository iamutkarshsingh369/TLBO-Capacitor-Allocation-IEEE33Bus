"""
=============================================================================
 Capacitor Allocation and Sizing in Distribution Systems
 Using Particle Swarm Optimization (PSO)
 Reference: IEEE 33-Bus Radial Distribution System
 Method: Kennedy & Eberhart (1995) - PSO Algorithm
=============================================================================
 HOW TO RUN:
   1. Make sure common.py is in the same folder
   2. Run: py -3.12 pso_capacitor.py
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

# ── PSO PARAMETERS ───────────────────────────────────────────────────────────
n_particles = 30
n_iter = 100
n_cap = 3
dim = 2 * n_cap

w_max, w_min = 0.9, 0.4     # inertia weight (linearly decreasing)
c1, c2 = 1.5, 1.5           # cognitive & social coefficients

lb = np.zeros(dim)
ub = np.zeros(dim)
for i in range(n_cap):
    lb[2 * i] = 1
    lb[2 * i + 1] = 0
    ub[2 * i] = n_bus - 1
    ub[2 * i + 1] = len(cap_sizes) - 1

rng = np.random.default_rng(42)

# Initialize swarm
pos = rng.random((n_particles, dim)) * (ub - lb) + lb
vel = np.zeros((n_particles, dim))
v_max = 0.2 * (ub - lb)

fit = np.array([evaluate_fitness(pos[i, :], cap_sizes, from_bus, to_bus, R_pu, X_pu,
                                  P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)
                for i in range(n_particles)])

pbest_pos = pos.copy()
pbest_fit = fit.copy()
gbest_idx = np.argmin(fit)
gbest_pos = pos[gbest_idx, :].copy()
gbest_fit = fit[gbest_idx]

best_hist = np.zeros(n_iter)
avg_hist = np.zeros(n_iter)

print("\n=============================================================")
print(f"  PSO: {n_particles} particles | {n_iter} iterations | {n_cap} capacitors")
print("=============================================================")

for it in range(n_iter):
    w = w_max - (w_max - w_min) * it / n_iter   # linearly decreasing inertia

    for i in range(n_particles):
        r1 = rng.random(dim)
        r2 = rng.random(dim)
        vel[i, :] = (w * vel[i, :]
                     + c1 * r1 * (pbest_pos[i, :] - pos[i, :])
                     + c2 * r2 * (gbest_pos - pos[i, :]))
        vel[i, :] = np.clip(vel[i, :], -v_max, v_max)

        pos[i, :] = pos[i, :] + vel[i, :]
        pos[i, :] = np.clip(pos[i, :], lb, ub)

        f = evaluate_fitness(pos[i, :], cap_sizes, from_bus, to_bus, R_pu, X_pu,
                              P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)
        fit[i] = f

        if f < pbest_fit[i]:
            pbest_fit[i] = f
            pbest_pos[i, :] = pos[i, :].copy()

    cur_idx = np.argmin(pbest_fit)
    if pbest_fit[cur_idx] < gbest_fit:
        gbest_fit = pbest_fit[cur_idx]
        gbest_pos = pbest_pos[cur_idx, :].copy()

    best_hist[it] = gbest_fit
    avg_hist[it] = np.mean(fit)

    if (it + 1) % 20 == 0:
        print(f"  Iter {it + 1:4d}/{n_iter} | Best Fitness = {gbest_fit:.4f}")

print(f"\n  Done! Best Fitness = {gbest_fit:.4f}")

# ── DECODE BEST SOLUTION ─────────────────────────────────────────────────────
Qcap_opt, cap_bus, cap_kvar = decode_solution(gbest_pos, cap_sizes, n_bus, n_cap)
V_opt, Ploss_opt = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                                  Qcap_opt, n_bus, n_branch)
Ploss_opt_kW = Ploss_opt * Sbase
reduction = 100 * (Ploss_base_kW - Ploss_opt_kW) / Ploss_base_kW

print("\n=============================================================")
print("  OPTIMAL CAPACITOR PLACEMENT RESULTS (PSO)")
print("=============================================================")
print(f"  {'Bus':<6}  {'Size (kVAR)':<14}")
for i in range(n_cap):
    print(f"  {cap_bus[i]:<6d}  {cap_kvar[i]:<14.0f}")
print(f"\n  Total Capacitor : {np.sum(cap_kvar):.0f} kVAR")
print(f"\n  Base Case Power Loss  : {Ploss_base_kW:.4f} kW")
print(f"  PSO Optimized Loss    : {Ploss_opt_kW:.4f} kW")
print(f"  Loss Reduction        : {reduction:.2f}%")
print(f"  Optimized Min Voltage : {np.min(V_opt):.4f} p.u.")
print("=============================================================")

# Save results for comparison script
np.savez('pso_results.npz', best_hist=best_hist, avg_hist=avg_hist,
          Ploss_opt_kW=Ploss_opt_kW, V_opt=V_opt, cap_bus=cap_bus, cap_kvar=cap_kvar,
          reduction=reduction)
print("\n  Results saved to 'pso_results.npz' (for comparison script)\n")
