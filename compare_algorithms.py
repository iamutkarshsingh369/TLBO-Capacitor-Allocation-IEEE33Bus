"""
=============================================================================
 Capacitor Allocation - Algorithm Comparison
 TLBO vs PSO vs GA on IEEE 33-Bus Distribution System
 Same fitness function, same load flow, same bounds -> fair comparison
=============================================================================
 HOW TO RUN:
   1. Make sure common.py is in the same folder
   2. Run: py -3.12 compare_algorithms.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
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

# ── SHARED SETTINGS (identical for all 3 algorithms) ────────────────────────
n_pop = 30
n_iter = 100
n_cap = 3
dim = 2 * n_cap

lb = np.zeros(dim)
ub = np.zeros(dim)
for i in range(n_cap):
    lb[2 * i] = 1
    lb[2 * i + 1] = 0
    ub[2 * i] = n_bus - 1
    ub[2 * i + 1] = len(cap_sizes) - 1


def fit_fn(x):
    return evaluate_fitness(x, cap_sizes, from_bus, to_bus, R_pu, X_pu,
                             P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)


# =============================================================================
# TLBO
# =============================================================================
def run_tlbo(seed=42):
    rng = np.random.default_rng(seed)
    pop = rng.random((n_pop, dim)) * (ub - lb) + lb
    fit = np.array([fit_fn(pop[i, :]) for i in range(n_pop)])
    best_idx = np.argmin(fit)
    best_f, best_X = fit[best_idx], pop[best_idx, :].copy()
    hist = np.zeros(n_iter)

    for it in range(n_iter):
        t_idx = np.argmin(fit)
        teacher = pop[t_idx, :]
        mean_pop = np.mean(pop, axis=0)
        TF = rng.integers(1, 3)

        for i in range(n_pop):
            r = rng.random(dim)
            X_new = np.clip(pop[i, :] + r * (teacher - TF * mean_pop), lb, ub)
            f_new = fit_fn(X_new)
            if f_new < fit[i]:
                pop[i, :], fit[i] = X_new, f_new

        for i in range(n_pop):
            j = rng.integers(0, n_pop - 1)
            if j >= i:
                j += 1
            r = rng.random(dim)
            if fit[i] < fit[j]:
                X_new = pop[i, :] + r * (pop[i, :] - pop[j, :])
            else:
                X_new = pop[i, :] + r * (pop[j, :] - pop[i, :])
            X_new = np.clip(X_new, lb, ub)
            f_new = fit_fn(X_new)
            if f_new < fit[i]:
                pop[i, :], fit[i] = X_new, f_new

        cur_idx = np.argmin(fit)
        if fit[cur_idx] < best_f:
            best_f, best_X = fit[cur_idx], pop[cur_idx, :].copy()
        hist[it] = best_f

    return best_X, best_f, hist


# =============================================================================
# PSO
# =============================================================================
def run_pso(seed=42):
    rng = np.random.default_rng(seed)
    w_max, w_min, c1, c2 = 0.9, 0.4, 1.5, 1.5
    pos = rng.random((n_pop, dim)) * (ub - lb) + lb
    vel = np.zeros((n_pop, dim))
    v_max = 0.2 * (ub - lb)

    fit = np.array([fit_fn(pos[i, :]) for i in range(n_pop)])
    pbest_pos, pbest_fit = pos.copy(), fit.copy()
    g_idx = np.argmin(fit)
    gbest_pos, gbest_fit = pos[g_idx, :].copy(), fit[g_idx]
    hist = np.zeros(n_iter)

    for it in range(n_iter):
        w = w_max - (w_max - w_min) * it / n_iter
        for i in range(n_pop):
            r1, r2 = rng.random(dim), rng.random(dim)
            vel[i, :] = (w * vel[i, :] + c1 * r1 * (pbest_pos[i, :] - pos[i, :])
                         + c2 * r2 * (gbest_pos - pos[i, :]))
            vel[i, :] = np.clip(vel[i, :], -v_max, v_max)
            pos[i, :] = np.clip(pos[i, :] + vel[i, :], lb, ub)
            f = fit_fn(pos[i, :])
            if f < pbest_fit[i]:
                pbest_fit[i], pbest_pos[i, :] = f, pos[i, :].copy()

        c_idx = np.argmin(pbest_fit)
        if pbest_fit[c_idx] < gbest_fit:
            gbest_fit, gbest_pos = pbest_fit[c_idx], pbest_pos[c_idx, :].copy()
        hist[it] = gbest_fit

    return gbest_pos, gbest_fit, hist


# =============================================================================
# GA
# =============================================================================
def run_ga(seed=42):
    rng = np.random.default_rng(seed)
    pc, pm, k_tourn, elite_n = 0.8, 0.1, 3, 2
    pop = rng.random((n_pop, dim)) * (ub - lb) + lb
    fit = np.array([fit_fn(pop[i, :]) for i in range(n_pop)])
    best_idx = np.argmin(fit)
    best_f, best_X = fit[best_idx], pop[best_idx, :].copy()
    hist = np.zeros(n_iter)

    def tourney():
        idx = rng.integers(0, n_pop, size=k_tourn)
        return pop[idx[np.argmin(fit[idx])], :].copy()

    for it in range(n_iter):
        elite_idx = np.argsort(fit)[:elite_n]
        new_pop = [pop[i, :].copy() for i in elite_idx]

        while len(new_pop) < n_pop:
            p1, p2 = tourney(), tourney()
            if rng.random() < pc:
                a = rng.random(dim)
                c1v, c2v = a * p1 + (1 - a) * p2, a * p2 + (1 - a) * p1
            else:
                c1v, c2v = p1.copy(), p2.copy()
            for child in (c1v, c2v):
                mask = rng.random(dim) < pm
                rand_vals = rng.random(dim) * (ub - lb) + lb
                child[mask] = rand_vals[mask]
                child[:] = np.clip(child, lb, ub)
            new_pop.append(c1v)
            if len(new_pop) < n_pop:
                new_pop.append(c2v)

        pop = np.array(new_pop[:n_pop])
        fit = np.array([fit_fn(pop[i, :]) for i in range(n_pop)])
        c_idx = np.argmin(fit)
        if fit[c_idx] < best_f:
            best_f, best_X = fit[c_idx], pop[c_idx, :].copy()
        hist[it] = best_f

    return best_X, best_f, hist


# =============================================================================
# RUN ALL THREE
# =============================================================================
print("\n  Running TLBO...")
X_tlbo, f_tlbo, hist_tlbo = run_tlbo()
print("  Running PSO...")
X_pso, f_pso, hist_pso = run_pso()
print("  Running GA...")
X_ga, f_ga, hist_ga = run_ga()

results = {}
for name, X in [('TLBO', X_tlbo), ('PSO', X_pso), ('GA', X_ga)]:
    Qcap, cap_bus, cap_kvar = decode_solution(X, cap_sizes, n_bus, n_cap)
    V, Ploss = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu, Qcap, n_bus, n_branch)
    Ploss_kW = Ploss * Sbase
    reduction = 100 * (Ploss_base_kW - Ploss_kW) / Ploss_base_kW
    results[name] = dict(V=V, Ploss_kW=Ploss_kW, reduction=reduction,
                          cap_bus=cap_bus, cap_kvar=cap_kvar)

# ── PRINT COMPARISON TABLE ───────────────────────────────────────────────────
print("\n=============================================================")
print("  ALGORITHM COMPARISON - IEEE 33-Bus Capacitor Allocation")
print("=============================================================")
print(f"  {'Algorithm':<10}  {'Loss (kW)':<12}  {'Reduction':<11}  {'Min V (p.u.)':<12}")
print(f"  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*12}")
print(f"  {'Base':<10}  {Ploss_base_kW:<12.2f}  {'-':<11}  {np.min(V_base):<12.4f}")
for name in ['TLBO', 'PSO', 'GA']:
    r = results[name]
    print(f"  {name:<10}  {r['Ploss_kW']:<12.2f}  {r['reduction']:<10.2f}%  {np.min(r['V']):<12.4f}")
print("=============================================================")

for name in ['TLBO', 'PSO', 'GA']:
    r = results[name]
    print(f"\n  {name} Capacitor Placement:")
    for b, s in zip(r['cap_bus'], r['cap_kvar']):
        print(f"    Bus {b:<4d} : {s:.0f} kVAR")

# ── PLOTS ──────────────────────────────────────────────────────────────────
plt.style.use('dark_background')
fig = plt.figure(figsize=(13, 8.5), facecolor='black')
gs = fig.add_gridspec(2, 2)
colors = {'TLBO': '#3fb950', 'PSO': '#58a6ff', 'GA': '#f78166'}

# Plot 1: Voltage profiles
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('black')
ax1.grid(True, color=(0.2, 0.2, 0.2))
buses = np.arange(1, n_bus + 1)
ax1.plot(buses, V_base, 'o--', color='gray', linewidth=1.5, markersize=4, label='Base Case', alpha=0.7)
for name in ['TLBO', 'PSO', 'GA']:
    ax1.plot(buses, results[name]['V'], 'o-', color=colors[name], linewidth=2,
              markersize=4, label=f'{name} Optimized')
ax1.axhline(Vmin, linestyle='--', color='#f85149', linewidth=1.2, alpha=0.7)
ax1.axhline(Vmax, linestyle='--', color='#d29922', linewidth=1.2, alpha=0.7)
ax1.set_xlabel('Bus Number', color='white', fontsize=11)
ax1.set_ylabel('Voltage (p.u.)', color='white', fontsize=11)
ax1.set_title('Voltage Profile Comparison - IEEE 33-Bus System', color='white', fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', facecolor='black', edgecolor=(0.2, 0.2, 0.2), labelcolor='white', fontsize=9)
ax1.set_xlim(1, n_bus)

# Plot 2: Convergence curves
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('black')
ax2.grid(True, color=(0.2, 0.2, 0.2))
ax2.plot(np.arange(1, n_iter + 1), hist_tlbo, color=colors['TLBO'], linewidth=2, label='TLBO')
ax2.plot(np.arange(1, n_iter + 1), hist_pso, color=colors['PSO'], linewidth=2, label='PSO')
ax2.plot(np.arange(1, n_iter + 1), hist_ga, color=colors['GA'], linewidth=2, label='GA')
ax2.set_xlabel('Iteration', color='white', fontsize=11)
ax2.set_ylabel('Best Fitness', color='white', fontsize=11)
ax2.set_title('Convergence Comparison', color='white', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', facecolor='black', edgecolor=(0.2, 0.2, 0.2), labelcolor='white')

# Plot 3: Power loss bar chart
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('black')
ax3.grid(True, color=(0.2, 0.2, 0.2), axis='y')
names = ['Base', 'TLBO', 'PSO', 'GA']
losses = [Ploss_base_kW] + [results[n]['Ploss_kW'] for n in ['TLBO', 'PSO', 'GA']]
bar_colors = ['gray', colors['TLBO'], colors['PSO'], colors['GA']]
bars = ax3.bar(names, losses, color=bar_colors, edgecolor=(0.13, 0.15, 0.18))
for i, v in enumerate(losses):
    ax3.text(i, v + 1, f'{v:.1f}', color='white', ha='center', fontweight='bold', fontsize=9)
ax3.set_ylabel('Real Power Loss (kW)', color='white', fontsize=11)
ax3.set_title('Power Loss Comparison', color='white', fontsize=12, fontweight='bold')

fig.suptitle('Capacitor Allocation: TLBO vs PSO vs GA - IEEE 33-Bus System',
             color='white', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('comparison_results.png', dpi=150, facecolor='black')
print("\n  Plot saved as 'comparison_results.png'")
plt.show()
