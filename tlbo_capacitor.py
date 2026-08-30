"""
=============================================================================
 Capacitor Allocation and Sizing in Distribution Systems
 Using Teaching-Learning Based Optimization (TLBO)
 Reference: IEEE 33-Bus Radial Distribution System
 Method: Rao et al. (2011) - TLBO Algorithm
 (Python reimplementation of MATLAB TLBO dissertation code)
=============================================================================
 HOW TO RUN:
   1. Open terminal in this folder
   2. Run: py -3.12 tlbo_capacitor.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ── SYSTEM PARAMETERS ───────────────────────────────────────────────────────
Vbase = 12.66      # kV
Sbase = 1000       # kVA
Vmin = 0.90        # p.u.
Vmax = 1.05        # p.u.
n_bus = 33

# Standard Capacitor Sizes (kVAR)
cap_sizes = np.array([0, 150, 300, 450, 600, 750, 900, 1050, 1200])

# ── IEEE 33-BUS BRANCH DATA ─────────────────────────────────────────────────
# Columns: [From, To, R(ohm), X(ohm), P_load(kW), Q_load(kVAR)]
branch = np.array([
    [1, 2, 0.0922, 0.0470, 100, 60],
    [2, 3, 0.4930, 0.2511, 90, 40],
    [3, 4, 0.3660, 0.1864, 120, 80],
    [4, 5, 0.3811, 0.1941, 60, 30],
    [5, 6, 0.8190, 0.7070, 60, 20],
    [6, 7, 0.1872, 0.6188, 200, 100],
    [7, 8, 0.7114, 0.2351, 200, 100],
    [8, 9, 1.0300, 0.7400, 60, 20],
    [9, 10, 1.0440, 0.7400, 60, 20],
    [10, 11, 0.1966, 0.0650, 45, 30],
    [11, 12, 0.3744, 0.1238, 60, 35],
    [12, 13, 1.4680, 1.1550, 60, 35],
    [13, 14, 0.5416, 0.7129, 120, 80],
    [14, 15, 0.5910, 0.5260, 60, 10],
    [15, 16, 0.7463, 0.5450, 60, 20],
    [16, 17, 1.2890, 1.7210, 60, 20],
    [17, 18, 0.7320, 0.5740, 90, 40],
    [2, 19, 0.1640, 0.1565, 90, 40],
    [19, 20, 1.5042, 1.3554, 90, 40],
    [20, 21, 0.4095, 0.4784, 90, 40],
    [21, 22, 0.7089, 0.9373, 90, 40],
    [3, 23, 0.4512, 0.3083, 90, 50],
    [23, 24, 0.8980, 0.7091, 420, 200],
    [24, 25, 0.8960, 0.7011, 420, 200],
    [6, 26, 0.2030, 0.1034, 60, 25],
    [26, 27, 0.2842, 0.1447, 60, 25],
    [27, 28, 1.0590, 0.9337, 60, 20],
    [28, 29, 0.8042, 0.7006, 120, 70],
    [29, 30, 0.5075, 0.2585, 200, 600],
    [30, 31, 0.9744, 0.9630, 150, 70],
    [31, 32, 0.3105, 0.3619, 210, 100],
    [32, 33, 0.3410, 0.5302, 60, 40],
])

from_bus = branch[:, 0].astype(int)
to_bus = branch[:, 1].astype(int)
R = branch[:, 2]
X = branch[:, 3]
P_load = branch[:, 4]   # kW
Q_load = branch[:, 5]   # kVAR
n_branch = branch.shape[0]

# Base impedance conversion
Zbase = (Vbase * 1e3) ** 2 / (Sbase * 1e3)
R_pu = R / Zbase
X_pu = X / Zbase
P_pu = P_load / Sbase
Q_pu = Q_load / Sbase

P_total = np.sum(P_load)
Q_total = np.sum(Q_load)

print("\n=============================================================")
print("  IEEE 33-Bus System Loaded")
print(f"  Total Load : {P_total:.0f} kW  +  {Q_total:.0f} kVAR")
print("=============================================================")


# ── BFS LOAD FLOW FUNCTION ──────────────────────────────────────────────────
def bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu, Qcap_pu, n_bus, n_branch):
    """Backward-Forward Sweep Load Flow for Radial Distribution Network.
    Buses are 1-indexed (bus 1..n_bus); arrays are 0-indexed internally
    with a +1 offset lookup to match MATLAB's 1-based indexing.
    """
    P_bus = np.zeros(n_bus + 1)
    Q_bus = np.zeros(n_bus + 1)
    for k in range(n_branch):
        tb = to_bus[k]
        P_bus[tb] = P_pu[k]
        Q_bus[tb] = Q_pu[k] - Qcap_pu[tb]

    V2 = np.ones(n_bus + 1)
    P_br = np.zeros(n_branch)
    Q_br = np.zeros(n_branch)

    for _ in range(100):
        V2_old = V2.copy()

        # Backward sweep
        for k in range(n_branch - 1, -1, -1):
            tb = to_bus[k]
            P_br[k] = P_bus[tb]
            Q_br[k] = Q_bus[tb]
            for m in range(n_branch):
                if from_bus[m] == tb:
                    I2m = (P_br[m] ** 2 + Q_br[m] ** 2) / max(V2[tb], 1e-6)
                    P_br[k] += P_br[m] - R_pu[m] * I2m
                    Q_br[k] += Q_br[m] - X_pu[m] * I2m

        # Forward sweep
        V2[1] = 1.0
        for k in range(n_branch):
            fb = from_bus[k]
            tb = to_bus[k]
            I2 = (P_br[k] ** 2 + Q_br[k] ** 2) / max(V2[fb], 1e-6)
            V2[tb] = V2[fb] - 2 * (R_pu[k] * P_br[k] + X_pu[k] * Q_br[k]) \
                + (R_pu[k] ** 2 + X_pu[k] ** 2) * I2

        if np.max(np.abs(V2 - V2_old)) < 1e-6:
            break

    V = np.sqrt(np.abs(V2[1:n_bus + 1]))  # drop index 0, keep buses 1..n_bus

    # Power loss
    Ploss = 0.0
    for k in range(n_branch):
        fb = from_bus[k]
        I2 = (P_br[k] ** 2 + Q_br[k] ** 2) / max(V2[fb], 1e-6)
        Ploss += R_pu[k] * I2

    return V, Ploss


# ── DECODE SOLUTION FUNCTION ────────────────────────────────────────────────
def decode_solution(Xvec, cap_sizes, n_bus, n_cap):
    """Decode continuous TLBO variable to capacitor bus & size (1-indexed buses)."""
    Qcap = np.zeros(n_bus + 1)  # index 0 unused, buses 1..n_bus
    cap_bus = np.zeros(n_cap, dtype=int)
    cap_kvar = np.zeros(n_cap)
    for i in range(n_cap):
        bus = int(max(1, min(n_bus - 1, round(Xvec[2 * i]))))
        sidx = int(max(0, min(len(cap_sizes) - 1, round(Xvec[2 * i + 1]))))
        sz = cap_sizes[sidx]
        Qcap[bus] += sz / 1000.0   # p.u. (Sbase = 1000 kVA)
        cap_bus[i] = bus
        cap_kvar[i] = sz
    return Qcap, cap_bus, cap_kvar


# ── FITNESS FUNCTION ─────────────────────────────────────────────────────────
def evaluate_fitness(Xvec, cap_sizes, from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                      n_bus, n_branch, Sbase, Vmin, Vmax):
    """fitness = w1*Ploss + w2*VDI + penalty*violations"""
    w1, w2, pen = 1.0, 0.5, 1e6
    n_cap = len(Xvec) // 2
    Qcap, _, _ = decode_solution(Xvec, cap_sizes, n_bus, n_cap)
    V, Ploss = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu, Qcap, n_bus, n_branch)
    VDI = np.sum((V - 1.0) ** 2)
    viol = np.sum(np.maximum(0, Vmin - V) ** 2 + np.maximum(0, V - Vmax) ** 2)
    f = w1 * Ploss * Sbase + w2 * VDI * 1000 + pen * viol
    return f


# ── BASE CASE LOAD FLOW (No Capacitors) ─────────────────────────────────────
Qcap_base = np.zeros(n_bus + 1)
V_base, Ploss_base = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                                    Qcap_base, n_bus, n_branch)
Ploss_base_kW = Ploss_base * Sbase

minV_base = np.min(V_base)
minV_bus = np.argmin(V_base) + 1  # 1-indexed bus number

print("\n  BASE CASE RESULTS:")
print(f"  Total Power Loss : {Ploss_base_kW:.4f} kW")
print(f"  Min Voltage      : {minV_base:.4f} p.u.  (Bus {minV_bus})")

# ── TLBO PARAMETERS ──────────────────────────────────────────────────────────
n_pop = 30      # Number of students (population)
n_iter = 100    # Number of iterations
n_cap = 3       # Number of capacitors to place
dim = 2 * n_cap  # [bus, size_index] per capacitor

# Variable bounds (interleaved: bus, size_idx, bus, size_idx, ...)
lb = np.zeros(dim)
ub = np.zeros(dim)
for i in range(n_cap):
    lb[2 * i] = 1
    lb[2 * i + 1] = 0
    ub[2 * i] = n_bus - 1
    ub[2 * i + 1] = len(cap_sizes) - 1

# ── TLBO ALGORITHM ───────────────────────────────────────────────────────────
rng = np.random.default_rng(42)  # for reproducibility

# Initialize population
pop = rng.random((n_pop, dim)) * (ub - lb) + lb

# Evaluate initial fitness
fit = np.zeros(n_pop)
for i in range(n_pop):
    fit[i] = evaluate_fitness(pop[i, :], cap_sizes, from_bus, to_bus, R_pu, X_pu,
                               P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)

best_idx = np.argmin(fit)
best_f = fit[best_idx]
best_X = pop[best_idx, :].copy()

best_hist = np.zeros(n_iter)
avg_hist = np.zeros(n_iter)

print("\n=============================================================")
print(f"  TLBO: {n_pop} students | {n_iter} iterations | {n_cap} capacitors")
print("=============================================================")

for it in range(n_iter):

    # ── TEACHER PHASE ────────────────────────────────────────────────────
    t_idx = np.argmin(fit)
    teacher = pop[t_idx, :]
    mean_pop = np.mean(pop, axis=0)
    TF = rng.integers(1, 3)  # Teaching Factor: 1 or 2

    for i in range(n_pop):
        r = rng.random(dim)
        X_new = pop[i, :] + r * (teacher - TF * mean_pop)
        X_new = np.clip(X_new, lb, ub)
        f_new = evaluate_fitness(X_new, cap_sizes, from_bus, to_bus, R_pu, X_pu,
                                  P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)
        if f_new < fit[i]:
            pop[i, :] = X_new
            fit[i] = f_new

    # ── LEARNER PHASE ────────────────────────────────────────────────────
    for i in range(n_pop):
        candidates = [x for x in range(n_pop) if x != i]
        j = candidates[rng.integers(0, len(candidates))]

        r = rng.random(dim)
        if fit[i] < fit[j]:
            X_new = pop[i, :] + r * (pop[i, :] - pop[j, :])
        else:
            X_new = pop[i, :] + r * (pop[j, :] - pop[i, :])
        X_new = np.clip(X_new, lb, ub)
        f_new = evaluate_fitness(X_new, cap_sizes, from_bus, to_bus, R_pu, X_pu,
                                  P_pu, Q_pu, n_bus, n_branch, Sbase, Vmin, Vmax)
        if f_new < fit[i]:
            pop[i, :] = X_new
            fit[i] = f_new

    # ── UPDATE BEST ──────────────────────────────────────────────────────
    cur_idx = np.argmin(fit)
    cur_best_f = fit[cur_idx]
    if cur_best_f < best_f:
        best_f = cur_best_f
        best_X = pop[cur_idx, :].copy()

    best_hist[it] = best_f
    avg_hist[it] = np.mean(fit)

    if (it + 1) % 20 == 0:
        print(f"  Iter {it + 1:4d}/{n_iter} | Best Fitness = {best_f:.4f}")

print(f"\n  Done! Best Fitness = {best_f:.4f}")

# ── DECODE BEST SOLUTION ─────────────────────────────────────────────────────
Qcap_opt, cap_bus, cap_kvar = decode_solution(best_X, cap_sizes, n_bus, n_cap)
V_opt, Ploss_opt = bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu,
                                  Qcap_opt, n_bus, n_branch)
Ploss_opt_kW = Ploss_opt * Sbase
reduction = 100 * (Ploss_base_kW - Ploss_opt_kW) / Ploss_base_kW

# ── PRINT RESULTS ─────────────────────────────────────────────────────────────
print("\n=============================================================")
print("  OPTIMAL CAPACITOR PLACEMENT RESULTS")
print("=============================================================")
print(f"  {'Bus':<6}  {'Size (kVAR)':<14}")
print(f"  {'------':<6}  {'----------':<14}")
for i in range(n_cap):
    print(f"  {cap_bus[i]:<6d}  {cap_kvar[i]:<14.0f}")
print(f"\n  Total Capacitor : {np.sum(cap_kvar):.0f} kVAR")
print(f"\n  Base Case Power Loss  : {Ploss_base_kW:.4f} kW")
print(f"  Optimized Power Loss  : {Ploss_opt_kW:.4f} kW")
print(f"  Loss Reduction        : {reduction:.2f}%")
print(f"\n  Base Case Min Voltage : {np.min(V_base):.4f} p.u.")
print(f"  Optimized Min Voltage : {np.min(V_opt):.4f} p.u.")
print("=============================================================")

# ── PLOTS ──────────────────────────────────────────────────────────────────
buses = np.arange(1, n_bus + 1)

plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 8.2), facecolor='black')
gs = fig.add_gridspec(2, 2)

# Plot 1: Voltage Profile (spans top row)
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('black')
ax1.grid(True, color=(0.2, 0.2, 0.2))
ax1.plot(buses, V_base, 'o-', color='#58a6ff', linewidth=2, markersize=5, label='Base Case')
ax1.plot(buses, V_opt, 's-', color='#3fb950', linewidth=2, markersize=5, label='Optimized (TLBO)')
ax1.axhline(Vmin, linestyle='--', color='#f85149', linewidth=1.5)
ax1.text(1, Vmin, 'V_min=0.90', color='#f85149', fontsize=9, va='bottom')
ax1.axhline(Vmax, linestyle='--', color='#d29922', linewidth=1.5)
ax1.text(1, Vmax, 'V_max=1.05', color='#d29922', fontsize=9, va='bottom')
for i in range(n_cap):
    if cap_kvar[i] > 0:
        ax1.axvline(cap_bus[i], linestyle=':', color='#f78166', linewidth=1.2, alpha=0.8)
        ax1.text(cap_bus[i] + 0.3, V_opt[cap_bus[i] - 1] + 0.006,
                  f'{cap_kvar[i]:.0f} kVAR', color='#f78166', fontsize=8)
ax1.set_xlabel('Bus Number', color='white', fontsize=11)
ax1.set_ylabel('Voltage (p.u.)', color='white', fontsize=11)
ax1.set_title('Voltage Profile - IEEE 33-Bus System', color='white', fontsize=13, fontweight='bold')
ax1.legend(loc='lower left', facecolor='black', edgecolor=(0.2, 0.2, 0.2), labelcolor='white')
ax1.set_xlim(1, n_bus)

# Plot 2: TLBO Convergence
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('black')
ax2.grid(True, color=(0.2, 0.2, 0.2))
ax2.plot(np.arange(1, n_iter + 1), best_hist, color='#3fb950', linewidth=2, label='Best Fitness')
ax2.plot(np.arange(1, n_iter + 1), avg_hist, '--', color='#58a6ff', linewidth=1.5, label='Avg Fitness')
ax2.set_xlabel('Iteration', color='white', fontsize=11)
ax2.set_ylabel('Fitness Value', color='white', fontsize=11)
ax2.set_title('TLBO Convergence Curve', color='white', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', facecolor='black', edgecolor=(0.2, 0.2, 0.2), labelcolor='white')

# Plot 3: Power Loss Bar Chart
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('black')
ax3.grid(True, color=(0.2, 0.2, 0.2))
bars = ax3.bar(['Base Case', 'TLBO Optimized'], [Ploss_base_kW, Ploss_opt_kW],
                width=0.45, color=['#58a6ff', '#3fb950'], edgecolor=(0.13, 0.15, 0.18))
ax3.text(0, Ploss_base_kW + 0.5, f'{Ploss_base_kW:.2f} kW', color='white',
          ha='center', fontweight='bold', fontsize=10)
ax3.text(1, Ploss_opt_kW + 0.5, f'{Ploss_opt_kW:.2f} kW', color='white',
          ha='center', fontweight='bold', fontsize=10)
ax3.set_ylabel('Real Power Loss (kW)', color='white', fontsize=11)
ax3.set_title(f'Power Loss Comparison (down {reduction:.1f}%)', color='white', fontsize=12, fontweight='bold')

fig.suptitle('Capacitor Allocation & Sizing - TLBO on IEEE 33-Bus Distribution System',
             color='white', fontsize=14, fontweight='bold')
fig.tight_layout()

# ── SAVE RESULTS TABLE ────────────────────────────────────────────────────────
print("\n  Per-Bus Voltage Summary:")
print(f"  {'Bus':<5}  {'V_base(p.u.)':<14}  {'V_opt(p.u.)':<14}  {'dV(p.u.)':<12}  {'Cap(kVAR)':<10}")
print(f"  {'-----':<5}  {'------------':<14}  {'----------':<14}  {'--------':<12}  {'--------':<10}")
for b_num in range(1, n_bus + 1):
    cap_here = 0
    for i in range(n_cap):
        if cap_bus[i] == b_num:
            cap_here += cap_kvar[i]
    print(f"  {b_num:<5d}  {V_base[b_num-1]:<14.4f}  {V_opt[b_num-1]:<14.4f}  "
          f"{V_opt[b_num-1]-V_base[b_num-1]:<12.4f}  {cap_here:<10.0f}")

print("\n  Simulation Complete!\n")

plt.savefig('tlbo_results.png', dpi=150, facecolor='black')
print("  Plot saved as 'tlbo_results.png'")
plt.show()
