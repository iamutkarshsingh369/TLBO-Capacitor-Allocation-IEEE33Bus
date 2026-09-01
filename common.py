"""
=============================================================================
 Shared functions for Capacitor Allocation optimization
 (Load flow, fitness evaluation, solution decoding, system data)
 Used by TLBO, PSO, and GA so comparison is apples-to-apples.
=============================================================================
"""

import numpy as np

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
P_load = branch[:, 4]
Q_load = branch[:, 5]
n_branch = branch.shape[0]

Zbase = (Vbase * 1e3) ** 2 / (Sbase * 1e3)
R_pu = R / Zbase
X_pu = X / Zbase
P_pu = P_load / Sbase
Q_pu = Q_load / Sbase

P_total = np.sum(P_load)
Q_total = np.sum(Q_load)


# ── BFS LOAD FLOW FUNCTION ──────────────────────────────────────────────────
def bfs_load_flow(from_bus, to_bus, R_pu, X_pu, P_pu, Q_pu, Qcap_pu, n_bus, n_branch):
    """Backward-Forward Sweep Load Flow for Radial Distribution Network."""
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

        for k in range(n_branch - 1, -1, -1):
            tb = to_bus[k]
            P_br[k] = P_bus[tb]
            Q_br[k] = Q_bus[tb]
            for m in range(n_branch):
                if from_bus[m] == tb:
                    I2m = (P_br[m] ** 2 + Q_br[m] ** 2) / max(V2[tb], 1e-6)
                    P_br[k] += P_br[m] - R_pu[m] * I2m
                    Q_br[k] += Q_br[m] - X_pu[m] * I2m

        V2[1] = 1.0
        for k in range(n_branch):
            fb = from_bus[k]
            tb = to_bus[k]
            I2 = (P_br[k] ** 2 + Q_br[k] ** 2) / max(V2[fb], 1e-6)
            V2[tb] = V2[fb] - 2 * (R_pu[k] * P_br[k] + X_pu[k] * Q_br[k]) \
                + (R_pu[k] ** 2 + X_pu[k] ** 2) * I2

        if np.max(np.abs(V2 - V2_old)) < 1e-6:
            break

    V = np.sqrt(np.abs(V2[1:n_bus + 1]))

    Ploss = 0.0
    for k in range(n_branch):
        fb = from_bus[k]
        I2 = (P_br[k] ** 2 + Q_br[k] ** 2) / max(V2[fb], 1e-6)
        Ploss += R_pu[k] * I2

    return V, Ploss


# ── DECODE SOLUTION FUNCTION ────────────────────────────────────────────────
def decode_solution(Xvec, cap_sizes, n_bus, n_cap):
    """Decode continuous variable to capacitor bus & size (1-indexed buses)."""
    Qcap = np.zeros(n_bus + 1)
    cap_bus = np.zeros(n_cap, dtype=int)
    cap_kvar = np.zeros(n_cap)
    for i in range(n_cap):
        bus = int(max(1, min(n_bus - 1, round(Xvec[2 * i]))))
        sidx = int(max(0, min(len(cap_sizes) - 1, round(Xvec[2 * i + 1]))))
        sz = cap_sizes[sidx]
        Qcap[bus] += sz / 1000.0
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
