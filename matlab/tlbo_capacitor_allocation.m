%% =========================================================================
%  Capacitor Allocation and Sizing in Distribution Systems
%  Using Teaching-Learning Based Optimization (TLBO)
%  Reference: IEEE 33-Bus Radial Distribution System
%  Method: Rao et al. (2011) - TLBO Algorithm
%% =========================================================================
%  HOW TO RUN:
%    1. Open MATLAB
%    2. Open this file
%    3. Press F5 or click "Run"
%% =========================================================================

clc; clear; close all;

%% ── SYSTEM PARAMETERS ────────────────────────────────────────────────────
Vbase  = 12.66;     % kV
Sbase  = 1000;      % kVA
Vmin   = 0.90;      % p.u.
Vmax   = 1.05;      % p.u.
n_bus  = 33;

% Standard Capacitor Sizes (kVAR)
cap_sizes = [0, 150, 300, 450, 600, 750, 900, 1050, 1200];

%% ── IEEE 33-BUS BRANCH DATA ──────────────────────────────────────────────
% Columns: [From  To  R(ohm)  X(ohm)  P_load(kW)  Q_load(kVAR)]
branch = [
    1   2   0.0922  0.0470  100  60;
    2   3   0.4930  0.2511  90   40;
    3   4   0.3660  0.1864  120  80;
    4   5   0.3811  0.1941  60   30;
    5   6   0.8190  0.7070  60   20;
    6   7   0.1872  0.6188  200  100;
    7   8   0.7114  0.2351  200  100;
    8   9   1.0300  0.7400  60   20;
    9   10  1.0440  0.7400  60   20;
    10  11  0.1966  0.0650  45   30;
    11  12  0.3744  0.1238  60   35;
    12  13  1.4680  1.1550  60   35;
    13  14  0.5416  0.7129  120  80;
    14  15  0.5910  0.5260  60   10;
    15  16  0.7463  0.5450  60   20;
    16  17  1.2890  1.7210  60   20;
    17  18  0.7320  0.5740  90   40;
    2   19  0.1640  0.1565  90   40;
    19  20  1.5042  1.3554  90   40;
    20  21  0.4095  0.4784  90   40;
    21  22  0.7089  0.9373  90   40;
    3   23  0.4512  0.3083  90   50;
    23  24  0.8980  0.7091  420  200;
    24  25  0.8960  0.7011  420  200;
    6   26  0.2030  0.1034  60   25;
    26  27  0.2842  0.1447  60   25;
    27  28  1.0590  0.9337  60   20;
    28  29  0.8042  0.7006  120  70;
    29  30  0.5075  0.2585  200  600;
    30  31  0.9744  0.9630  150  70;
    31  32  0.3105  0.3619  210  100;
    32  33  0.3410  0.5302  60   40;
];

from_bus = branch(:,1);
to_bus   = branch(:,2);
R        = branch(:,3);
X        = branch(:,4);
P_load   = branch(:,5);   % kW
Q_load   = branch(:,6);   % kVAR
n_branch = size(branch,1);

% Base impedance conversion
Zbase = (Vbase*1e3)^2 / (Sbase*1e3);
R_pu  = R / Zbase;
X_pu  = X / Zbase;
P_pu  = P_load / Sbase;
Q_pu  = Q_load / Sbase;

P_total = sum(P_load);
Q_total = sum(Q_load);

fprintf('\n=============================================================\n');
fprintf('  IEEE 33-Bus System Loaded\n');
fprintf('  Total Load : %.0f kW  +  %.0f kVAR\n', P_total, Q_total);
fprintf('=============================================================\n');

%% ── BASE CASE LOAD FLOW (No Capacitors) ──────────────────────────────────
Qcap_base = zeros(n_bus, 1);
[V_base, Ploss_base] = BFS_LoadFlow(from_bus, to_bus, R_pu, X_pu, ...
                                     P_pu, Q_pu, Qcap_base, n_bus, n_branch);
Ploss_base_kW = Ploss_base * Sbase;

[minV_base, minV_bus] = min(V_base);
fprintf('\n  BASE CASE RESULTS:\n');
fprintf('  Total Power Loss : %.4f kW\n', Ploss_base_kW);
fprintf('  Min Voltage      : %.4f p.u.  (Bus %d)\n', minV_base, minV_bus);

%% ── TLBO PARAMETERS ──────────────────────────────────────────────────────
n_pop  = 30;    % Number of students (population)
n_iter = 100;   % Number of iterations
n_cap  = 3;     % Number of capacitors to place
dim    = 2 * n_cap;  % [bus, size_index] per capacitor

% Variable bounds
lb = [ones(1,n_cap)*1;    zeros(1,n_cap)];   lb = lb(:)';
ub = [ones(1,n_cap)*(n_bus-1); ones(1,n_cap)*(length(cap_sizes)-1)]; ub = ub(:)';

%% ── TLBO ALGORITHM ───────────────────────────────────────────────────────
rng(42);  % for reproducibility

% Initialize population
pop = rand(n_pop, dim) .* (ub - lb) + lb;

% Evaluate initial fitness
fit = zeros(n_pop, 1);
for i = 1:n_pop
    fit(i) = evaluate_fitness(pop(i,:), cap_sizes, from_bus, to_bus, ...
                               R_pu, X_pu, P_pu, Q_pu, n_bus, n_branch, ...
                               Sbase, Vmin, Vmax);
end

[best_f, best_idx] = min(fit);
best_X = pop(best_idx, :);

best_hist = zeros(n_iter, 1);
avg_hist  = zeros(n_iter, 1);

fprintf('\n=============================================================\n');
fprintf('  TLBO: %d students | %d iterations | %d capacitors\n', n_pop, n_iter, n_cap);
fprintf('=============================================================\n');

for it = 1:n_iter

    %% ── TEACHER PHASE ────────────────────────────────────────────────────
    [~, t_idx] = min(fit);
    teacher  = pop(t_idx, :);
    mean_pop = mean(pop, 1);
    TF       = randi([1 2]);   % Teaching Factor: 1 or 2

    for i = 1:n_pop
        r     = rand(1, dim);
        X_new = pop(i,:) + r .* (teacher - TF * mean_pop);
        X_new = max(lb, min(ub, X_new));   % clip to bounds
        f_new = evaluate_fitness(X_new, cap_sizes, from_bus, to_bus, ...
                                  R_pu, X_pu, P_pu, Q_pu, n_bus, n_branch, ...
                                  Sbase, Vmin, Vmax);
        if f_new < fit(i)
            pop(i,:) = X_new;
            fit(i)   = f_new;
        end
    end

    %% ── LEARNER PHASE ────────────────────────────────────────────────────
    for i = 1:n_pop
        % Pick a random different student
        candidates = setdiff(1:n_pop, i);
        j = candidates(randi(length(candidates)));

        r = rand(1, dim);
        if fit(i) < fit(j)
            X_new = pop(i,:) + r .* (pop(i,:) - pop(j,:));
        else
            X_new = pop(i,:) + r .* (pop(j,:) - pop(i,:));
        end
        X_new = max(lb, min(ub, X_new));
        f_new = evaluate_fitness(X_new, cap_sizes, from_bus, to_bus, ...
                                  R_pu, X_pu, P_pu, Q_pu, n_bus, n_branch, ...
                                  Sbase, Vmin, Vmax);
        if f_new < fit(i)
            pop(i,:) = X_new;
            fit(i)   = f_new;
        end
    end

    %% ── UPDATE BEST ──────────────────────────────────────────────────────
    [cur_best_f, cur_idx] = min(fit);
    if cur_best_f < best_f
        best_f = cur_best_f;
        best_X = pop(cur_idx, :);
    end

    best_hist(it) = best_f;
    avg_hist(it)  = mean(fit);

    if mod(it, 20) == 0
        fprintf('  Iter %4d/%d | Best Fitness = %.4f\n', it, n_iter, best_f);
    end
end

fprintf('\n  Done! Best Fitness = %.4f\n', best_f);

%% ── DECODE BEST SOLUTION ─────────────────────────────────────────────────
[Qcap_opt, cap_bus, cap_kvar] = decode_solution(best_X, cap_sizes, n_bus, n_cap);
[V_opt, Ploss_opt] = BFS_LoadFlow(from_bus, to_bus, R_pu, X_pu, ...
                                   P_pu, Q_pu, Qcap_opt, n_bus, n_branch);
Ploss_opt_kW = Ploss_opt * Sbase;
reduction    = 100*(Ploss_base_kW - Ploss_opt_kW)/Ploss_base_kW;

%% ── PRINT RESULTS ────────────────────────────────────────────────────────
fprintf('\n=============================================================\n');
fprintf('  OPTIMAL CAPACITOR PLACEMENT RESULTS\n');
fprintf('=============================================================\n');
fprintf('  %-6s  %-14s\n','Bus','Size (kVAR)');
fprintf('  %-6s  %-14s\n','------','----------');
for i = 1:n_cap
    fprintf('  %-6d  %-14.0f\n', cap_bus(i), cap_kvar(i));
end
fprintf('\n  Total Capacitor : %.0f kVAR\n', sum(cap_kvar));
fprintf('\n  Base Case Power Loss  : %.4f kW\n', Ploss_base_kW);
fprintf('  Optimized Power Loss  : %.4f kW\n', Ploss_opt_kW);
fprintf('  Loss Reduction        : %.2f%%\n',  reduction);
fprintf('\n  Base Case Min Voltage : %.4f p.u.\n', min(V_base));
fprintf('  Optimized Min Voltage : %.4f p.u.\n', min(V_opt));
fprintf('=============================================================\n');

%% ── PLOTS ────────────────────────────────────────────────────────────────
buses = 1:n_bus;
fig = figure('Color','k','Position',[100 80 1200 820]);

%% Plot 1: Voltage Profile
subplot(2,2,[1 2]);
set(gca,'Color','k','XColor','w','YColor','w','GridColor',[0.2 0.2 0.2],'GridAlpha',1);
hold on; grid on;
plot(buses, V_base, 'o-', 'Color','#58a6ff','LineWidth',2,'MarkerSize',5,'DisplayName','Base Case');
plot(buses, V_opt,  's-', 'Color','#3fb950','LineWidth',2,'MarkerSize',5,'DisplayName','Optimized (TLBO)');
yline(Vmin,'--','Color','#f85149','LineWidth',1.5,'Label','V_{min}=0.90','LabelHorizontalAlignment','left');
yline(Vmax,'--','Color','#d29922','LineWidth',1.5,'Label','V_{max}=1.05','LabelHorizontalAlignment','left');
for i = 1:n_cap
    if cap_kvar(i) > 0
        xline(cap_bus(i),':','Color','#f78166','LineWidth',1.2,'Alpha',0.8);
        text(cap_bus(i)+0.3, V_opt(cap_bus(i))+0.006, ...
             sprintf('%d kVAR',cap_kvar(i)),'Color','#f78166','FontSize',8);
    end
end
xlabel('Bus Number','Color','w','FontSize',11);
ylabel('Voltage (p.u.)','Color','w','FontSize',11);
title('Voltage Profile – IEEE 33-Bus System','Color','w','FontSize',13,'FontWeight','bold');
legend('Location','southwest','TextColor','w','Color','k','EdgeColor',[0.2 0.2 0.2]);
xlim([1 n_bus]);

%% Plot 2: TLBO Convergence
subplot(2,2,3);
set(gca,'Color','k','XColor','w','YColor','w','GridColor',[0.2 0.2 0.2],'GridAlpha',1);
hold on; grid on;
plot(1:n_iter, best_hist,'Color','#3fb950','LineWidth',2,'DisplayName','Best Fitness');
plot(1:n_iter, avg_hist, '--','Color','#58a6ff','LineWidth',1.5,'DisplayName','Avg Fitness');
xlabel('Iteration','Color','w','FontSize',11);
ylabel('Fitness Value','Color','w','FontSize',11);
title('TLBO Convergence Curve','Color','w','FontSize',12,'FontWeight','bold');
legend('Location','northeast','TextColor','w','Color','k','EdgeColor',[0.2 0.2 0.2]);

%% Plot 3: Power Loss Bar Chart
subplot(2,2,4);
set(gca,'Color','k','XColor','w','YColor','w','GridColor',[0.2 0.2 0.2],'GridAlpha',1);
hold on; grid on;
b = bar([Ploss_base_kW, Ploss_opt_kW], 0.45);
b.FaceColor = 'flat';
b.CData = [0.345 0.647 1.0; 0.247 0.722 0.314];
b.EdgeColor = [0.13 0.15 0.18];
text(1, Ploss_base_kW+0.5, sprintf('%.2f kW',Ploss_base_kW),'Color','w','HorizontalAlignment','center','FontWeight','bold','FontSize',10);
text(2, Ploss_opt_kW+0.5,  sprintf('%.2f kW',Ploss_opt_kW), 'Color','w','HorizontalAlignment','center','FontWeight','bold','FontSize',10);
set(gca,'XTickLabel',{'Base Case','TLBO Optimized'});
ylabel('Real Power Loss (kW)','Color','w','FontSize',11);
title(sprintf('Power Loss Comparison (↓ %.1f%%)',reduction),'Color','w','FontSize',12,'FontWeight','bold');

% Overall figure title
sgtitle('Capacitor Allocation & Sizing — TLBO on IEEE 33-Bus Distribution System',...
        'Color','w','FontSize',14,'FontWeight','bold');

%% ── SAVE RESULTS TABLE ───────────────────────────────────────────────────
fprintf('\n  Per-Bus Voltage Summary:\n');
fprintf('  %-5s  %-14s  %-14s  %-12s  %-10s\n','Bus','V_base(p.u.)','V_opt(p.u.)','ΔV(p.u.)','Cap(kVAR)');
fprintf('  %-5s  %-14s  %-14s  %-12s  %-10s\n','-----','------------','----------','--------','--------');
for b_num = 1:n_bus
    cap_here = 0;
    for i = 1:n_cap
        if cap_bus(i) == b_num
            cap_here = cap_here + cap_kvar(i);
        end
    end
    fprintf('  %-5d  %-14.4f  %-14.4f  %-12.4f  %-10.0f\n',...
            b_num, V_base(b_num), V_opt(b_num), V_opt(b_num)-V_base(b_num), cap_here);
end

fprintf('\n  ✓ Simulation Complete!\n\n');

%% =========================================================================
%% LOCAL FUNCTIONS
%% =========================================================================

function [V, Ploss] = BFS_LoadFlow(from_bus, to_bus, R_pu, X_pu, ...
                                    P_pu, Q_pu, Qcap_pu, n_bus, n_branch)
%% Backward-Forward Sweep Load Flow for Radial Distribution Network
    P_bus = zeros(n_bus,1);
    Q_bus = zeros(n_bus,1);
    for k = 1:n_branch
        tb = to_bus(k);
        P_bus(tb) = P_pu(k);
        Q_bus(tb) = Q_pu(k) - Qcap_pu(tb);
    end

    V2 = ones(n_bus,1);
    P_br = zeros(n_branch,1);
    Q_br = zeros(n_branch,1);

    for iter = 1:100
        V2_old = V2;

        % Backward sweep
        for k = n_branch:-1:1
            tb = to_bus(k);
            P_br(k) = P_bus(tb);
            Q_br(k) = Q_bus(tb);
            for m = 1:n_branch
                if from_bus(m) == tb
                    I2m = (P_br(m)^2 + Q_br(m)^2) / max(V2(tb), 1e-6);
                    P_br(k) = P_br(k) + P_br(m) - R_pu(m)*I2m;
                    Q_br(k) = Q_br(k) + Q_br(m) - X_pu(m)*I2m;
                end
            end
        end

        % Forward sweep
        V2(1) = 1.0;
        for k = 1:n_branch
            fb = from_bus(k); tb = to_bus(k);
            I2 = (P_br(k)^2 + Q_br(k)^2) / max(V2(fb), 1e-6);
            V2(tb) = V2(fb) - 2*(R_pu(k)*P_br(k) + X_pu(k)*Q_br(k)) ...
                     + (R_pu(k)^2 + X_pu(k)^2)*I2;
        end

        if max(abs(V2 - V2_old)) < 1e-6, break; end
    end

    V = sqrt(abs(V2));

    % Power loss
    Ploss = 0;
    for k = 1:n_branch
        fb = from_bus(k);
        I2 = (P_br(k)^2 + Q_br(k)^2) / max(V2(fb), 1e-6);
        Ploss = Ploss + R_pu(k) * I2;
    end
end

%% ─────────────────────────────────────────────────────────────────────────

function f = evaluate_fitness(X, cap_sizes, from_bus, to_bus, ...
                               R_pu, X_pu, P_pu, Q_pu, n_bus, n_branch, ...
                               Sbase, Vmin, Vmax)
%% Compute fitness = w1*Ploss + w2*VDI + penalty*violations
    w1 = 1.0; w2 = 0.5; pen = 1e6;
    [Qcap, ~, ~] = decode_solution(X, cap_sizes, n_bus, length(X)/2);
    [V, Ploss] = BFS_LoadFlow(from_bus, to_bus, R_pu, X_pu, ...
                               P_pu, Q_pu, Qcap, n_bus, n_branch);
    VDI  = sum((V - 1.0).^2);
    viol = sum(max(0, Vmin-V).^2 + max(0, V-Vmax).^2);
    f    = w1*Ploss*Sbase + w2*VDI*1000 + pen*viol;
end

%% ─────────────────────────────────────────────────────────────────────────

function [Qcap, cap_bus, cap_kvar] = decode_solution(X, cap_sizes, n_bus, n_cap)
%% Decode continuous TLBO variable to capacitor bus & size
    Qcap     = zeros(n_bus,1);
    cap_bus  = zeros(1,n_cap);
    cap_kvar = zeros(1,n_cap);
    for i = 1:n_cap
        bus  = max(1, min(n_bus-1, round(X(2*i-1))));
        sidx = max(1, min(length(cap_sizes), round(X(2*i))+1));
        sz   = cap_sizes(sidx);
        Qcap(bus)   = Qcap(bus) + sz/1000;   % p.u. (Sbase=1000 kVA)
        cap_bus(i)  = bus;
        cap_kvar(i) = sz;
    end
end
