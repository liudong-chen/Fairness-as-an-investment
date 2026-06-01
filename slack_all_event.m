%% slack_all_event.m
% Monte-Carlo fairness-profit evaluation -- EVENT-ONLY model.
%
%   Only hours that carry an event are used -- no zero hours, no quiet days.
%   Every period is a treated hour (an event); T = 225 events run
%   back-to-back, wall-clock gaps ignored. The state transitions ONCE per
%   event:  S_{i,t+1} = beta*S_{i,t} + rho*(x_{i,t}/Bmax_i), with rho the
%   per-consumer event-scale engagement from rho_event.csv.
%   Scenarios : Monte-Carlo pathways -- each a random permutation of the
%               225 events (the events occur in a different order).
%
% Figures: (1) profit vs fairness level, with an added curtailment band;
%          (2) histogram of consumer cost c_i; (3) histogram of initial
%          availability A_1; (4) histogram of engagement weight rho_i;
%          (5) market price and demand curve; (6) total availability under
%          benchmark / strict / slack (chronological deterministic at
%          ALPHA_PLOT, with extreme-event amber bands); (7) per-consumer
%          availability for 4 consumers, benchmark vs slack (chronological
%          deterministic at ALPHA_PLOT); (8) Theorem 2 event-pair
%          verification scatter (pi_{t+1} - pi*_t vs realised Delta Pi_t,
%          coloured by Case 1 / Case 2).
%
% Companion of the day-period slack_all.m, which is kept separately.
clc; clear;

%% ---------- 1. consumers: cost, capacity, event-scale rho ----------
load c_predict.mat
RT  = readtable('rho_event.csv','TextType','string');
ids = string(regI_keep.ID);
a_i = regI_keep.c_hat_int(:);
oID = string(outT_keep.ID);
Bmax = nan(numel(ids),1);
for i = 1:numel(ids)
    bi = outT_keep.Baseline_kWh(oID == ids(i));
    bi = bi(isfinite(bi) & bi > 0);
    if ~isempty(bi), Bmax(i) = max(bi); end
end
[tf,loc]   = ismember(ids, RT.ID);
rho_vec    = nan(numel(ids),1);
rho_vec(tf)= RT.rho_event(loc(tf));                    % <-- event-scale naive AR(1) rho (no noise correction)

keep = isfinite(a_i) & a_i > 0 & isfinite(Bmax) & Bmax > 0 ...
     & isfinite(rho_vec) & rho_vec > 0 & rho_vec <= 1;
a_i = a_i(keep);  Bmax = Bmax(keep);  rho_vec = rho_vec(keep);
N   = numel(a_i);
fprintf('Consumers kept: %d  (rho_event: median %.2f, range [%.2f, %.2f])\n', ...
        N, median(rho_vec), min(rho_vec), max(rho_vec));

%% ---------- 2. event-time market pool ----------
opts = detectImportOptions('price_demand_event.csv');
opts = setvartype(opts,'datetime_UTC','char');
PD   = readtable('price_demand_event.csv',opts);
ev   = PD(PD.is_event == 1, :);            % 225 events, chronological
price_pool  = ev.wholesale_EUR_per_MWh(:)/2;
demand_pool = ev.demand_kWh(:);
Tev = height(ev);
fprintf('Event pool: %d events  (price %.1f-%.1f, demand %.0f-%.0f kWh)\n', ...
        Tev, min(price_pool), max(price_pool), min(demand_pool), max(demand_pool));

%% ---------- 3. settings ----------
nMC        = 100;
equity_interval = 10;
alpha_grid = (0:equity_interval)'/equity_interval;   % <-- starts at 0 (fairness off)
nA         = numel(alpha_grid);
baseSeed   = 20260522;

P.kappa        = 6;
P.rho          = rho_vec;
P.Amin         = zeros(N,1);
P.lambda_slack = 1e5;
P.sigma_alpha  = 5e-4;
P.lp_opts      = optimoptions('linprog','Display','off','Algorithm','dual-simplex');

ALPHA_PLOT     = 0.8;                                   % fairness level for availability figures
nplot          = 4;                                     % consumers shown in Figure 7
sel_idx        = round(linspace(1, N, nplot));          % consumer indices to plot in Fig 7

%% ---------- 4. Monte-Carlo over event-order pathways ----------
prof_bench  = zeros(nMC,1);
prof_strict = zeros(nMC,nA);   curt_strict = zeros(nMC,nA);
prof_slack  = zeros(nMC,nA);   curt_slack  = zeros(nMC,nA);

fprintf('Running %d Monte-Carlo scenarios x %d fairness levels ...\n', nMC, nA);
tic;
parfor m = 1:nMC
    rng(baseSeed + m);
    perm  = randperm(Tev);                         % <-- different event order per scenario
    Dpath = demand_pool(perm);
    ppath = price_pool(perm);
    shock = P.sigma_alpha * randn(Tev,1);

    Rb = simulate_trajectory('benchmark', Dpath, ppath, shock, a_i, Bmax, 0, P);

    ps = zeros(1,nA); cs = zeros(1,nA);
    pl = zeros(1,nA); cl = zeros(1,nA);
    for k = 1:nA
        Rs = simulate_trajectory('strict', Dpath, ppath, shock, a_i, Bmax, alpha_grid(k), P);
        Rl = simulate_trajectory('slack',  Dpath, ppath, shock, a_i, Bmax, alpha_grid(k), P);
        ps(k) = Rs.profit;  cs(k) = Rs.curtail;
        pl(k) = Rl.profit;  cl(k) = Rl.curtail;
    end
    prof_bench(m)    = Rb.profit;
    prof_strict(m,:) = ps;  curt_strict(m,:) = cs;
    prof_slack(m,:)  = pl;  curt_slack(m,:)  = cl;
end
fprintf('Monte-Carlo done in %.1f s\n', toc);

%% ---------- 5. aggregate: mean and 10-90%% bands ----------
mb = mean(prof_bench);       qb = quantile(prof_bench,[0.10 0.90]);
ms = mean(prof_strict,1)';   sl = quantile(prof_strict,0.10,1)';  su = quantile(prof_strict,0.90,1)';
ml = mean(prof_slack ,1)';   ll = quantile(prof_slack ,0.10,1)';  lu = quantile(prof_slack ,0.90,1)';
mc_s = mean(curt_strict,1)'; cs_l = quantile(curt_strict,0.10,1)'; cs_u = quantile(curt_strict,0.90,1)';
mc_l = mean(curt_slack ,1)'; cl_l = quantile(curt_slack ,0.10,1)'; cl_u = quantile(curt_slack ,0.90,1)';
save('slackall_event_data.mat','alpha_grid','prof_bench','prof_strict', ...
     'prof_slack','curt_strict','curt_slack','nMC');

%% ---------- 5b. chronological deterministic trajectories at ALPHA_PLOT ----------
shock0 = zeros(Tev,1);
Rb_chr = simulate_trajectory('benchmark', demand_pool, price_pool, shock0, a_i, Bmax, 0,          P);
Rs_chr = simulate_trajectory('strict',    demand_pool, price_pool, shock0, a_i, Bmax, ALPHA_PLOT, P);
Rl_chr = simulate_trajectory('slack',     demand_pool, price_pool, shock0, a_i, Bmax, ALPHA_PLOT, P);
totA_b = sum(Rb_chr.A_store, 1);
totA_s = sum(Rs_chr.A_store, 1);
totA_l = sum(Rl_chr.A_store, 1);

% col_bench=[0.35 0.35 0.40]; col_strict=[0.20 0.45 0.70]; col_slack=[0.85 0.40 0.30];
% col_curt =[0.30 0.30 0.30];
col_bench   = [0.35 0.35 0.40];   % graphite — neutral, recedes
col_strict  = [0.20 0.45 0.70];   % deep blue — classic, anchored
col_slack   = [0.85 0.40 0.30];   % terracotta — warm, lively
col_curt = [0.55 0.70 0.55];   % sage green — distinct from the profit family

%% ---------- FIGURE 1 : profit vs fairness level (+ curtailment band) ----------
x  = 100*alpha_grid;  xb = [x; flipud(x)];
figure('Position',[100 100 700 350]); hold on;
yyaxis left;
fill(xb,[qb(1)*ones(nA,1);qb(2)*ones(nA,1)],col_bench ,'FaceAlpha',.12,'EdgeColor','none','HandleVisibility','off');
fill(xb,[sl;flipud(su)],col_strict,'FaceAlpha',.15,'EdgeColor','none','HandleVisibility','off');
fill(xb,[ll;flipud(lu)],col_slack ,'FaceAlpha',.15,'EdgeColor','none','HandleVisibility','off');
hB = plot(x, mb*ones(nA,1),'-','Color',col_bench ,'LineWidth',1.8);
hS = plot(x, ms,           '-' ,'Color',col_strict,'LineWidth',1.8);
hL = plot(x, ml,           '-' ,'Color',col_slack ,'LineWidth',1.8);
ylabel('Cumulative profit (NOK)');  set(gca,'YColor','k');
yyaxis right;
fill(xb,[cs_l;flipud(cs_u)],col_curt,'FaceAlpha',.18,'EdgeColor','none','HandleVisibility','off');
hC = plot(x, mc_s,'-','Color',col_curt,'LineWidth',1.9);
ylabel('Cumulative curtailment (kWh)');  set(gca,'YColor',col_curt);
xlabel('Fairness level (%)');
legend([hB hS hL hC], {'Benchmark','Strict fairness', ...
       'Slack fairness','Curtailment'}, 'Location','best','Box','off');
% title('Cumulative profit vs. fairness level  (with curtailment band)');
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event.png','Resolution',200);

%% ---------- FIGURE 2 : histogram of consumer cost c_i ----------
figure('Position',[100 100 700 350])
histogram(a_i, 30, 'FaceColor',col_strict,'EdgeColor','w');
xline(median(a_i),'--k','LineWidth',1.6);
xlabel('Cost (NOK/kWh)'); ylabel('Number of consumers');
% title(sprintf('Consumer cost  (N=%d, median %.2f)', N, median(a_i)));
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event_hist_cost.png','Resolution',200);

%% ---------- FIGURE 3 : histogram of initial availability A_1 ----------
figure('Position',[100 100 700 350])
histogram(Bmax, 30, 'FaceColor',[0.30 0.55 0.35],'EdgeColor','w');
xline(median(Bmax),'--k','LineWidth',1.6);
xlabel('Initial availability (kW)'); ylabel('Number of consumers');
ylim([0,30])
% title(sprintf('Initial availability  (N=%d, median %.2f)', N, median(Bmax)));
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event_hist_avail.png','Resolution',200);

%% ---------- FIGURE 4 : histogram of engagement weight rho_i ----------
figure('Position',[100 100 700 350])
histogram(rho_vec, 30, 'FaceColor',col_slack,'EdgeColor','w');
xline(median(rho_vec),'--k','LineWidth',1.6);
xlabel('Engagement weight'); ylabel('Number of consumers');
ylim([0,30])
% title(sprintf('Engagement weight  (N=%d, median %.2f)', N, median(rho_vec)));
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event_hist_rho.png','Resolution',200);

%% ---------- FIGURE 5 : market price and demand curve ----------
figure('Position',[100 100 700 350])
ei = (1:Tev)';
yyaxis left;
plot(ei, price_pool, '-', 'Color',col_strict, 'LineWidth',1.6);
ylabel('Scaled market price (NOK/kWh)'); set(gca,'YColor',col_strict);
yyaxis right;
plot(ei, demand_pool, '-', 'Color',col_slack, 'LineWidth',1.6);
ylabel('Required energy (kW)'); set(gca,'YColor',col_slack);
xlabel('Period (chronological order)'); xlim([1 Tev]);
legend('Price','Required energy','Location','best','box','off');
% title('Market price and demand across the 225 events');
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event_market.png','Resolution',200);

[rho, pval] = corr(price_pool(:), demand_pool(:));
fprintf('Correlation = %.4f\n', rho);
fprintf('p-value     = %.4e\n', pval);
%% ---------- FIGURE 6 : total availability, benchmark / strict / slack (chronological) ----------
% extreme events = top-decile required demand (scarcity-type events)
ex_thr = quantile(demand_pool, 0.90);
ex_idx = find(demand_pool(:) >= ex_thr);
col_ex = [0.93 0.82 0.42];                       % amber band for extreme events
ei = (1:Tev)';
figure('Position',[100 100 700 350]); hold on;
yl = [min([totA_b totA_s totA_l]) max([totA_b totA_s totA_l])];
yl = yl + [-0.02 0.02]*range(yl);                % small padding
for j = 1:numel(ex_idx)                          % vertical bands behind the curves
    patch([ex_idx(j)-0.5 ex_idx(j)+0.5 ex_idx(j)+0.5 ex_idx(j)-0.5], ...
          [yl(1) yl(1) yl(2) yl(2)], col_ex, ...
          'FaceAlpha',0.35,'EdgeColor','none','HandleVisibility','off');
end
hB6 = plot(ei, totA_b, '-','Color',col_bench ,'LineWidth',1.7);
hS6 = plot(ei, totA_s, '-','Color',col_strict,'LineWidth',1.7);
hL6 = plot(ei, totA_l, '-','Color',col_slack ,'LineWidth',1.7);
hX6 = patch(NaN,NaN,col_ex,'FaceAlpha',0.35,'EdgeColor','none');   % legend proxy
xlabel('Period (chronological order)'); ylabel('Total availability (kW)');
xlim([1 Tev]); ylim(yl); set(gca,'Layer','top');
legend([hB6 hS6 hL6 hX6], {'Benchmark','Strict fairness','Slack fairness', ...
       sprintf('Extreme event (top 10%% demand, n=%d)',numel(ex_idx))}, ...
       'Location','best','Box','off');
% title(sprintf('Total availability over the horizon  (\\alpha = %.1f)', ALPHA_PLOT));
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event_totavail.png','Resolution',200);
%% ---------- FIGURE 7 : per-consumer availability, chronological single trajectory ----------
ei7 = (1:Tev)';

C = [0.00 0.45 0.74;
     0.85 0.33 0.10;
     0.93 0.69 0.13;
     0.49 0.18 0.56];

figure('Position',[100 100 700 350]); hold on;
for i = 1:nplot
    plot(ei7, Rb_chr.A_store(sel_idx(i), :), '-',  'Color', C(i,:), 'LineWidth', 1.4, ...
         'DisplayName', sprintf('Consumer %d, b', sel_idx(i)));
    plot(ei7, Rl_chr.A_store(sel_idx(i), :), '--', 'Color', C(i,:), 'LineWidth', 1.4, ...
         'DisplayName', sprintf('Consumer %d, f', sel_idx(i)));
end

xlabel('Period (chronological order)'); ylabel('Availability (kW)');
xlim([1 Tev]);
ylim([0, 22])
legend('Location','eastoutside','FontSize',18,'Box','off');
set(gca,'FontSize',20,'LineWidth',1.1,'box','on','XColor',[0.15 0.15 0.15]);
grid on;
% exportgraphics(gcf,'slack_all_event_indavail.png','Resolution',200);

%% ---------- 8. Theorem 2 verification: event-pair price-quantity threshold ----------
% For each event pair (t, t+1) on the chronological trajectory at ALPHA_PLOT,
% classify the pair as Case 1 (supply-binding) or Case 2 (demand-binding),
% compute the threshold pi*_t from the appropriate case formula, and the
% realised event-pair profit gap dPi_t = (Pi^al_t + Pi^al_{t+1}) - (Pi^0_t + Pi^0_{t+1}).
% Theorem 2 predicts:
%   Case 1 (iff)        : sign(pi_{t+1} - pi*_t) == sign(dPi_t).
%   Case 2 (sufficient) : pi_{t+1} >= pi*_t  =>  dPi_t >= 0  (converse may fail).
c_max = max(a_i);
nPair = Tev - 1;
case_id  = zeros(nPair,1);     % 1, 2, or 0 (other)
pi_star  = nan(nPair,1);       % threshold price
dPi      = nan(nPair,1);       % realised profit gap
gap_diag = nan(nPair,1);       % pi_{t+1} - pi*_t

for t = 1:nPair
    A0_tp1  = Rb_chr.A_store(:,t+1);     % A^0_{i,t+1}
    Aal_tp1 = Rl_chr.A_store(:,t+1);     % A^{alpha,lambda}_{i,t+1}
    D0_t    = Rb_chr.x_store(:,t);       % D^0_{i,t}
    Dal_t   = Rl_chr.x_store(:,t);       % D^{alpha,lambda}_{i,t}
    Dbar    = demand_pool(t+1);          % bar D_{t+1}

    sumA0  = sum(A0_tp1);
    sumAal = sum(Aal_tp1);
    DA     = sumAal - sumA0;             % Delta A_{t+1}
    delta  = Dbar - sumA0;               % delta_{t+1}

    if DA <= 1e-9 || sumA0 < 1e-9, continue; end   % skip degenerate pairs

    C_re  = sum(a_i .* (Dal_t - D0_t));
    C_mix = sum(a_i .* (Aal_tp1 - A0_tp1));
    cbar0 = sum(a_i .* A0_tp1) / sumA0;

    dPi(t) = (Rl_chr.profit_t(t)   - Rb_chr.profit_t(t)) ...
           + (Rl_chr.profit_t(t+1) - Rb_chr.profit_t(t+1));

    if Dbar >= sumAal
        case_id(t) = 1;                  % supply-binding (iff)
        pi_star(t) = (C_re + C_mix) / DA;
    elseif Dbar > sumA0
        case_id(t) = 2;                  % demand-binding (sufficient)
        pi_star(t) = c_max + (C_re + (c_max - cbar0)*sumA0) / delta;
    else
        case_id(t) = 0;                  % no scarcity at t+1; theorem inactive
    end
    if case_id(t) > 0
        gap_diag(t) = price_pool(t+1) - pi_star(t);
    end
end

% --- headline statistics ---
isC1 = (case_id == 1);   nC1 = sum(isC1);
isC2 = (case_id == 2);   nC2 = sum(isC2);

c1_correct          = sum(isC1 & (sign(gap_diag) == sign(dPi)));
c1_viol_above       = sum(isC1 & gap_diag >= 0 & dPi <  0);
c1_viol_below       = sum(isC1 & gap_diag <  0 & dPi >= 0);

nC2_above           = sum(isC2 & gap_diag >= 0);
c2_correct_above    = sum(isC2 & gap_diag >= 0 & dPi >= 0);
c2_viol             = sum(isC2 & gap_diag >= 0 & dPi <  0);
nC2_below           = nC2 - nC2_above;
c2_below_pos        = sum(isC2 & gap_diag <  0 & dPi >= 0);   % below-thr but profitable: admissible, signals threshold looseness
c2_below_neg        = sum(isC2 & gap_diag <  0 & dPi <  0);   % below-thr and unprofitable: admissible, consistent

fprintf('\n=== Theorem 2 verification (chronological, alpha = %.2f) ===\n', ALPHA_PLOT);
fprintf('Event pairs : %d  (Case 1: %d, Case 2: %d, no scarcity: %d)\n', ...
        nPair, nC1, nC2, nPair - nC1 - nC2);
fprintf('Case 1 (iff)        : sign match = %d / %d   (viol above-thr & dPi<0 = %d; below-thr & dPi>=0 = %d)\n', ...
        c1_correct, nC1, c1_viol_above, c1_viol_below);
fprintf('Case 2 (sufficient) : violations (above-thr & dPi<0) = %d / %d total\n', ...
        c2_viol, nC2);
fprintf('                      [above-thr: %d (profit=%d, viol=%d);  below-thr (admissible): %d (profit=%d, loss=%d)]\n', ...
        nC2_above, c2_correct_above, c2_viol, nC2_below, c2_below_pos, c2_below_neg);

%% ---------- FIGURE 8 : Theorem 2 event-pair verification scatter ----------
figure('Position',[100 100 700 350]); hold on;
xline(0, '--k', 'LineWidth', 1.0, 'HandleVisibility','off');
yline(0, '--k', 'LineWidth', 1.0, 'HandleVisibility','off');
if nC1 > 0
    scatter(gap_diag(isC1), dPi(isC1), 162, col_strict, 'o', 'filled', ...
            'MarkerFaceAlpha', 0.75, 'DisplayName', sprintf('Case 1 event (n=%d)', nC1));
end
if nC2 > 0
    scatter(gap_diag(isC2), dPi(isC2), 162, col_slack,  '^', 'filled', ...
            'MarkerFaceAlpha', 0.75, 'DisplayName', sprintf('Case 2 event (n=%d)', nC2));
end
xlabel('Realized price over threshold (NOK/kWh)');
ylabel('Profit over benchmark (NOK)');
legend('Location','best','Box','off');
xlim([-1000, 200]); xticks(-1000:300:200);
ylim([-15000,60000])
grid on; set(gca,'FontSize',20);
% exportgraphics(gcf,'slack_all_event_thm2_verify.png','Resolution',200);

%% ---------- summary ----------
fprintf('\n=== summary (Monte-Carlo mean) ===\n');
fprintf('benchmark profit           : %.0f NOK\n', mb);
[bestL,iL]=max(ml); [bestS,iS]=max(ms);
fprintf('slack  fairness best profit : %.0f NOK at alpha=%.2f\n', bestL, alpha_grid(iL));
fprintf('strict fairness best profit : %.0f NOK at alpha=%.2f\n', bestS, alpha_grid(iS));
fprintf('Saved slackall_event_data.mat and 8 figures (slack_all_event*.png)\n');

%% ================================================================
%% local functions
%% ================================================================
function R = simulate_trajectory(mode, Dpath, ppath, shock, a_i, Bmax, alpha, P)
% Event-only simulation: one state transition per event.
N = numel(a_i);  T = numel(Dpath);
a_i = a_i(:);  Bmax = Bmax(:);
S = ones(N,1);
A = availability_from_state(S, Bmax, P.kappa, P.Amin, 0);
profit_t  = zeros(T,1);
curtail_t = zeros(T,1);
A_store   = zeros(N,T);
x_store   = zeros(N,T);   % NEW: per-event dispatch (needed for Theorem 2 verification)

for t = 1:T
    A_store(:,t) = A;                      % availability entering event t
    p = ppath(t);  D = Dpath(t);  ub = A;
    x_ref = solve_benchmark_greedy(p, a_i, ub, D);
    Dref  = sum(x_ref);
    switch mode
        case 'benchmark'
            x = x_ref;
        case {'strict','slack'}
            r_ref = x_ref ./ Bmax;
            Delta = max(r_ref) - min(r_ref);
            Rr    = (1 - alpha) * Delta;
            if Delta <= 1e-12
                x = x_ref;
            elseif strcmp(mode,'strict')
                x = solve_fair_lp_strict(p, a_i, ub, Bmax, Dref, Rr, P.lp_opts);
            else
                x = solve_fair_lp_slack(p, a_i, ub, Bmax, Dref, Rr, ...
                                        P.lambda_slack, P.lp_opts);
            end
    end
    x_store(:,t) = x;                                  % NEW
    profit_t(t)  = sum((p - a_i) .* x);
    curtail_t(t) = max(0, Dref - sum(x));
    S = (1 - P.rho).*S + P.rho.*(x ./ Bmax);          % per-event state update
    A = availability_from_state(S, Bmax, P.kappa, P.Amin, shock(t));
end
R.profit = sum(profit_t);  R.curtail = sum(curtail_t);  R.A_store = A_store;
R.profit_t = profit_t;     % NEW: per-event profit (for Theorem 2 verification)
R.x_store  = x_store;      % NEW: per-event dispatch (for Theorem 2 verification)
end

function A = availability_from_state(S, Bmax, kappa, Amin, shock)
z = min(max(S + shock, 0), 1);
A = Bmax .* (1 - exp(-kappa .* z));
A = max(Amin, min(Bmax, A));
A(~isfinite(A)) = 0;
end

function x = solve_benchmark_greedy(p, a, ub, D)
n = numel(a);  x = zeros(n,1);
if D <= 0 || p <= 0, return; end
ub = max(ub,0);
Dcap = min(D, sum(ub));
if Dcap <= 0, return; end
marg = p - a(:);
pos  = find(marg > 0);
if isempty(pos), return; end
[~,ord] = sort(marg(pos),'descend');
idx = pos(ord);
rem = Dcap;
for k = 1:numel(idx)
    xi = min(ub(idx(k)), rem);
    x(idx(k)) = xi;  rem = rem - xi;
    if rem <= 1e-12, break; end
end
end

function x = solve_fair_lp_strict(p, a, ub, Bmax, D, Rr, lp_opts)
n = numel(a);
ub = max(ub,0);  Bmax = max(Bmax,1e-12);
x = zeros(n,1);
if D <= 0 || p <= 0, return; end
c  = [-(p - a(:)); 0; 0];
A1 = [ones(1,n), 0, 0];                 b1 = D;
A2 = [eye(n), -Bmax(:), zeros(n,1)];    b2 = zeros(n,1);
A3 = [-eye(n), zeros(n,1), Bmax(:)];    b3 = zeros(n,1);
A4 = [zeros(1,n), 1, -1];               b4 = Rr;
[z,~,ef] = linprog(c,[A1;A2;A3;A4],[b1;b2;b3;b4],[],[], ...
                   [zeros(n,1);0;0], [ub;max(max(ub./Bmax),1e-6)*[1;1]], lp_opts);
if ef <= 0 || isempty(z)
    x0 = min(ub,(min(D,sum(ub))/n)*ones(n,1));
    sx = sum(x0); if sx>1e-12, x0 = x0*(min(D,sum(ub))/sx); end
    x = x0;
else
    x = z(1:n);
end
end

function x = solve_fair_lp_slack(p, a, ub, Bmax, D, Rr, lambda_slack, lp_opts)
n = numel(a);
ub = max(ub,0);  Bmax = max(Bmax,1e-12);
x = zeros(n,1);
if D <= 0 || p <= 0, return; end
D = min(D, sum(ub));
c   = [-(p - a(:)); 0; 0; lambda_slack];
Aeq = [ones(1,n), 0, 0, 0];                        beq = D;
A2  = [eye(n), -Bmax(:), zeros(n,1), zeros(n,1)];  b2  = zeros(n,1);
A3  = [-eye(n), zeros(n,1), Bmax(:), zeros(n,1)];  b3  = zeros(n,1);
A4  = [zeros(1,n), 1, -1, -1];                     b4  = Rr;
[z,~,ef] = linprog(c,[A2;A3;A4],[b2;b3;b4],Aeq,beq, ...
                   [zeros(n,1);0;0;0], [ub;max(max(ub./Bmax),1e-6)*[1;1];1], lp_opts);
if ef <= 0 || isempty(z)
    x = solve_benchmark_greedy(p, a, ub, D);
else
    x = z(1:n);
end
end
