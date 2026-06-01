%% pareto_event.m
% Event-only model -- full-scale fairness-profit Pareto front and its
% robustness to the engagement response rho.
%
% A 2x2 panel: each panel scales every consumer's identified rho_i by
% {10, 20, 50, 100}% and traces BOTH strict ("hard") and slack ("soft")
% fairness as a Pareto curve of profit gain over the profit-only benchmark
% vs the realized Gini index of cumulative dispatch. The square marks the
% benchmark (gain 0); open circles mark declared fairness alpha = 0.2,
% 0.5, 0.8. Mirrors pareto_rho.m in the Archive paper, but on the
% event-only horizon (T = 225 events, no zero hours), with rho from
% rho_event.csv.
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
[tf,loc]    = ismember(ids, RT.ID);
rho_vec     = nan(numel(ids),1);
rho_vec(tf) = RT.rho_event(loc(tf));                   % event-scale noise-corrected rho

keep = isfinite(a_i) & a_i > 0 & isfinite(Bmax) & Bmax > 0 ...
     & isfinite(rho_vec) & rho_vec > 0 & rho_vec <= 1;
a_i = a_i(keep);  Bmax = Bmax(keep);  rho_vec = rho_vec(keep);
N   = numel(a_i);

%% ---------- 2. event-time market pool ----------
opts = detectImportOptions('price_demand_event.csv');
opts = setvartype(opts,'datetime_UTC','char');
PD   = readtable('price_demand_event.csv',opts);
ev   = PD(PD.is_event == 1, :);
price_pool  = ev.wholesale_EUR_per_MWh(:)/2;
demand_pool = ev.demand_kWh(:);
Tev = height(ev);
fprintf('Event-only Pareto: N=%d consumers, T=%d events\n', N, Tev);

%% ---------- 3. settings ----------
nMC        = 100;                                       % Monte-Carlo event-order pathways
alpha_grid = [0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0]';   % includes 0.2/0.5/0.8
nA         = numel(alpha_grid);
scales     = [0.10 0.40 0.70 1.00];                    % rho robustness levels
nC         = numel(scales);
cfgLabel   = {'10% of identified \rho','40% of identified \rho', ...
              '70% of identified \rho','100% of identified \rho'};
baseSeed   = 20260522;

P.kappa        = 6;
P.Amin         = zeros(N,1);
P.lambda_slack = 1e5;
P.sigma_alpha  = 5e-4;
P.lp_opts      = optimoptions('linprog','Display','off','Algorithm','dual-simplex');

%% ---------- 4. Monte-Carlo sweep over event-order pathways ----------
prof_b = zeros(nMC,nC);          gini_b = zeros(nMC,nC);
prof_s = zeros(nMC,nC,nA);       gini_s = zeros(nMC,nC,nA);
prof_l = zeros(nMC,nC,nA);       gini_l = zeros(nMC,nC,nA);

fprintf('Running %d MC scenarios x %d rho-levels x %d fairness levels ...\n', nMC, nC, nA);
tic;
parfor m = 1:nMC
    rng(baseSeed + m);
    perm  = randperm(Tev);                             % different event order per scenario
    Dpath = demand_pool(perm);
    ppath = price_pool(perm);
    shock = P.sigma_alpha * randn(Tev,1);

    pb = zeros(1,nC);   gb = zeros(1,nC);
    ps = zeros(nC,nA);  gs = zeros(nC,nA);
    pl = zeros(nC,nA);  gl = zeros(nC,nA);
    for c = 1:nC
        Pc = P;  Pc.rho = min(scales(c)*rho_vec, 1);   % scaled rho, capped at 1
        Rb = simulate_trajectory('benchmark', Dpath, ppath, shock, a_i, Bmax, 0, Pc);
        pb(c) = Rb.profit;  gb(c) = compute_gini(Rb.D_cum);
        for k = 1:nA
            Rs = simulate_trajectory('strict', Dpath, ppath, shock, a_i, Bmax, alpha_grid(k), Pc);
            Rl = simulate_trajectory('slack',  Dpath, ppath, shock, a_i, Bmax, alpha_grid(k), Pc);
            ps(c,k) = Rs.profit;  gs(c,k) = compute_gini(Rs.D_cum);
            pl(c,k) = Rl.profit;  gl(c,k) = compute_gini(Rl.D_cum);
        end
    end
    prof_b(m,:)   = pb;                   gini_b(m,:)   = gb;
    prof_s(m,:,:) = reshape(ps,1,nC,nA);  gini_s(m,:,:) = reshape(gs,1,nC,nA);
    prof_l(m,:,:) = reshape(pl,1,nC,nA);  gini_l(m,:,:) = reshape(gl,1,nC,nA);
end
fprintf('done in %.1f s\n', toc);

%% ---------- 5. aggregate (Monte-Carlo mean) ----------
profB = mean(prof_b,1);                   % 1 x nC
giniB = mean(gini_b,1);
profS = squeeze(mean(prof_s,1));          % nC x nA
giniS = squeeze(mean(gini_s,1));
profL = squeeze(mean(prof_l,1));
giniL = squeeze(mean(gini_l,1));
gainS = (profS - profB') / 1e6;           % profit gain over benchmark, million NOK
gainL = (profL - profB') / 1e6;
save('pareto_event_data.mat','alpha_grid','scales','profB','giniB', ...
     'profS','giniS','profL','giniL','gainS','gainL','nMC');

%% ---------- 6. FIGURE : 2x2 Pareto front (profit gain vs Gini) ----------
col_strict = [0.20 0.45 0.70];
col_slack  = [0.85 0.40 0.30];
allx = [giniB(:); giniS(:); giniL(:)];
ally = [0; gainS(:); gainL(:)];
xl = [min(allx)-0.03, max(allx)+0.03];
yl = [min(ally)-0.10*range(ally), max(ally)+0.16*range(ally)];
FS = 16;

fig = figure;
tl  = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for c = 1:nC
    ax = nexttile; hold(ax,'on');
    plot(ax, xl, [0 0], '-', 'Color',[0.6 0.6 0.6], 'LineWidth',1.5, ...
         'HandleVisibility','off');
    plot(ax, giniS(c,:), gainS(c,:), '--o', 'Color',col_strict, ...
         'LineWidth',1.6, 'MarkerSize',3, 'MarkerFaceColor',col_strict);
    plot(ax, giniL(c,:), gainL(c,:), '-o',  'Color',col_slack, ...
         'LineWidth',1.6, 'MarkerSize',3, 'MarkerFaceColor',col_slack);
    plot(ax, giniB(c), 0, 'ks', 'MarkerSize',7, 'MarkerFaceColor','w', ...
         'LineWidth',1.1, 'HandleVisibility','off');
    for a0 = [0.2 0.5 0.8]
        [~,j] = min(abs(alpha_grid - a0));
        plot(ax, giniS(c,j), gainS(c,j), 'o', 'MarkerSize',7, ...
             'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',1.0, ...
             'HandleVisibility','off');
        plot(ax, giniL(c,j), gainL(c,j), 'o', 'MarkerSize',7, ...
             'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',1.0, ...
             'HandleVisibility','off');
        text(ax, giniL(c,j), gainL(c,j)+0.07*range(yl), sprintf('\\alpha=%.1f',a0), ...
             'HorizontalAlignment','center','FontSize',FS-1,'Color',[0.15 0.15 0.15]);
    end
    title(ax, cfgLabel{c}, 'FontWeight','normal','FontSize',FS);
    xlim(ax,xl); ylim(ax,yl); grid(ax,'on'); ax.GridAlpha = 0.15; box(ax,'on');
    set(ax,'FontSize',FS);
    if c==1
        legend(ax,{'Strict fairness','Slack fairness'}, ...
               'Location','northeast','FontSize',FS,'Box','off');
    end
end
xlabel(tl,'Realized Gini index','FontSize',FS);
ylabel(tl,'Profit gain over benchmark (million NOK)','FontSize',FS);
% title(tl,'Event-only fairness-profit Pareto front and robustness to \rho','FontSize',FS+1);
% exportgraphics(fig,'pareto_event.png','Resolution',300);

%% Figure - real use!!!!!
fig = figure;
tl  = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

for c = 1:nC
    ax = nexttile; 
    hold(ax,'on');

    plot(ax, xl, [0 0], '-', 'Color',[0.6 0.6 0.6], ...
         'LineWidth',1.5,'HandleVisibility','off');

    plot(ax, giniS(c,:), gainS(c,:), '--o', ...
         'Color',col_strict, ...
         'LineWidth',1.6, ...
         'MarkerSize',3, ...
         'MarkerFaceColor',col_strict);

    plot(ax, giniL(c,:), gainL(c,:), '-o', ...
         'Color',col_slack, ...
         'LineWidth',1.6, ...
         'MarkerSize',3, ...
         'MarkerFaceColor',col_slack);

    plot(ax, giniB(c), 0, 'ks', ...
         'MarkerSize',7, ...
         'MarkerFaceColor','w', ...
         'LineWidth',1.1, ...
         'HandleVisibility','off');

    for a0 = [0.2 0.5 0.8]
        [~,j] = min(abs(alpha_grid-a0));

        plot(ax, giniS(c,j), gainS(c,j), 'o', ...
             'MarkerSize',7, ...
             'MarkerEdgeColor','k', ...
             'MarkerFaceColor','none', ...
             'LineWidth',1.0, ...
             'HandleVisibility','off');

        plot(ax, giniL(c,j), gainL(c,j), 'o', ...
             'MarkerSize',7, ...
             'MarkerEdgeColor','k', ...
             'MarkerFaceColor','none', ...
             'LineWidth',1.0, ...
             'HandleVisibility','off');

        text(ax, giniL(c,j)-0.01*range(yl), gainL(c,j)+0.08*range(yl), ...
             sprintf('\\alpha=%.1f',a0), ...
             'HorizontalAlignment','center', ...
             'FontSize',FS-2, ...
             'Color',[0.15 0.15 0.15]);
    end

    title(ax, cfgLabel{c}, ...
          'FontWeight','normal' ,'FontSize', 11 ...
          );

    xlim(ax,xl);
    ylim(ax,yl);

    grid(ax,'on');
    box(ax,'on');
    ax.GridAlpha = 0.15;

    set(ax,'FontSize',FS);

    % remove inner tick labels
    if ~ismember(c,[3 4])   % top row
        ax.XTickLabel = [];
    end

    if ~ismember(c,[1 3])   % right column
        ax.YTickLabel = [];
    end

    if c==1
        legend(ax,{'Strict fairness','Slack fairness'}, ...
               'Location','northwest', ...
               'FontSize',FS, ...
               'Box','off');
    end
end

% single shared labels
xlabel(tl,'Realized Gini index','FontSize',FS+1);
ylabel(tl,'Profit gain over benchmark (million NOK)','FontSize',FS+1);
% col_strict = [0.20 0.45 0.70];
% col_slack  = [0.85 0.40 0.30];
% allx = [giniB(:); giniS(:); giniL(:)];
% ally = [0; gainS(:); gainL(:)];
% xl = [min(allx)-0.03, max(allx)+0.03];
% yl = [min(ally)-0.10*range(ally), max(ally)+0.16*range(ally)];
% FS = 18;
% for c = 1:nC
%     figure
%     ax = nexttile; hold(ax,'on');
%     plot(ax, xl, [0 0], '-', 'Color',[0.6 0.6 0.6], 'LineWidth',1.7, ...
%          'HandleVisibility','off');
%     plot(ax, giniS(c,:), gainS(c,:), '-o', 'Color',col_strict, ...
%          'LineWidth',1.7, 'MarkerSize',3, 'MarkerFaceColor',col_strict);
%     plot(ax, giniL(c,:), gainL(c,:), '-o',  'Color',col_slack, ...
%          'LineWidth',1.7, 'MarkerSize',3, 'MarkerFaceColor',col_slack);
%     plot(ax, giniB(c), 0, 'ks', 'MarkerSize',7, 'MarkerFaceColor','w', ...
%          'LineWidth',1.7, 'HandleVisibility','off');
%     for a0 = [0.2 0.5 0.8]
%         [~,j] = min(abs(alpha_grid - a0));
%         plot(ax, giniS(c,j), gainS(c,j), 'o', 'MarkerSize',7, ...
%              'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',1.0, ...
%              'HandleVisibility','off');
%         plot(ax, giniL(c,j), gainL(c,j), 'o', 'MarkerSize',7, ...
%              'MarkerEdgeColor','k','MarkerFaceColor','none','LineWidth',1.0, ...
%              'HandleVisibility','off');
%         text(ax, giniL(c,j), gainL(c,j)+0.07*range(yl), sprintf('\\alpha=%.1f',a0), ...
%              'HorizontalAlignment','center','FontSize',FS-1,'Color',[0.15 0.15 0.15]);
%     end
%     title(ax, cfgLabel{c}, 'FontWeight','normal','FontSize',FS);
%     xlim(ax,xl); ylim(ax,yl); grid(ax,'on'); ax.GridAlpha = 0.15; box(ax,'on');
%     xlabel('Realized Gini index')
%     ylabel('Profit gain over benchmark (million NOK)')
%     set(ax,'FontSize',FS);
%     if c==1
%         legend(ax,{'Strict fairness','Slack fairness'}, ...
%                'Location','northeast','FontSize',FS-1,'Box','off');
%     end
% end

%% ---------- 7. summary ----------
fprintf('\n=== event-only Pareto (N=%d, T=%d, nMC=%d) ===\n', N, Tev, nMC);
for c = 1:nC
    fprintf('%-22s  benchGini=%.3f | strict gain [%6.3f,%6.3f] M | slack gain [%6.3f,%6.3f] M NOK\n', ...
        cfgLabel{c}, giniB(c), min(gainS(c,:)), max(gainS(c,:)), ...
        min(gainL(c,:)), max(gainL(c,:)));
end
fprintf('Saved pareto_event_data.mat and pareto_event.png\n');

%% ================================================================
%% local functions
%% ================================================================
function R = simulate_trajectory(mode, Dpath, ppath, shock, a_i, Bmax, alpha, P)
% Event-only simulation: one state transition per event. Returns cumulative
% profit and the per-consumer cumulative dispatch (for the Gini index).
N = numel(a_i);  T = numel(Dpath);
a_i = a_i(:);  Bmax = Bmax(:);
S = ones(N,1);
A = availability_from_state(S, Bmax, P.kappa, P.Amin, 0);
profit_t = zeros(T,1);
D_cum    = zeros(N,1);

for t = 1:T
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
    profit_t(t) = sum((p - a_i) .* x);
    D_cum       = D_cum + x;
    S = (1 - P.rho).*S + P.rho.*(x ./ Bmax);
    A = availability_from_state(S, Bmax, P.kappa, P.Amin, shock(t));
end
R.profit = sum(profit_t);
R.D_cum  = D_cum;
end

function g = compute_gini(x)
% Gini index of a nonnegative allocation vector (0 = equal, ->1 = concentrated).
x = x(:);  n = numel(x);  s = sum(x);
if s <= 0, g = 0; return; end
xs = sort(x);  idx = (1:n)';
g = (2*sum(idx.*xs) - (n+1)*s) / (n*s);
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
