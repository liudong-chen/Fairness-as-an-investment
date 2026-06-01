clc
clear
load iflex_phase2_survey3.mat
T = outT;
%% ---------- Experimental HOURS filter (need numeric price for regression) ----------
isExpHour = ~isnan(T.Price_NOK_kWh) | (strlength(string(T.Price_signal)) > 0);
Texp = T(isExpHour, :);
% Require numeric price, baseline, demand
Texp = Texp(~isnan(Texp.Price_NOK_kWh), :);
Texp = Texp(~isnan(Texp.Baseline_kWh) & ~isnan(Texp.Demand_kWh), :);
% Dispatch indicator: 1 if consumer reduced demand below baseline, 0 otherwise
% Under linear cost, the threshold rule says D > 0 iff price > c_i.
% In iFlex, "dispatched" = demand reduction = (Baseline - Demand) > 0.
Texp.deltaD = Texp.Baseline_kWh - Texp.Demand_kWh;
Texp.dispatched = double(Texp.deltaD > 0);
% Remove bad rows
Texp = Texp(isfinite(Texp.deltaD) & isfinite(Texp.Price_NOK_kWh), :);
%% ---------- Group by ID ----------
minObs = 30;   % per ID: require at least this many experimental observations
[G, idG] = findgroups(Texp.ID);
nIDs = max(G);
regI = table( ...
    idG, ...
    NaN(nIDs,1), NaN(nIDs,1), NaN(nIDs,1), ...  % with intercept (logistic)
    NaN(nIDs,1), NaN(nIDs,1), NaN(nIDs,1), ...  % slope, intercept-coeff, dispatch rate
    NaN(nIDs,1), NaN(nIDs,1), ...
'VariableNames', {'ID', ...
'c_hat_int','p_int','R2_int', ...
'b_slope','a_intercept','dispatch_rate', ...
'N','price_std'} );
for g = 1:nIDs
    idx = (G == g);
    Ti = Texp(idx, :);
    n = height(Ti);
    regI.N(g) = n;
if n < minObs
continue;
end
    s = std(Ti.Price_NOK_kWh, "omitnan");
    regI.price_std(g) = s;
if s < 1e-9
continue; % no price variation => threshold unidentified
end
    % Skip if dispatch is all 0 or all 1 (no variation in outcome)
    dispatchRate = mean(Ti.dispatched, "omitnan");
    regI.dispatch_rate(g) = dispatchRate;
if dispatchRate < 0.02 || dispatchRate > 0.98
continue; % degenerate outcome => threshold unidentified
end
    x = Ti.Price_NOK_kWh;
    y = Ti.dispatched;
% ---- Logistic regression: P(dispatched | price) = 1 / (1 + exp(-(a + b*price))) ----
try
        mdl = fitglm(x, y, 'Distribution', 'binomial', 'Link', 'logit');
        a_hat = mdl.Coefficients.Estimate(1);
        b_hat = mdl.Coefficients.Estimate(2);
        p_b   = mdl.Coefficients.pValue(2);
        regI.a_intercept(g) = a_hat;
        regI.b_slope(g)     = b_hat;
        regI.p_int(g)       = p_b;
        regI.R2_int(g)      = 1 - mdl.Deviance / mdl.SST; % McFadden-like pseudo-R2
        % c_hat = -a/b is the price at which dispatch probability = 0.5
        if b_hat > 0
            regI.c_hat_int(g) = -a_hat / b_hat;
        end
catch
        % fitglm may fail for ill-conditioned cases; leave as NaN
continue;
end
end
% Keep IDs with successful estimates
regI_valid_int   = regI(~isnan(regI.c_hat_int), :);
figure;
tiledlayout(2,1,"TileSpacing","compact","Padding","compact");
nexttile;
histogram(regI_valid_int.p_int, 50);
grid on;
xlabel("p-value of slope b_i (logistic)");
ylabel("Number of IDs");
title("Per-ID logistic regression: p-values of slope");
nexttile;
histogram(regI_valid_int.c_hat_int, 50);
grid on;
xlabel("Estimated threshold cost c_i = -a_i/b_i (NOK/kWh)");
ylabel("Number of IDs");
title("Per-ID estimated cost thresholds");
%% ============================================================
% Filter IDs: logistic regression, keep p < 0.05, positive slope, and positive c
% Save another file
alpha = 0.05;
% Make sure we only use IDs with valid estimates
validInt = ~isnan(regI.c_hat_int) & ~isnan(regI.p_int);
keepMask = validInt & (regI.p_int < alpha) & (regI.b_slope > 0) & (regI.c_hat_int > 0);
regI_keep = regI(keepMask, :);
ids_keep  = regI_keep.ID;
outT_keep = outT(ismember(outT.ID, ids_keep), :);
outT_keep = outT_keep(~isnan(outT_keep.Baseline_kWh),:);

save("c_predict.mat", ...
"outT_keep", "ids_keep", "regI_keep", "alpha", "-v7.3");