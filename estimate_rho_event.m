%% estimate_rho_event.m
% Estimate the per-consumer event-scale engagement weight rho (and the
% persistence beta = 1 - rho) and write rho_event.csv, the file read by
% slack_all_event.m and pareto_event.m.
%
% Model (event-only): the engagement state follows an AR(1)
%       S_{i,t+1} = beta_i * S_{i,t} + rho_i * (x_{i,t}/Bmax_i),   beta_i + rho_i = 1,
% so rho_i is identified from the lag-1 autocorrelation of the per-consumer
% response series across consecutive events:
%
%   beta_event = OLS slope of resp_{t+1} on resp_t        (naive AR(1))
%   rho_event  = 1 - beta_event
%
% A noise-corrected variant removes attenuation from measurement error by
% taking the ratio of the lag-2 to lag-1 autocovariance of the
% price-residualized series (immune to white meter noise):
%
%   beta_event_nc = r2 / r1   (on residuals of resp ~ price)
%   rho_event_nc  = 1 - beta_event_nc
%
% Companion of the multi-method diagnostic estimate_rho.m; this script keeps
% only the event-scale naive AR(1) (used by the case study) plus its
% noise-corrected counterpart, and emits one row per case-study consumer.
%
% Input : c_predict.mat            (outT_keep, ids_keep  from regre_linear.m)
% Output: rho_event.csv
%   columns: ID, rho_event, beta_event, rho_event_raw, beta_event_raw,
%            rho_event_nc, beta_event_nc, n_pairs, flag_imputed, flag_clipped
clc; clear;

load c_predict.mat          % outT_keep, ids_keep

ids = ids_keep;
nH  = numel(ids);

MIN_PAIRS = 15;             % min consecutive event pairs to trust an estimate
AC_MIN    = 0.03;           % min lag-1 autocorr magnitude for noise correction

% lag-k autocorrelation of a vector
ac = @(x,k) (x(1+k:end)-mean(x))' * (x(1:end-k)-mean(x)) / sum((x-mean(x)).^2);

beta_raw   = nan(nH,1);  rho_raw   = nan(nH,1);
beta_nc    = nan(nH,1);  rho_nc    = nan(nH,1);
n_pairs    = zeros(nH,1);
valid_main = false(nH,1); valid_nc = false(nH,1);

for i = 1:nH
    Ti = outT_keep(outT_keep.ID == ids(i), :);
    [~,ord] = sort(Ti.Time);  Ti = Ti(ord,:);

    resp = Ti.Baseline_kWh - Ti.Demand_kWh;   % demand reduction = dispatch
    pr   = Ti.Price_NOK_kWh;                   % finite only on event hours
    ok   = isfinite(resp) & isfinite(pr);      % -> keep experiment events
    resp = resp(ok);  pr = pr(ok);

    n = numel(resp);
    n_pairs(i) = max(n-1, 0);
    if n < 3 || std(resp) < 1e-6, continue; end

    % ---- naive AR(1): resp_{t+1} = b0 + beta*resp_t ----
    y = resp(2:end);  x = resp(1:end-1);
    b1 = [ones(n-1,1) x] \ y;
    beta_raw(i) = b1(2);
    rho_raw(i)  = 1 - beta_raw(i);

    % ---- noise-corrected persistence r2/r1 on price-residualized series ----
    if std(pr) > 1e-9
        e = resp - [ones(n,1) pr] * ([ones(n,1) pr] \ resp);
    else
        e = resp - mean(resp);
    end
    r1 = ac(e,1);  r2 = ac(e,2);
    if r1 > AC_MIN && r2 > 0 && r2 < r1
        beta_nc(i) = r2 / r1;
        rho_nc(i)  = 1 - beta_nc(i);
        if beta_nc(i) >= 0 && beta_nc(i) <= 1
            valid_nc(i) = true;
        end
    end

    % an estimate is trustworthy if it has enough pairs and lands in [0,1]
    if n_pairs(i) >= MIN_PAIRS && beta_raw(i) >= 0 && beta_raw(i) <= 1
        valid_main(i) = true;
    end
end

% population medians used to impute untrustworthy consumers
med_rho     = median(rho_raw(valid_main));     med_beta    = 1 - med_rho;
med_rho_nc  = median(rho_nc(valid_nc));         med_beta_nc = 1 - med_rho_nc;

rho_event     = nan(nH,1);  beta_event     = nan(nH,1);
rho_event_nc  = nan(nH,1);  beta_event_nc  = nan(nH,1);
flag_imputed  = zeros(nH,1); flag_clipped  = zeros(nH,1);

for i = 1:nH
    % ---- main (naive AR(1)) ----
    if valid_main(i)
        r  = rho_raw(i);
        rc = min(max(r,0),1);                  % safety clip into [0,1]
        if abs(rc - r) > 1e-9, flag_clipped(i) = 1; end
        rho_event(i)  = rc;
        beta_event(i) = 1 - rc;
    else
        rho_event(i)   = med_rho;              % impute population median
        beta_event(i)  = med_beta;
        flag_imputed(i) = 1;
    end

    % ---- noise-corrected ----
    if valid_nc(i)
        rho_event_nc(i)  = rho_nc(i);
        beta_event_nc(i) = beta_nc(i);
    else
        rho_event_nc(i)  = med_rho_nc;
        beta_event_nc(i) = med_beta_nc;
    end
end

% raw columns keep the uncorrected estimates (NaN where uncomputable)
rho_event_raw  = rho_raw;
beta_event_raw = beta_raw;

T = table(string(ids(:)), rho_event, beta_event, rho_event_raw, beta_event_raw, ...
          rho_event_nc, beta_event_nc, n_pairs, flag_imputed, flag_clipped, ...
    'VariableNames', {'ID','rho_event','beta_event','rho_event_raw', ...
        'beta_event_raw','rho_event_nc','beta_event_nc','n_pairs', ...
        'flag_imputed','flag_clipped'});

writetable(T, 'rho_event.csv');

fprintf('Wrote rho_event.csv: %d consumers (%d imputed, %d clipped)\n', ...
        nH, sum(flag_imputed), sum(flag_clipped));
fprintf('rho_event: median %.3f  range [%.3f, %.3f]\n', ...
        median(rho_event), min(rho_event), max(rho_event));
