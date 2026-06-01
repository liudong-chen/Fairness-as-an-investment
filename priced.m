clc
clear

priceDir = "./price";   % <- change
loadDir  = "./demand";% <- change

startDate = datetime(2025,1,1);
endDate   = datetime(2025,12,31);

timeFmt = "MM/dd/yyyy HH:mm";           % your confirmed format

getFileDate = @(fn) datetime(extractBetween(string(fn), 1, 8), "InputFormat","yyyyMMdd");

%% ============================================================
% 1) Read Day-Ahead zonal LBMP (Zone J = "N.Y.C.")
priceFiles = dir(fullfile(priceDir, "*damlbmp_zone.csv"));
assert(~isempty(priceFiles), "No price files found: *damlbmp_zone.csv");

priceAll = table();

for k = 1:numel(priceFiles)
    fn = fullfile(priceFiles(k).folder, priceFiles(k).name);
    fdate = getFileDate(priceFiles(k).name);

    if fdate < startDate || fdate > endDate
        continue;
    end

    T = readtable(fn, "TextType","string");

    % Filter NYC zone
    T = T(T.Name == "N.Y.C.", :);
    if isempty(T), continue; end

    % Parse time
    tt = datetime(T.("TimeStamp"), "InputFormat", timeFmt);

    % Find LBMP column
    v = string(T.Properties.VariableNames);
    lmpVar = v(contains(v, "LBMP", "IgnoreCase", true));
    assert(~isempty(lmpVar), "No LBMP column in %s", fn);
    lmp = double(T.(lmpVar(1)));

    priceAll = [priceAll; table(tt, lmp, repmat(fdate,height(T),1), ...
        'VariableNames', {'Time','DA_LBMP_$/MWh','FileDate'})]; %#ok<AGROW>
end

assert(~isempty(priceAll), "No DA price rows collected. Check folder/columns.");

priceAll = priceAll(priceAll.Time >= startDate & priceAll.Time <= endDate + days(1), :);
priceAll = sortrows(priceAll, "Time");

priceTT = table2timetable(priceAll(:,{'Time','DA_LBMP_$/MWh'}), "RowTimes","Time");
priceTT = retime(priceTT, "hourly", "mean");


%% =========================
% 2) Zonal load forecast (NYC, keep same-day only)

fcstFiles = dir(fullfile(loadDir, "*isolf.csv"));
assert(~isempty(fcstFiles), "No forecast files found: *isolf.csv (adjust pattern)");

fcstAll = table();

for k = 1:numel(fcstFiles)
    fn = fullfile(fcstFiles(k).folder, fcstFiles(k).name);
    fdate = getFileDate(fcstFiles(k).name);

    if fdate < startDate || fdate > endDate
        continue;
    end

    T = readtable(fn, "TextType","string");

    % Parse time
    tt = datetime(T.("TimeStamp"), "InputFormat", timeFmt);

    % NYC column
    assert(any(strcmp(T.Properties.VariableNames,"N_Y_C_")), "No N_Y_C_ column in %s", fn);
    nycMW = double(T.("N_Y_C_"));

    % Keep same-day forecast only
    isSameDay = dateshift(tt,"start","day") == dateshift(fdate,"start","day");

    tt = tt(isSameDay);
    nycMW = nycMW(isSameDay);
    if isempty(tt), continue; end

    fcstAll = [fcstAll; table(tt, nycMW, repmat(fdate,numel(tt),1), ...
        'VariableNames', {'Time','NYC_LoadFcst_MW','FileDate'})]; %#ok<AGROW>
end

fcstAll = fcstAll(fcstAll.Time >= startDate & fcstAll.Time <= endDate + days(1), :);
fcstAll = sortrows(fcstAll, "Time");

fcstTT = table2timetable(fcstAll, "RowTimes","Time");
fcstTT = retime(fcstTT, "hourly", "mean");

nyisoTT = synchronize(priceTT, fcstTT, "union");
nyisoTT = sortrows(nyisoTT);

save("price_demand.mat", "nyisoTT", "-v7.3");

%% scale demand to fit the norway data
% If Time is datetime -> use datenum; if numeric already -> use as-is.
if isdatetime(outT_keep.Time)
    key = datenum(outT_keep.Time);
else
    key = outT_keep.Time;
end

% Sum baseline across the 239 IDs for each hour
[G, ~] = findgroups(key);
S = splitapply(@(x) sum(x,"omitnan"), outT_keep.Baseline_kWh, G);
S = S(:);

% Drop NaNs / nonpositive (optional)
S = S(isfinite(S) & S > 0);

nyisoTT.Properties.VariableNames = matlab.lang.makeValidName(nyisoTT.Properties.VariableNames);

L = nyisoTT.NYC_LoadFcst_MW * 1000;
L = L(isfinite(L) & L > 0);

price = nyisoTT.DA_LBMP___MWh;
price = price(isfinite(L) & L > 0);

qN = 0.95;      % "normal high" quantile
qE = 0.99;      % extreme quantile
theta = 0.90;   % ensure normal-high required < available
phi   = 1.15;   % ensure extreme required > available

QL_N = quantile(L, qN);
QL_E = quantile(L, qE);
QS_N = quantile(S, qN);
QS_E = quantile(S, qE);

% Solve a + b L mapping
b = (phi*QS_E - theta*QS_N) / (QL_E - QL_N);
a = theta*QS_N - b*QL_N;

Dreq = max(0, a + b*L);  % required service in kWh

share_regular_feasible = mean(Dreq <= QS_E);  % rough: compare to high-capacity quantile
fprintf("Share Dreq <= QS_E (rough regular feasibility): %.3f\n", share_regular_feasible);

