clc
clear

participantsFile = "participants.csv";
hourlyFile       = "data_hourly.csv";
outMatFile       = "iflex_phase2_survey3.mat";

readSizeRows     = 250000;

%% ---------- 1) Read participants + filter Phase 2 & Survey 3 ----------
participantsAll = readtable(participantsFile, "TextType", "string");

% Strict: Phase_2 exactly
mask_strict = participantsAll.Control_Price_Phase2 == "Price group" & ...
              participantsAll.Survey3_answered == "Yes" & ...
              participantsAll.Participation_Phase == "Phase_2";

ids = participantsAll.ID(mask_strict);

participants = participantsAll(ismember(participantsAll.ID, ids), :);

%% ---------- 2) Stream-read data_hourly.csv and keep only selected IDs (Phase_2) ----------
ds = tabularTextDatastore(hourlyFile, ...
    "Delimiter", ",", ...
    "TextType", "string");

% Keep only columns we need (plus useful extras)
keepVars = ["ID","From","Date","Hour","Participation_Phase", ...
            "Demand_kWh","Price_signal","Experiment_price_NOK_kWh", ...
            "Temperature","Temperature24","Temperature48","Temperature72"];
ds.SelectedVariableNames = keepVars;

ds.ReadSize = readSizeRows;

Tkeep = table();
reset(ds);

while hasdata(ds)
    chunk = read(ds);

    % Phase 2 rows only
    chunk = chunk(chunk.Participation_Phase == "Phase_2", :);
    if isempty(chunk), continue; end

    % Selected IDs only
    chunk = chunk(ismember(chunk.ID, ids), :);
    if ~isempty(chunk)
        Tkeep = [Tkeep; chunk]; %#ok<AGROW>
    end
end

%% ---------- 3) Parse time + convert numerics ----------
% From column is hourly timestamp string like "yyyy-mm-dd HH:mm:ss"
Tkeep.Time = datetime(Tkeep.From, "InputFormat","yyyy-MM-dd'T'HH:mm:ssXXX", "TimeZone","UTC");
Tkeep.Day  = dateshift(Tkeep.Time, "start", "day");
Tkeep.Hour = double(Tkeep.Hour);

% Demand
Tkeep.Demand = double(Tkeep.Demand_kWh);

% Experiment price (may be "NA")
Tkeep.Price = str2double(replace(Tkeep.Experiment_price_NOK_kWh, "NA", ""));
if ~ismember("Price_signal", Tkeep.Properties.VariableNames)
    Tkeep.Price_signal = strings(height(Tkeep),1);
end

% Temps
Tkeep.Temp   = double(Tkeep.Temperature);
Tkeep.Temp24 = double(Tkeep.Temperature24);
Tkeep.Temp48 = double(Tkeep.Temperature48);
Tkeep.Temp72 = double(Tkeep.Temperature72);

%% ---------- 4) Compute baseline (hour-of-day mean of last 10 valid days) ----------
% Holidays: Norway public holidays for years in data (extend if needed)
holidayList = datetime([ ...
    "2020-11-22", ...
    "2020-12-25", ...
    "2020-12-26", ...
    "2021-01-01", ...
    "2021-04-01", ...
    "2021-04-02", ...
    "2021-04-04"  ...
], "InputFormat","yyyy-MM-dd", "TimeZone","UTC");

holidayList = dateshift(holidayList, "start", "day");

BaselineDemand = NaN(height(Tkeep), 1);

% Group by ID
[idsUnique, ~, gid] = unique(Tkeep.ID, "stable");

for k = 1:numel(idsUnique)
    idx = find(gid == k);
    Tk = Tkeep(idx, :);

    % Sort by time
    [~, ord] = sort(Tk.Time);
    Tk = Tk(ord, :);
    idx = idx(ord);

    % Unique days for this ID
    days = unique(Tk.Day, "stable");
    nDays = numel(days);

    % Map each row to day index
    [~, dayPos] = ismember(Tk.Day, days);

    % Demand matrix (day x hour)
    demandMat = NaN(nDays, 24);
    for r = 1:height(Tk)
        h = Tk.Hour(r);
        if h >= 1 && h <= 24
            demandMat(dayPos(r), h) = Tk.Demand(r);
        end
    end

    % Define experiment days for this ID:
    % any non-empty Price_signal OR any non-NaN experiment price on that day
    expDay = false(nDays, 1);
    for d = 1:nDays
        rowsD = (dayPos == d);
        anySignal = any(strlength(Tk.Price_signal(rowsD)) > 0);
        anyPrice  = any(~isnan(Tk.Price(rowsD)));
        expDay(d) = anySignal || anyPrice;
    end

    % Exclude weekends + holidays
    isWknd = ismember(weekday(days), [1 7]); % Sunday=1, Saturday=7
    isHol  = ismember(days, holidayList);

    validDay = ~expDay & ~isWknd & ~isHol;

    % Baseline matrix for this ID
    baseMat = NaN(nDays, 24);

    for d = 1:nDays
        prev = find(validDay(1:d-1));
        if isempty(prev), continue; end

        % last up to 10 valid days
        prev = prev(max(1, numel(prev)-9):end);

        baseMat(d, :) = mean(demandMat(prev, :), 1, "omitnan");
    end

    % Assign baseline back to hourly rows
    for r = 1:height(Tk)
        h = Tk.Hour(r);
        d = dayPos(r);
        if h >= 1 && h <= 24
            BaselineDemand(idx(r)) = baseMat(d, h);
        end
    end
end

%% ---------- 5) Build output table ----------
outT = table();
outT.ID   = Tkeep.ID;
outT.Time = Tkeep.Time;
outT.Date = Tkeep.Day;
outT.Hour = Tkeep.Hour;

% Main signals
outT.Price_NOK_kWh = Tkeep.Price;         % NaN on non-experiment hours
outT.Demand_kWh    = Tkeep.Demand;
outT.Baseline_kWh  = BaselineDemand;

% Useful extra info
outT.Price_signal  = Tkeep.Price_signal;
outT.Temperature   = Tkeep.Temp;
outT.Temperature24 = Tkeep.Temp24;
outT.Temperature48 = Tkeep.Temp48;
outT.Temperature72 = Tkeep.Temp72;

% save(outMatFile, "outT", "ids", "participants", "-v7.3");



