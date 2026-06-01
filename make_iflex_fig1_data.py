"""
make_iflex_fig1_data.py
-----------------------
Generate `data/iflex_dispatch_pattern_data.mat` (the input to plot_fig1.m)
directly from the raw iFlex hourly meter file.

Input  : data_hourly.csv      (download from Zenodo, see README; ~1.1 GB)
Output : data/iflex_dispatch_pattern_data.mat

Pipeline (self-contained; no intermediate .pkl, no scipy required):
  1. Stream the hourly file, keep Phase-2 households.
  2. response_t   = baseline demand (non-event mean by ID x weekday x hour)
                    minus metered demand  ->  "demand reduction".
  3. exposure_t   = rolling 168-hour sum of past response (engagement state).
  4. Sample the 800 households with the most high-incentive exposure.
  5. Within-household demean + temperature-residualize the response.
  6. Bin response vs. exposure by incentive-price quartile (and pooled),
     with 95% CI error bars, and fit per-group OLS slopes.
  7. Write everything to a MATLAB v5 .mat file plot_fig1.m can load.

Usage:  python make_iflex_fig1_data.py [path/to/data_hourly.csv]

This is the empirical counterpart of datapre.m / regre_linear.m: it works from
the same raw hourly file but targets the Figure-1 response-vs-exposure pattern
rather than the per-consumer cost thresholds.
"""
import os
import sys
import struct
import numpy as np
import pandas as pd

DATA    = sys.argv[1] if len(sys.argv) > 1 else "data_hourly.csv"
OUT_MAT = "iflex_dispatch_pattern_data.mat"

N_HOUSEHOLDS = 800     # sample size (paper: most-exposed households)
MIN_TREATED  = 20      # min high-incentive hours to be eligible
PRICE_TREAT  = 2.0     # NOK/kWh threshold that defines an "event" hour
WINDOW_H     = 168     # rolling exposure window (hours = one week)
N_BINS       = 16


# ----------------------------------------------------------------------
# 1. Build the treated-hour panel from the raw hourly file
# ----------------------------------------------------------------------
def build_panel(path):
    usecols = ["ID", "From", "Demand_kWh", "Experiment_price_NOK_kWh",
               "Temperature", "Participation_Phase"]
    parts = []
    for ch in pd.read_csv(path, usecols=usecols, chunksize=2_000_000,
                          low_memory=False):
        ch = ch[ch["Participation_Phase"] == "Phase_2"]
        if len(ch) == 0:
            continue
        ch = ch.drop(columns="Participation_Phase")
        ch["price"] = pd.to_numeric(ch["Experiment_price_NOK_kWh"],
                                    errors="coerce").fillna(1.0).astype("float32")
        ch["Demand_kWh"] = pd.to_numeric(ch["Demand_kWh"], errors="coerce").astype("float32")
        ch["Temperature"] = pd.to_numeric(ch["Temperature"], errors="coerce").astype("float32")
        ch = ch.drop(columns="Experiment_price_NOK_kWh")
        parts.append(ch)
    df = pd.concat(parts, ignore_index=True)

    df["From"] = pd.to_datetime(df["From"], utc=True)
    df["hour"] = df["From"].dt.hour.astype("int8")
    df["wday"] = df["From"].dt.weekday.astype("int8")
    df["is_treated"] = (df["price"] >= PRICE_TREAT).astype("int8")
    df = df.dropna(subset=["Demand_kWh"]).reset_index(drop=True)

    # Sample the most-exposed households (>= MIN_TREATED high-incentive hours)
    treated_counts = df.loc[df["is_treated"] == 1].groupby("ID").size()
    eligible = treated_counts[treated_counts >= MIN_TREATED]
    keep_ids = set(eligible.sort_values(ascending=False).head(N_HOUSEHOLDS).index)
    df = df[df["ID"].isin(keep_ids)].reset_index(drop=True)
    print(f"Sampled households: {len(keep_ids)}")

    # Baseline per (ID, weekday, hour) from non-event hours
    baseline = (df[df["is_treated"] == 0]
                .groupby(["ID", "wday", "hour"], observed=True)["Demand_kWh"]
                .mean().rename("baseline").reset_index())
    df = df.merge(baseline, on=["ID", "wday", "hour"], how="left")
    df = df.dropna(subset=["baseline"]).reset_index(drop=True)
    df["response"] = (df["baseline"] - df["Demand_kWh"]).astype("float32")
    df["dispatch"] = np.where(df["is_treated"] == 1,
                              np.maximum(df["response"], 0.0), 0.0).astype("float32")

    # Rolling 168-hour cumulative exposure (sum of past response), per household
    df = df.sort_values(["ID", "From"]).reset_index(drop=True)
    cum = np.zeros(len(df), dtype="float32")
    for ID, idx in df.groupby("ID", observed=True).indices.items():
        sub = df.iloc[idx][["From", "dispatch"]].set_index("From")
        cum[idx] = sub["dispatch"].rolling(f"{WINDOW_H}h", closed="left").sum().fillna(0.0).values
    df["cum168"] = cum

    panel = df.loc[df["is_treated"] == 1,
                   ["ID", "Temperature", "price", "response", "cum168"]].reset_index(drop=True)
    lo, hi = panel["response"].quantile([0.01, 0.99])           # winsorize tails
    panel = panel[(panel["response"] >= lo) & (panel["response"] <= hi)].reset_index(drop=True)
    print(f"Treated-hour panel rows: {len(panel):,}")
    return panel


# ----------------------------------------------------------------------
# 2. Bin response vs. exposure and fit slopes
# ----------------------------------------------------------------------
def binscatter(x, y, n_bins=N_BINS):
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return np.array([]), np.array([]), np.array([])
    cats = pd.cut(x, qs, include_lowest=True, duplicates="drop")
    g = pd.DataFrame({"x": x, "y": y, "c": cats}).groupby("c", observed=True)
    bx = g["x"].mean().values
    by = g["y"].mean().values
    se = (g["y"].std() / np.sqrt(g["y"].count())).values
    return bx, by, 1.96 * se


def ols_slope(x, y):
    n = len(x)
    b = np.cov(x, y, bias=False)[0, 1] / np.var(x, ddof=1)
    yhat = b * x + (y.mean() - b * x.mean())
    se = np.sqrt(np.sum((y - yhat) ** 2) / (n - 2)) / (x.std() * np.sqrt(n - 1))
    return b, se


def make_figure_data(p):
    p = p.dropna(subset=["Temperature"]).copy()
    # within-household demean and temperature-residualize the response
    p["resp_dm"] = p["response"] - p.groupby("ID", observed=True)["response"].transform("mean")
    p["temp_dm"] = p["Temperature"] - p.groupby("ID", observed=True)["Temperature"].transform("mean")
    b_temp = np.cov(p["temp_dm"], p["resp_dm"], bias=False)[0, 1] / np.var(p["temp_dm"], ddof=1)
    p["resp_resid"] = p["resp_dm"] - b_temp * p["temp_dm"]

    cap = p["cum168"].quantile(0.99)
    p["cum168_c"] = p["cum168"].clip(0, cap)

    labels = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
    p["pq"] = pd.qcut(p["price"], 4, labels=labels)
    color_map = {"Q1 (0-25%)": [0x3b, 0x75, 0xaf], "Q2 (25-50%)": [0x51, 0x9e, 0x8a],
                 "Q3 (50-75%)": [0xef, 0x8a, 0x3c], "Q4 (75-100%)": [0xd6, 0x40, 0x4e]}

    names, colors, X, Y, ERR, slopes, ses = [], [], [], [], [], [], []
    for q in labels:
        sub = p[p["pq"] == q]
        if len(sub) < 200:
            continue
        bx, by, err = binscatter(sub["cum168_c"].values, sub["resp_resid"].values)
        b, se = ols_slope(sub["cum168_c"].values, sub["resp_resid"].values)
        names.append(q); colors.append(np.array(color_map[q], float) / 255.0)
        X.append(bx); Y.append(by); ERR.append(err); slopes.append(b); ses.append(se)

    bx, by, err = binscatter(p["cum168_c"].values, p["resp_resid"].values)
    b, se = ols_slope(p["cum168_c"].values, p["resp_resid"].values)
    names.append("Pool (Q1-Q4)"); colors.append(np.array([0.0, 0.0, 0.0]))
    X.append(bx); Y.append(by); ERR.append(err); slopes.append(b); ses.append(se)
    return names, colors, X, Y, ERR, slopes, ses


# ----------------------------------------------------------------------
# 3. Minimal MAT v5 writer (2-D double + char matrices; no scipy needed)
# ----------------------------------------------------------------------
miINT8, miINT32, miUINT16, miUINT32, miDOUBLE, miMATRIX = 1, 5, 4, 6, 9, 14
mxCHAR_CLASS, mxDOUBLE_CLASS = 4, 6


def _pad8(b):
    return b + b"\x00" * ((8 - len(b) % 8) % 8)


def _element(dtype, payload):
    return _pad8(struct.pack("<II", dtype, len(payload)) + payload)


def _matrix(name, dims, cls, data_dtype, data_payload):
    sub = _element(miUINT32, struct.pack("<II", cls, 0))
    sub += _element(miINT32, struct.pack("<%di" % len(dims), *dims))
    sub += _element(miINT8, name.encode("ascii"))
    sub += _element(data_dtype, data_payload)
    return _element(miMATRIX, sub)


def _double_matrix(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return _matrix(name, list(arr.shape), mxDOUBLE_CLASS, miDOUBLE,
                   arr.astype("<f8").tobytes(order="F"))


def _char_matrix(name, strings):
    if isinstance(strings, str):
        strings = [strings]
    maxlen = max(len(s) for s in strings)
    mat = np.full((len(strings), maxlen), ord(" "), dtype=np.uint16)
    for i, s in enumerate(strings):
        for j, ch in enumerate(s):
            mat[i, j] = ord(ch)
    return _matrix(name, [len(strings), maxlen], mxCHAR_CLASS, miUINT16,
                   mat.astype("<u2").tobytes(order="F"))


def write_mat5(filename, variables):
    header = b"MATLAB 5.0 MAT-file, iflex fig1 export"
    header = header + b" " * (116 - len(header))
    header += struct.pack("<Q", 0) + struct.pack("<H", 0x0100) + b"IM"
    body = b""
    for name, val in variables.items():
        if isinstance(val, str) or (isinstance(val, list) and isinstance(val[0], str)):
            body += _char_matrix(name, val)
        else:
            body += _double_matrix(name, val)
    with open(filename, "wb") as f:
        f.write(header + body)


# ----------------------------------------------------------------------
def main():
    panel = build_panel(DATA)
    names, colors, X, Y, ERR, slopes, ses = make_figure_data(panel)

    G = len(names)
    B = max(len(v) for v in X)

    def pad(list_of_vecs):
        M = np.full((G, B), np.nan)
        for i, v in enumerate(list_of_vecs):
            M[i, :len(v)] = v
        return M

    os.makedirs(os.path.dirname(OUT_MAT), exist_ok=True)
    write_mat5(OUT_MAT, {
        "names":  names,
        "colors": np.vstack(colors),
        "x":      pad(X),
        "y":      pad(Y),
        "err":    pad(ERR),
        "slope":  np.asarray(slopes).reshape(-1, 1),
        "se":     np.asarray(ses).reshape(-1, 1),
        "n_obs":  np.array([[len(panel)]], float),
        "xlabel": "Rolling 168-hour sum of past response (kW)",
        "ylabel": "Demand reduction (kW)",
        "legend_title": "Incentive price quartile (NOK/kWh)",
    })
    print(f"Saved {OUT_MAT}  ({G} groups, up to {B} bins each)")
    for nm, b, se in zip(names, slopes, ses):
        print(f"  {nm:14s} slope={b:+.4f} SE={se:.4f}")


if __name__ == "__main__":
    main()
