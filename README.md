This is to support paper "Fairness as an Investment: Dynamic Participation and Long-Run Profit in Virtual Power Plants"

# Code and Data

This folder contains the data and code needed to reproduce the figures of the
paper: the empirical Figure 1 (household response vs. cumulative event exposure)
and the numerical case-study figures (`slack_all_event.m`, `pareto_event.m`).

## 1. Data source

All household data derive from the public iFlex dynamic-pricing experiment
(Norway, winters 2020–2021):

> Hofmann, M., & Siebenbrunner, T. (2023). *A rich dataset of hourly residential
> electricity consumption data and survey answers from the iFlex dynamic pricing
> experiment.* Zenodo. https://doi.org/10.5281/zenodo.8248802 (CC BY 4.0)

### `data_hourly.csv` can be downloaded directly (~1.1 GB), require to run `datapre.m` or
`make_iflex_fig1_data.py`:

If you only want to **redraw the figures**, you do not need it — the derived data
files below are already provided.

Market price/load inputs in `price/` and `demand/` are day-ahead LBMP and load
forecasts (ENTSO NO1.), used by `priced.m` to build the market pool.


## 2. What is in this folder

### Plotting and optimizing scripts (the three figures)
| Script | Reads | Produces                                                                                                                                                |
|--------|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `plot_fig1.m` | `data/iflex_dispatch_pattern_data.mat` | Figure 1 — demand reduction vs. rolling 168-h past response, by price quartile + pooled, with OLS slopes.                                               |
| `slack_all_event.m` | `c_predict.mat`, `rho_event.csv`, `price_demand_event.csv` | Full 257-consumer Monte-Carlo case study (profit vs. fairness, parameter histograms, market panel, total & per-consumer availability, Theorem-2 check). |
| `pareto_event.m` | `c_predict.mat`, `rho_event.csv`, `price_demand_event.csv` | Fairness–profit Pareto front and its robustness to scaling ρ.                                                                                           |

### Data generation scripts (provided / already written)
| Script | Inputs | Output |
|--------|--------|--------|
| `datapre.m` | `participants.csv`, `data_hourly.csv` | `iflex_phase2_survey3.mat` — per-hour demand, baseline, price, temperature for Phase-2 / Survey-3 households. |
| `regre_linear.m` | `iflex_phase2_survey3.mat` | `c_predict.mat` — per-consumer threshold cost `c_i` from a per-ID logistic regression of dispatch on price (keeps significant, positively-sloped IDs). |
| `estimate_rho_event.m` | `c_predict.mat` | `rho_event.csv` — per-consumer event-scale engagement ρ (= 1 − β) from a per-ID lag-1 AR(1) of the response series, with a noise-corrected variant; out-of-range or low-sample consumers imputed to the median. |
| `priced.m` | `price/`, `demand/`, `outT_keep` (from `c_predict.mat`) | `price_demand.mat` and the event-time market pool `price_demand_event.csv` (prices in NOK/kWh, required demand scaled to the Norway system). |
| `make_iflex_fig1_data.py` | `data_hourly.csv` | `data/iflex_dispatch_pattern_data.mat` — the binned points, 95 % CI error bars, and per-group OLS slopes behind Figure 1. |

### Data files
| File | Role                                                                              |
|------|-----------------------------------------------------------------------------------|
| `iflex_phase2_survey3.mat` | Intermediate household panel (from `datapre.m`).                                  |
| `c_predict.mat` | Per-consumer cost `c_i` and baseline caps (`regI_keep`, `outT_keep`).             |
| `rho_event.csv` | Per-consumer event-scale engagement weights ρ (and β) — from `estimate_rho_event.m`. |
| `price_demand.mat`, `price_demand_event.csv` | Market price / required-demand pool (from `priced.m`).                            |
| `data/iflex_dispatch_pattern_data.mat` | Figure-1 plotting data (from `make_iflex_fig1_data.py`).                          |
| `participants.csv` | iFlex participant table.                                                    |
| `price/`, `demand/` | Raw ENTSO day-ahead price and load-forecast CSVs (market inputs).                 |


```bash
python make_iflex_fig1_data.py            # expects ./data_hourly.csv
# or
python make_iflex_fig1_data.py /path/to/data_hourly.csv
```

The reference build gives a pooled slope of **+0.0232 (SE 0.0003)** — response
rises with past exposure (engagement-building, not fatigue).


## 3. Reproduction steps

**Figures only (data already provided):**
- `plot_fig1.m` → Figure 1.
- `slack_all_event.m`, `pareto_event.m` → case-study figures.

**Full pipeline from raw data:**
1. Download `data_hourly.csv` into this folder (§1).
2. `datapre.m` → `iflex_phase2_survey3.mat`
3. `regre_linear.m` → `c_predict.mat`
4. `estimate_rho_event.m` → `rho_event.csv`
5. `priced.m` → `price_demand.mat`, `price_demand_event.csv`
6. `python make_iflex_fig1_data.py` → `data/iflex_dispatch_pattern_data.mat`
7. Run the three plotting scripts.


### Requirements
- **MATLAB** R2020a+ (uses `exportgraphics`); Statistics Toolbox for `fitglm` in `regre_linear.m`.
- **Python** 3.9+ with `numpy` and `pandas` (for `make_iflex_fig1_data.py`).


