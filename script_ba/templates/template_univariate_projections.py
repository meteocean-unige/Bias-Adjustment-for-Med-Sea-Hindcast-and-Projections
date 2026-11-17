#!/usr/bin/env python
# coding: utf-8
"""
Seasonal quantile-mapping bias adjustment of climate-projection wave data
(e.g. EURO-CORDEX driven wave models: historical/baseline + future projections)

The reference ("obs") is the OUTPUT of bias_adjustment_hindcast-renalysis.py, i.e. the
hindcast already corrected against buoy data (file *_bias_adj_seasonal.txt,
columns: datetime, <VAR>_ba).

Method:
  1. For each season the data are split at the percentile `THRESH`:
       - lower/mid part  (values below the threshold)
       - upper tail      (values above the threshold)
     The future run is split at the threshold of the HISTORICAL run.
  2. Reference (obs) and model historical/baseline: for each part the
     best-fitting distribution is chosen with `fitter`
     (lower/mid: DIST_MID, upper tail: DIST_UPPER).
  3. Both historical and future model values are mapped with the
     distribution fitted on the HISTORICAL period:
         x_adj = F_obs^-1( F_mod_hist(x) )
     so the climate-change signal of the future scenario is preserved.
  4. Corrected historical and future series are saved, with CDF plots.

Terminology
  obs : reference data used to correct the models
  mod : model data to be corrected (historical/baseline and future projections)

Requirements
  pip install -r requirements.txt

"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter

warnings.filterwarnings("ignore")


# ========================================================================
# USER SETTINGS  -- edit this section to adapt the script to your data
# ========================================================================

# --- Folders ------------------------------------------------------------
DIR_DATA = Path("data_projection")          # folder with reference and model files
DIR_OUT = DIR_DATA / "bias_adjusted_outputs"  # corrected series are written here
DIR_PICS = DIR_OUT / "pics"                  # figures are written here
# (output folders are created if missing)

# --- Site name (used in the output file names) --------------------------
SITE_NAME = "STATION_NAME"

# --- Variable name ------------------------------------------------------
VAR = "hs"   # column of the model files to correct; output column: f"{VAR}_ba"

# --- Reference data (bias-adjusted hindcast, output of bias_adjustment.py)
OBS_FILE = DIR_DATA / "output_file.txt" #OUTPUT of bias_adjustment_hindcast-renalysis.py,
OBS_SEP = ","
OBS_TIME_COL = 0         # time column (name or position, from 0)
OBS_VAR_COL = f"{VAR}_ba"  # column with the corrected hindcast (name or position)

# Period of the reference used for the calibration (years, inclusive).
# It should overlap the historical period of the models.
CALIB_START = 1979
CALIB_END = 2005

# --- Model data (data to be CORRECTED) ----------------------------------
# For each model two files are searched in DIR_DATA with these patterns;
# {code} is replaced by the model code (second element of MODELS below).
HIST_PATTERN = "*baseline_{code}.txt"   # historical / baseline file
FUT_PATTERN = "*rcp85_{code}.txt"       # future scenario file
SCENARIO_NAME = "rcp85"                 # used in the output file names
MOD_SEP = ","
MOD_TIME_COL = "time"    # name of the time column in the model files

# Models to process: {id: [model name (used in outputs), model code (used in file names)]}
# Models whose files are not found are skipped.
MODELS = {
    1: ["CLMcom-CCCma-CanESM2", "CCLM4-CanESM2"],
    2: ["CLMcom-MIROC-MIROC5", "CCLM4-MIROC5"],
    3: ["SMHI-MPI-M-MPI-ESM-LR", "RCA4-MPI-ESM-LR"],
    4: ["SMHI-NCC-NorESM1-M", "RCA4-NorESM1-M"],
    5: ["SMHI-CNRM-CERFACS-CNRM-CM5", "RCA4-CNRM-CM5"],
    6: ["SMHI-IPSL-IPSL-CM5A-MR", "RCA4-IPSL-CM5A-MR"],
    7: ["SMHI-MOHC-HadGEM2-ES", "RCA4-HadGEM2-ES"],
    8: ["SMHI-ICHEC-EC-EARTH", "RCA4-EC-EARTH"],
    9: ["DMI-ICHEC-EC-EARTH", "HIRHAM5-EC-EARTH"],
    10: ["DMI-NCC-NorESM1-M", "HIRHAM5-NorESM1-M"],
    11: ["DMI-MOHC-HadGEM2-ES", "HIRHAM5-HadGEM2-ES"],
    12: ["DMI-MPI-M-MPI-ESM-LR", "HIRHAM5-MPI-ESM-LR"],
    13: ["DMI-CNRM-CERFACS-CNRM-CM5", "HIRHAM5-CNRM-CM5"],
    14: ["DMI-IPSL-IPSL-CM5A-MR", "HIRHAM5-IPSL-CM5A-MR"],
    15: ["CLMcom-ETH-ICHEC-EC-EARTH", "COSMO-crCLIM1-EC-EARTH"],
    16: ["CLMcom-ETH-NCC-NorESM1-M", "COSMO-crCLIM1-NorESM1-M"],
    17: ["CLMcom-ETH-MOHC-HadGEM2-ES", "COSMO-crCLIM1-HadGEM2-ES"],
    18: ["KNMI-CNRM-CERFACS-CNRM-CM5", "RACMO22E-CNRM-CM5"],
    19: ["KNMI-ICHEC-EC-EARTH", "RACMO22E-EC-EARTH"],
    20: ["KNMI-IPSL-IPSL-CM5A-MR", "RACMO22E-IPSL-CM5A-MR"],
    21: ["KNMI-MOHC-HadGEM2-ES", "RACMO22E-HadGEM2-ES"],
}

# --- Bias-adjustment parameters -----------------------------------------
THRESH = # threshold percentile separating lower/mid part and upper tail
              # (use the value chosen for the corrected hindcast)

# Seasons: {season label: list of months}
# The order defines the quarter number used in prints and file names
# (Quarter 1 -> qt0, ...). It is the same order used by bias_adjustment.py
# (alphabetical), so Quarter N refers to the same season in both scripts.
SEASONS = {
    "DJF": [12, 1, 2],
    "JJA": [6, 7, 8],
    "MAM": [3, 4, 5],
    "SON": [9, 10, 11],
}

# Candidate distributions (names from scipy.stats) tested by `fitter`,
# for both the reference and the model historical data
DIST_MID = ["gamma", "lognorm", "beta", "norm"]        # lower/mid part
DIST_UPPER = ["gumbel_r", "expon", "weibull_max"]      # upper tail
FALLBACK_DIST = "norm"   # used when too few points or no fit converges
FIT_TIMEOUT = 30         # max seconds allowed for each distribution fit

# ========================================================================
# END OF USER SETTINGS
# ========================================================================


# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------
def fit_best_distribution(data, distributions, fallback=FALLBACK_DIST):
    """Fit the candidate distributions and return the best one.

    The best distribution is the one with the lowest sum of squared errors
    between the fitted PDF and the data histogram.

    Returns
    -------
    dist_name : str                  name of the chosen scipy.stats distribution
    dist_obj  : scipy.stats object   the distribution itself (has .cdf, .ppf)
    params    : tuple                fitted parameters (shape(s), loc, scale)
    """
    if len(data) < 5:
        # Too few points for a meaningful comparison: use the fallback
        dist_obj = getattr(scipy.stats, fallback)
        return fallback, dist_obj, dist_obj.fit(data)

    fit = Fitter(data, density=True, distributions=distributions, timeout=FIT_TIMEOUT)
    fit.fit()
    best_fit_dict = fit.get_best(method="sumsquare_error")

    if not best_fit_dict:
        # No candidate converged: use the fallback
        dist_obj = getattr(scipy.stats, fallback)
        return fallback, dist_obj, dist_obj.fit(data)

    dist_name = list(best_fit_dict.keys())[0]
    dist_obj = getattr(scipy.stats, dist_name)
    params = fit.fitted_param[dist_name]
    return dist_name, dist_obj, params


def read_model(fname):
    """Read a model file and keep only the time index and the variable VAR."""
    df = pd.read_csv(fname, sep=MOD_SEP, parse_dates=[MOD_TIME_COL],
                     index_col=MOD_TIME_COL)
    return pd.DataFrame(df[VAR])


def plot_ecdf(data_mod, data_ba, data_obs, title, fname):
    """Plot ECDFs of original model, bias-adjusted model and reference."""
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=data_mod, label="Original Model", color="blue")
    sns.ecdfplot(data=data_ba, label="Bias Adjusted", color="red")
    sns.ecdfplot(data=data_obs, label="Observed", color="green", linestyle="dashed")
    plt.title(title)
    plt.xlabel(VAR)
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


# Create output folders if they do not exist
DIR_OUT.mkdir(parents=True, exist_ok=True)
DIR_PICS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------
# Read reference data (bias-adjusted hindcast)
# ------------------------------------------------------------------------
raw = pd.read_csv(OBS_FILE, sep=OBS_SEP)
obs_time_col = raw.columns[OBS_TIME_COL] if isinstance(OBS_TIME_COL, int) else OBS_TIME_COL
obs_var_col = raw.columns[OBS_VAR_COL] if isinstance(OBS_VAR_COL, int) else OBS_VAR_COL
raw[obs_time_col] = pd.to_datetime(raw[obs_time_col])
df_obs = (
    raw.set_index(obs_time_col)[[obs_var_col]]
       .rename(columns={obs_var_col: VAR})   # same column name as the models
)
# Keep only the calibration period
df_obs = df_obs[(df_obs.index.year >= CALIB_START) & (df_obs.index.year <= CALIB_END)]


# ------------------------------------------------------------------------
# Main loop: one bias adjustment per model
# ------------------------------------------------------------------------
for model_name, model_code in MODELS.values():
    print(f"\nProcessing model: {model_name}")

    # Find historical and future files of this model
    hist_files = list(DIR_DATA.glob(HIST_PATTERN.format(code=model_code)))
    fut_files = list(DIR_DATA.glob(FUT_PATTERN.format(code=model_code)))
    if not hist_files or not fut_files:
        print(f"  Missing files for {model_name}, skipped.")
        continue

    df_mod_hist = read_model(hist_files[0])   # historical / baseline
    df_mod_fut = read_model(fut_files[0])     # future scenario

    # ---------------- Seasonal loop ----------------
    for mm, (season, months) in enumerate(SEASONS.items()):
        quarter = f"Quarter {mm+1} ({season})"   # label used in prints and plot titles

        # Select the months of this season
        obs_season = df_obs[df_obs.index.month.isin(months)]
        mod_season_hist = df_mod_hist[df_mod_hist.index.month.isin(months)]
        mod_season_fut = df_mod_fut[df_mod_fut.index.month.isin(months)]

        # Thresholds (percentile THRESH): the future run is split at the
        # threshold of the HISTORICAL run
        mod_thresh_hist = mod_season_hist[VAR].quantile(THRESH / 100)
        obs_thresh = obs_season[VAR].quantile(THRESH / 100)

        # ---------- LOWER / MID PART ----------
        mod_mid_hist = mod_season_hist[mod_season_hist[VAR] <= mod_thresh_hist][VAR].dropna()
        mod_mid_fut = mod_season_fut[mod_season_fut[VAR] <= mod_thresh_hist][VAR].dropna()
        obs_mid = obs_season[obs_season[VAR] < obs_thresh][VAR].dropna()

        # Best-fitting distributions (reference and model historical)
        dist_name, f_obs_mid, par_obs_mid = fit_best_distribution(obs_mid, DIST_MID)
        print(f"  [{quarter}] best fit reference lower/mid: {dist_name}")
        dist_name, f_mod_mid, par_mod_mid = fit_best_distribution(mod_mid_hist, DIST_MID)
        print(f"  [{quarter}] best fit model historical lower/mid: {dist_name}")

        # Quantile mapping; the FUTURE uses the HISTORICAL model distribution
        cdf_hist = f_mod_mid.cdf(mod_mid_hist, *par_mod_mid)
        cdf_fut = f_mod_mid.cdf(mod_mid_fut, *par_mod_mid)
        df_mod_hist.loc[mod_mid_hist.index, f"{VAR}_ba"] = f_obs_mid.ppf(cdf_hist, *par_obs_mid)
        df_mod_fut.loc[mod_mid_fut.index, f"{VAR}_ba"] = f_obs_mid.ppf(cdf_fut, *par_obs_mid)

        # ---------- UPPER TAIL ----------
        mod_upper_hist = mod_season_hist[mod_season_hist[VAR] > mod_thresh_hist][VAR].dropna()
        mod_upper_fut = mod_season_fut[mod_season_fut[VAR] > mod_thresh_hist][VAR].dropna()
        obs_upper = obs_season[obs_season[VAR] > obs_thresh][VAR].dropna()

        # Best-fitting distributions (reference and model historical)
        dist_name, f_obs_upper, par_obs_upper = fit_best_distribution(obs_upper, DIST_UPPER)
        print(f"  [{quarter}] best fit reference upper tail: {dist_name}")
        dist_name, f_mod_upper, par_mod_upper = fit_best_distribution(mod_upper_hist, DIST_UPPER)
        print(f"  [{quarter}] best fit model historical upper tail: {dist_name}")

        cdf_hist = f_mod_upper.cdf(mod_upper_hist, *par_mod_upper)
        cdf_fut = f_mod_upper.cdf(mod_upper_fut, *par_mod_upper)
        df_mod_hist.loc[mod_upper_hist.index, f"{VAR}_ba"] = f_obs_upper.ppf(cdf_hist, *par_obs_upper)
        df_mod_fut.loc[mod_upper_fut.index, f"{VAR}_ba"] = f_obs_upper.ppf(cdf_fut, *par_obs_upper)

        # ---------- Seasonal ECDF plot (historical period) ----------
        hist_months = df_mod_hist.index.month.isin(months)
        plot_ecdf(df_mod_hist.loc[hist_months, VAR],
                  df_mod_hist.loc[hist_months, f"{VAR}_ba"],
                  obs_season,
                  f"CDF - {model_name} - {quarter}",
                  DIR_PICS / f"{model_name}_CDF_qt{mm}.png")

    # ---------- ECDF plot, all seasons (historical period) ----------
    plot_ecdf(df_mod_hist[VAR], df_mod_hist[f"{VAR}_ba"], df_obs,
              f"CDF - {model_name} - ALL",
              DIR_PICS / f"{model_name}_CDF_ALL.png")

    # ---------- Save corrected series (columns: VAR, VAR_ba) ----------
    out_hist = DIR_OUT / f"{model_name}_historical_bias_adj_{SITE_NAME}.csv"
    out_fut = DIR_OUT / f"{model_name}_{SCENARIO_NAME}_bias_adj_{SITE_NAME}.csv"
    df_mod_hist.to_csv(out_hist)
    df_mod_fut.to_csv(out_fut)
    print(f"  Saved: {out_hist.name}, {out_fut.name}")
