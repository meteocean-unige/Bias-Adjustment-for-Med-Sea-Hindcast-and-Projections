#!/usr/bin/env python
# coding: utf-8
"""
Seasonal quantile-mapping bias adjustment of a model significant wave height time series


Method ("DM-seasonal"):

  1. Model and observed data are grouped by season (DJF, MAM, JJA, SON).
  2. Within each season the data are split at the threshold `QQU`:
       - lower/mid part  (values <= quantile QQU)
       - upper tail      (values >= quantile QQU)

    The threshold must be chosen by the user.
  3. For each part, the best-fitting parametric distribution is chosen
     (with `fitter`) separately for the model and for the observations.
  4. Each model value is mapped to the observed distribution:
         x_adj = F_obs^-1( F_mod(x) )
  5. The adjusted series is saved to a text file, together with diagnostic plots.

Terminology
  obs : reference data used to correct the model
  mod : model data to be corrected 

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
DIR_DATA = Path("data")   # folder containing input files (and where output is written)
DIR_PICS = Path("pics")   # folder where figures are saved (created if missing)

# --- Names (used only to build the output file name) --------------------
SITE_NAME = "STATION_NAME"     # site / buoy name
MODEL_NAME = "MODEL_NAME"       # model name (e.g. "ERA5", "Unstr")

# NOTE on columns: every *_COL setting accepts either
#   - the column NAME in the file header (e.g. "hs"), or
#   - the column POSITION as an integer, starting from 0 (e.g. 1 = second column).
# Use positions when the header is missing or has unusual names.

# --- Model data (data to be CORRECTED) ----------------------------------
# Text/CSV file with one time column and one column with the variable.
MOD_FILE = DIR_DATA / "model_file.txt"
MOD_SEP = ","            # column separator: "," for CSV, r"\s+" for whitespace
MOD_TIME_COL = "time"    # time column (name or position)
MOD_VAR_COL = "hs"       # variable column (name or position)
MOD_DAYFIRST = False     # True if dates are written as dd/mm/yyyy

# --- Observed data (REFERENCE used for the correction, e.g. buoy) -------
OBS_FILE = DIR_DATA / "observed_file.txt"
OBS_SEP = ","
OBS_TIME_COL = "time"    # time column (name or position)
OBS_VAR_COL = "hs"       # variable column (name or position)
OBS_DAYFIRST = False

# --- Variable name used inside the script and in the output ------------
# The model column is renamed to VAR, the observed column to f"{VAR}_obs",
# and the corrected column will be f"{VAR}_ba".
VAR = "hs"

# --- Time matching between model and observations ----------------------
# Model values are matched to observation times with a backward fill
# ("bfill") limited to MATCH_LIMIT steps of the model time series.
MATCH_METHOD = "bfill"
MATCH_LIMIT = 3

# --- Bias-adjustment parameters -----------------------------------------
QQU = #threshold value, quantile separating the lower/mid part from the upper tail

# Candidate distributions (names from scipy.stats) tested by `fitter`.
DIST_LOWER = ["gamma", "lognorm", "beta", "norm"]   # lower/mid part
DIST_UPPER = ["gumbel_r", "expon", "weibull_max"]   # upper tail
FALLBACK_DIST = "norm"   # used when too few points or no fit converges
FIT_TIMEOUT = 30         # max seconds allowed for each distribution fit

# --- Season definition (month number -> season label) -------------------
# Change this mapping to use other groupings (e.g. monthly: {1:'Jan', ...}).
MONTH_TO_SEASON = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}

# --- Output file with the bias-adjusted model series --------------------
OUT_FILE = DIR_DATA / f"{SITE_NAME}_{MODEL_NAME}_bias_adj_seasonal.txt"

# ========================================================================
# END OF USER SETTINGS
# ========================================================================


# ------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------
def ecdf(x, bins="auto", cdf=True):
    """Empirical CDF (or PDF if cdf=False) computed from a histogram.

    Returns bin centres and the corresponding cumulative (or density) values.
    """
    x = np.asarray(x)
    x = x[~np.isnan(x)]                                   # ignore missing values
    ysp, xsp = np.histogram(x, bins=bins, density=True)  # density per bin
    xsp = np.diff(xsp) / 2 + xsp[:-1]                    # bin edges -> bin centres
    if cdf:
        Ysp = np.cumsum(ysp) / np.cumsum(ysp)[-1]        # normalised cumulative sum
        return xsp, Ysp
    return xsp, ysp


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


def plot_fit(data_mod, dist_mod, par_mod, data_obs, dist_obs, par_obs,
             part_label, season, fname):
    """Compare empirical CDFs with the fitted theoretical CDFs (mod vs obs)."""
    fig, axs = plt.subplots(1, 2, sharey=True)
    for ax, data, dist, par, title in [
        (axs[0], data_mod, dist_mod, par_mod, f"Model {part_label}"),
        (axs[1], data_obs, dist_obs, par_obs, f"Obs {part_label}"),
    ]:
        x, y = ecdf(data)
        ax.plot(x, y, "*", label="ECDF")
        x_teo = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_teo, dist.cdf(x_teo, *par), label=f"{dist.name} fit")
        ax.legend()
        ax.set_title(title)
    fig.suptitle(f"Goodness-of-fit {part_label} - {season}")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------
# Read model data (to be corrected)
# ------------------------------------------------------------------------
def read_series(fname, sep, time_col, var_col, dayfirst, new_name):
    """Read a time series file and return a one-column DataFrame.

    time_col / var_col can be column names (str) or positions (int, from 0).
    The returned DataFrame has a datetime index and one column called new_name.
    """
    raw = pd.read_csv(fname, sep=sep)
    # Convert positions (int) to column names
    if isinstance(time_col, int):
        time_col = raw.columns[time_col]
    if isinstance(var_col, int):
        var_col = raw.columns[var_col]
    raw[time_col] = pd.to_datetime(raw[time_col], dayfirst=dayfirst)
    return (
        raw.set_index(time_col)[[var_col]]      # keep only the variable
           .rename(columns={var_col: new_name})  # standard column name
           .sort_index()                         # chronological order
    )


df_mod = read_series(MOD_FILE, MOD_SEP, MOD_TIME_COL, MOD_VAR_COL,
                     MOD_DAYFIRST, new_name=VAR)

# ------------------------------------------------------------------------
# Read observed data (reference)
# ------------------------------------------------------------------------
df_obs = read_series(OBS_FILE, OBS_SEP, OBS_TIME_COL, OBS_VAR_COL,
                     OBS_DAYFIRST, new_name=f"{VAR}_obs")

# Create the figure folder if it does not exist
DIR_PICS.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------
# Initial comparison (before correction): model vs obs at common times
# ------------------------------------------------------------------------
df = df_obs.join(
    df_mod.reindex(df_obs.index, method=MATCH_METHOD, limit=MATCH_LIMIT)
).dropna()

g = sns.jointplot(data=df, x=f"{VAR}_obs", y=VAR, kind="reg")
g.savefig(DIR_PICS / "initial_jointplot.png", bbox_inches="tight")
plt.close(g.figure)


# ------------------------------------------------------------------------
# Bias-adjustment function
# ------------------------------------------------------------------------
def bias_dm_fitter(var, qqu=QQU):
    """Seasonal distribution-mapping bias adjustment.

    Adds the column f"{var}_ba" (bias-adjusted model) to df_mod,
    produces diagnostic plots and saves the adjusted series to OUT_FILE.
    """
    var_mod = df_mod[var]               # model series (to correct)
    var_obs = df_obs[f"{var}_obs"]      # observed series (reference)
    df_mod[f"{var}_ba"] = np.nan        # column that will hold the corrected values

    # Group both series by season (label from MONTH_TO_SEASON).
    # groupby sorts the groups alphabetically: [DJF, JJA, MAM, SON].
    # NOTE: both datasets must contain all 4 seasons, because model and obs
    # groups are paired by their position in the list.
    list_mod = list(var_mod.groupby(lambda t: MONTH_TO_SEASON.get(t.month)))
    list_obs = list(var_obs.groupby(lambda t: MONTH_TO_SEASON.get(t.month)))

    for mm in range(4):
        season = f"Quarter {mm+1}"          # label used in prints and plot titles
        var_season_mod = list_mod[mm][1]    # model data of season mm
        var_season_obs = list_obs[mm][1]    # observed data of season mm

        # ---------------- LOWER / MID PART (<= quantile qqu) ----------------
        var_mod_low = var_season_mod[var_season_mod <= var_season_mod.quantile(qqu)]
        var_obs_low = var_season_obs[var_season_obs <= var_season_obs.quantile(qqu)]

        # Best-fitting distribution for model and obs
        name, f_mod_low, par_mod_low = fit_best_distribution(var_mod_low, DIST_LOWER)
        print(f"[{season}] best fit model lower/mid: {name}")
        name, f_obs_low, par_obs_low = fit_best_distribution(var_obs_low, DIST_LOWER)
        print(f"[{season}] best fit obs   lower/mid: {name}")

        # Quantile mapping: x_adj = F_obs^-1( F_mod(x) )
        cdf_vals = f_mod_low.cdf(var_mod_low.values, *par_mod_low)
        var_mod_low_ba = f_obs_low.ppf(cdf_vals, *par_obs_low)

        plot_fit(var_mod_low, f_mod_low, par_mod_low,
                 var_obs_low, f_obs_low, par_obs_low,
                 "lower/mid", season,
                 DIR_PICS / f"{var}_fit_lowermid_qt{mm}.png")

        # Store corrected values at the original model times
        df_mod.loc[var_mod_low.index, f"{var}_ba"] = var_mod_low_ba

        # ---------------- UPPER TAIL (>= quantile qqu) ----------------
        var_mod_high = var_season_mod[var_season_mod >= var_season_mod.quantile(qqu)]
        var_obs_high = var_season_obs[var_season_obs >= var_season_obs.quantile(qqu)]

        name, f_mod_high, par_mod_high = fit_best_distribution(var_mod_high, DIST_UPPER)
        print(f"[{season}] best fit model upper tail: {name}")
        name, f_obs_high, par_obs_high = fit_best_distribution(var_obs_high, DIST_UPPER)
        print(f"[{season}] best fit obs   upper tail: {name}")

        cdf_vals = f_mod_high.cdf(var_mod_high.values, *par_mod_high)
        var_mod_high_ba = f_obs_high.ppf(cdf_vals, *par_obs_high)

        plot_fit(var_mod_high, f_mod_high, par_mod_high,
                 var_obs_high, f_obs_high, par_obs_high,
                 "upper tail", season,
                 DIR_PICS / f"{var}_fit_upper_qt{mm}.png")

        # Note: a value exactly equal to the quantile belongs to both parts;
        # the upper-tail correction is the one kept (written last).
        df_mod.loc[var_mod_high.index, f"{var}_ba"] = var_mod_high_ba

        # ---------------- Seasonal ECDF: raw vs obs vs corrected ----------------
        season_ba = np.hstack([var_mod_low_ba, var_mod_high_ba])
        fig = plt.figure()
        for data, label in [(var_season_mod, "model raw"),
                            (var_season_obs, "obs"),
                            (season_ba, "model bias-adjusted")]:
            x, y = ecdf(data)
            plt.plot(x, y, label=label)
        plt.legend()
        plt.grid()
        plt.xlabel(var)
        plt.ylabel("ECDF")
        plt.title(f"Bias adjustment - {season}")
        fig.savefig(DIR_PICS / f"{var}_bias_qt{mm}.png", bbox_inches="tight")
        plt.close(fig)

    # ---------------- Full time series ----------------
    fig = plt.figure(figsize=(15, 10))
    plt.plot(df_mod[f"{var}_ba"], label="model bias-adjusted")
    plt.plot(var_mod, label="model raw")
    plt.plot(var_obs, label="obs", linewidth=2)
    plt.legend()
    fig.savefig(DIR_PICS / f"{var}_bias_timeseries.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- Full-period ECDF ----------------
    fig = plt.figure()
    for data, label in [(var_mod, "model raw"),
                        (var_obs, "obs"),
                        (df_mod[f"{var}_ba"], "model bias-adjusted")]:
        x, y = ecdf(data)
        plt.plot(x, y, label=label)
    plt.legend()
    plt.grid()
    plt.xlabel(var)
    plt.ylabel("ECDF")
    plt.title("Bias adjustment - all seasons")
    fig.savefig(DIR_PICS / f"{var}_bias_ALL.png", bbox_inches="tight")
    plt.close(fig)

    # ---------------- Final comparison: corrected model vs obs ----------------
    df_plot = df_obs.join(
        df_mod.reindex(df_obs.index, method=MATCH_METHOD, limit=MATCH_LIMIT)
    ).dropna()
    g = sns.jointplot(data=df_plot, x=f"{var}_obs", y=f"{var}_ba", kind="reg")
    g.savefig(DIR_PICS / f"joint_{var}_ba.png", bbox_inches="tight")
    plt.close(g.figure)

    # ---------------- Save the corrected series ----------------
    df_mod[f"{var}_ba"].to_csv(OUT_FILE, index_label="datetime")
    print(f"Bias-adjusted series saved to: {OUT_FILE}")


# ------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------
if __name__ == "__main__":
    bias_dm_fitter(VAR, qqu=QQU)
