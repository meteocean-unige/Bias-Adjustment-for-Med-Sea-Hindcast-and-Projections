#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
from fitter import Fitter
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
def ecdf(x, bins='auto', cdf=True):
    ysp, xsp = np.histogram(x, bins=bins, density=True)
    xsp = np.diff(xsp) / 2 + xsp[:-1]
    if cdf:
        Ysp = np.cumsum(ysp) / np.cumsum(ysp)[-1]
        return xsp, Ysp
    else:
        return xsp, ysp

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def fit_best_distribution(data, distributions, fallback='norm'):
    if len(data) < 5:
        # fallback if too few data points
        dist_obj = getattr(scipy.stats, fallback)
        params = dist_obj.fit(data)
        return fallback, dist_obj, params
    fit = Fitter(data, density=True, distributions=distributions, timeout=30)
    fit.fit()
    best_fit_dict = fit.get_best(method='sumsquare_error')
    if not best_fit_dict:
        dist_name = fallback
        dist_obj = getattr(scipy.stats, fallback)
        params = dist_obj.fit(data)
    else:
        dist_name = list(best_fit_dict.keys())[0]
        dist_obj = getattr(scipy.stats, dist_name)
        params = fit.fitted_param[dist_name]
    return dist_name, dist_obj, params

# ------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------
dir_pics = Path('pics_ww3-unst_Ponza')
dir_hind = Path('data_ww3-unst_Ponza')


# ------------------------------------------------------------------------
# File names
# ------------------------------------------------------------------------
# name_hind = 'Point_003209'
name_buoy = 'PONZA_boa_processed'

# ------------------------------------------------------------------------
# Read hindcast data
# ------------------------------------------------------------------------


mod = pd.read_table(dir_hind/"PONZA_ww3-unst_hindcast.txt", delimiter=',')
mod_df = mod[['hs']]
mod['time'] = pd.to_datetime(mod['time'], dayfirst=False)
df_hind = mod.set_index('time')[['hs']]

df_hind

# ------------------------------------------------------------------------
# Read observational data
# ------------------------------------------------------------------------
df_obs = pd.read_table(dir_hind / f'{name_buoy}.txt',
    sep=',',
    skiprows=1,
    usecols=[0,1],    
    index_col=0,
    names=['time', 'hs_obs']
)

df_obs.index = pd.to_datetime(df_obs.index)

# ------------------------------------------------------------------------
# Quick plots for initial check
# ------------------------------------------------------------------------


df = df_obs.join(df_hind.reindex(df_obs.index, method='bfill', limit=3)).dropna()

g = sns.jointplot(data=df, x='hs_obs', y='hs', kind='reg')
g.savefig(dir_pics / 'initial_jointplot.png', bbox_inches='tight')

# ------------------------------------------------------------------------
# Season mapping
# ------------------------------------------------------------------------
month_num_to_season = {
    1: 'DJF', 2: 'DJF',
    3: 'MAM', 4: 'MAM', 5: 'MAM',
    6: 'JJA', 7: 'JJA', 8: 'JJA',
    9: 'SON', 10: 'SON', 11: 'SON',
    12: 'DJF'
}

# ------------------------------------------------------------------------
# Bias correction function
# ------------------------------------------------------------------------
def bias_dm_fitter(var, qqu=0.95):

    var_hind = df_hind[var]
    var_obs = df_obs[f'{var}_obs']
    df_hind[f'{var}_ba'] = np.NaN

    list_hind = list(var_hind.groupby(lambda x: month_num_to_season.get(x.month)))
    list_obs = list(var_obs.groupby(lambda x: month_num_to_season.get(x.month)))

    for mm in range(4):
        var_month_hind = list_hind[mm][1]
        var_month_obs = list_obs[mm][1]

        # LOWER TAIL / MID
        var_hind_low = var_month_hind[
            var_month_hind <= var_month_hind.quantile(qqu)
        ]
        var_obs_low = var_month_obs[
            var_month_obs <= var_month_obs.quantile(qqu)
        ]

        dist_name, f_hindc, fit_var_hind_low = fit_best_distribution(
            var_hind_low,
            ["gamma", "lognorm", "beta", "norm"]
        )
        print(f"Best fit hindcast lower/mid Quarter {mm+1}: {dist_name}")

        dist_name, f_obsc, fit_var_obs_low = fit_best_distribution(
            var_obs_low,
            ["gamma", "lognorm", "beta", "norm"]
        )
        print(f"Best fit obs lower/mid Quarter {mm+1}: {dist_name}")

        cdf_vals = f_hindc.cdf(var_hind_low.values, *fit_var_hind_low)
        var_hind_low_ba = f_obsc.ppf(cdf_vals, *fit_var_obs_low)

        # plot lower/mid
        fig, axs = plt.subplots(1, 2, sharey=True)
        x, y = ecdf(var_hind_low)
        axs[0].plot(x, y, '*', label='ECDF')
        x_teo = np.linspace(var_hind_low.min(), var_hind_low.max(), 100)
        y_teo = f_hindc.cdf(x_teo, *fit_var_hind_low)
        axs[0].plot(x_teo, y_teo, label=f"{f_hindc.name} fit")
        axs[0].legend()
        axs[0].set_title('Hindcast lower/mid')

        x, y = ecdf(var_obs_low)
        axs[1].plot(x, y, '*', label='ECDF')
        x_teo = np.linspace(var_obs_low.min(), var_obs_low.max(), 100)
        y_teo = f_obsc.cdf(x_teo, *fit_var_obs_low)
        axs[1].plot(x_teo, y_teo, label=f"{f_obsc.name} fit")
        axs[1].legend()
        axs[1].set_title('Obs lower/mid')
        fig.suptitle(f"Goodness-of-fit lower/mid Quarter {mm+1}")
        plt.savefig(dir_pics / f"{var}_fit_lowermid_qt{mm}.png", bbox_inches='tight')

        df_hind.loc[var_hind_low.index, f"{var}_ba"] = var_hind_low_ba
        var_hind_ba_mid = var_hind_low_ba

        # UPPER TAIL
        var_hind_high = var_month_hind[
            var_month_hind >= var_month_hind.quantile(qqu)
        ]
        var_obs_high = var_month_obs[
            var_month_obs >= var_month_obs.quantile(qqu)
        ]

        dist_name, f_hind, fit_var_hind_high = fit_best_distribution(
            var_hind_high,
            ["gumbel_r", "expon", "weibull_max"]
        )
        print(f"Best fit hindcast upper-tail Quarter {mm+1}: {dist_name}")

        dist_name, f_obs, fit_var_obs_high = fit_best_distribution(
            var_obs_high,
            ["gumbel_r", "expon", "weibull_max"]
        )
        print(f"Best fit obs upper-tail Quarter {mm+1}: {dist_name}")

        cdf_vals = f_hind.cdf(var_hind_high.values, *fit_var_hind_high)
        var_hind_high_ba = f_obs.ppf(cdf_vals, *fit_var_obs_high)

        # plot upper-tail
        fig, axs = plt.subplots(1, 2, sharey=True)
        x, y = ecdf(var_hind_high)
        axs[0].plot(x, y, '*', label='ECDF')
        x_teo = np.linspace(var_hind_high.min(), var_hind_high.max(), 100)
        y_teo = f_hind.cdf(x_teo, *fit_var_hind_high)
        axs[0].plot(x_teo, y_teo, label=f"{f_hind.name} fit")
        axs[0].legend()
        axs[0].set_title('Hindcast upper-tail')

        x, y = ecdf(var_obs_high)
        axs[1].plot(x, y, '*', label='ECDF')
        x_teo = np.linspace(var_obs_high.min(), var_obs_high.max(), 100)
        y_teo = f_obs.cdf(x_teo, *fit_var_obs_high)
        axs[1].plot(x_teo, y_teo, label=f"{f_obs.name} fit")
        axs[1].legend()
        axs[1].set_title('Obs upper-tail')
        fig.suptitle(f"Goodness-of-fit upper-tail Quarter {mm+1}")
        plt.savefig(dir_pics / f"{var}_fit_upper_qt{mm}.png", bbox_inches='tight')

        dati_ba = np.hstack([var_hind_ba_mid, var_hind_high_ba])

        # plot combined
        plt.figure()
        x, y = ecdf(var_month_hind)
        plt.plot(x, y, label='hindcast raw', linestyle='-')
        x, y = ecdf(var_month_obs)
        plt.plot(x, y, label='obs', linestyle='-')
        x, y = ecdf(dati_ba)
        plt.plot(x, y, label='hindcast DM-quarter', linestyle='-')
        plt.legend()
        plt.grid()
        plt.xlabel(var)
        plt.ylabel('ECDF')
        plt.title(f"Bias-adjustment ALL Quarter {mm+1}")
        plt.savefig(dir_pics / f"{var}_bias_qt{mm}.png", bbox_inches='tight')

        df_hind.loc[var_hind_high.index, f"{var}_ba"] = var_hind_high_ba

    # plot full time series
    plt.figure(figsize=(15, 10))
    plt.plot(df_hind[f"{var}_ba"], label='hindcast DM-month')
    plt.plot(var_hind, label='hindcast raw')
    plt.plot(var_obs, label='obs', linewidth=2)
    plt.legend()
    plt.savefig(dir_pics / f"{var}_bias_timeseries.png", bbox_inches='tight')

    # plot full ECDF
    plt.figure()
    x, y = ecdf(var_hind)
    plt.plot(x, y, label='hindcast raw', linestyle='-')
    x, y = ecdf(var_obs)
    plt.plot(x, y, label='obs', linestyle='-')
    x, y = ecdf(df_hind[f"{var}_ba"])
    plt.plot(x, y, label='hindcast DM-quarter', linestyle='-')
    plt.legend()
    plt.grid()
    plt.xlabel(var)
    plt.ylabel('ECDF')
    plt.title("Bias-adjustment ALL")
    plt.savefig(dir_pics / f"{var}_bias_ALL.png", bbox_inches='tight')

    # final jointplot
    df_plot = df_obs.join(
        df_hind.reindex(df_obs.index, method='bfill', limit=3)
    ).dropna()

    g = sns.jointplot(
        data=df_plot,
        x=f"{var}_obs",
        y=f"{var}_ba",
        kind='reg'
    )
    g.savefig(dir_pics / f"joint_{var}_ba.png", bbox_inches='tight')

    # save corrected series
    fn_out = dir_hind / f"{name_buoy}_Unstr_bias_adj_seasonal.txt"
    df_hind[f"{var}_ba"].to_csv(fn_out, index_label='datetime')

# ------------------------------------------------------------------------
# Run bias correction
# ------------------------------------------------------------------------
var = 'hs'
qqu = 0.95
bias_dm_fitter(var, qqu=qqu)




