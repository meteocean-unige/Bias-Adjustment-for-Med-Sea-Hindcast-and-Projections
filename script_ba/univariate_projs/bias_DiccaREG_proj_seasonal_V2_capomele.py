#%% 

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import gumbel_r, lognorm, expon, gamma, beta, weibull_max
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from fitter import Fitter
import openpyxl   # solo per essere sicuri che il writer xlsx funzioni


def ecdf(x, bins='auto', cdf=True):
    ysp, xsp = np.histogram(x, bins=bins, density=True)
    xsp = np.diff(xsp) / 2 + xsp[:-1]
    if cdf:
        Ysp = np.cumsum(ysp) / np.cumsum(ysp)[-1]
        return xsp, Ysp
    else:
        return xsp, ysp


# Impostazioni iniziali
var = 'hs'
threshold_list = [85, 90, 95]

seasonal_data = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Fall': [9, 10, 11]
}

# Percorsi
dir_data = Path('/Users/carolina/Desktop/PROJECTION DICCA BIAS /capomele projection')
dir_proj_ba = dir_data / 'bias_adjusted_outputs'
dir_proj_ba.mkdir(parents=True, exist_ok=True)
plot_dir = dir_proj_ba / 'plots'
plot_dir.mkdir(exist_ok=True)

#%%  Dati osservati - hindcast corretto

buoy_name = 'CAPOMELE'
obs = pd.read_table(
    dir_data / f'{buoy_name}_boa_processed_Regu_bias_adj_seasonal.txt',
    delimiter=',',
    index_col=0,
    parse_dates=True
)
obs = obs.rename(columns={f'{var}_ba': var})
obs = pd.DataFrame(obs[var])
obs = obs[(obs.index.year >= 1979) & (obs.index.year <= 2005)]

#%%  Dizionario modelli CORDEX

info_cordex = {
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
    21: ["KNMI-MOHC-HadGEM2-ES", "RACMO22E-HadGEM2-ES"]
}

#%%  Funzione per il best-fit distribuzionale

def fit_best_distribution(data, distributions, fallback='norm'):
    data = pd.Series(data).dropna()
    if len(data) < 5:
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

#%%  Loop principale + raccolta summary

fit_summary = []

for thresh in threshold_list:

    for rcm in info_cordex.keys():
        model_name = info_cordex[rcm][0]
        model_code = info_cordex[rcm][1]
        print(f"\nProcessing model: {model_name} — thresh {thresh}")

        # File storico e scenario
        hist_files = list(dir_data.glob(f'*baseline_{model_code}.dat'))
        rcp_files  = list(dir_data.glob(f'*rcp85_{model_code}.dat'))

        if not hist_files or not rcp_files:
            print(f"File mancanti per {model_name}, salto al prossimo modello.")
            continue

        # Lettura file modello
        mod_hist = pd.read_table(hist_files[0], delimiter=',', parse_dates=['time'], index_col=0)
        mod_fut  = pd.read_table(rcp_files[0],  delimiter=',', parse_dates=['time'], index_col=0)

        mod_hist = pd.DataFrame(mod_hist[var])
        mod_fut  = pd.DataFrame(mod_fut[var])

        # inizializza colonna bias-adjusted
        mod_hist[f'{var}_ba'] = np.nan
        mod_fut[f'{var}_ba']  = np.nan

        # ----- LOOP STAGIONALE -----
        for season, months in seasonal_data.items():

            obs_season       = obs[obs.index.month.isin(months)]
            mod_season_hist  = mod_hist[mod_hist.index.month.isin(months)]
            mod_season_fut   = mod_fut[mod_fut.index.month.isin(months)]

            # --- Parte MID (<= soglia) ---
            mod_thresh_hist  = mod_season_hist[var].quantile(thresh / 100)
            mod_mid_hist     = mod_season_hist[mod_season_hist[var] <= mod_thresh_hist][var].dropna()
            mod_mid_fut      = mod_season_fut [mod_season_fut[var]  <= mod_thresh_hist][var].dropna()

            obs_thresh       = obs_season[var].quantile(thresh / 100)
            obs_mid          = obs_season[obs_season[var] < obs_thresh][var].dropna()

            dist_mid_hist, f_proj, fit_mid_hist = fit_best_distribution(
                mod_mid_hist, ["gamma", "lognorm", "beta", "norm"]
            )
            dist_mid_obs,  f_obs,  fit_mid_obs = fit_best_distribution(
                obs_mid, ["gamma", "lognorm", "beta", "norm"]
            )

            # CDF -> PPF (body)
            mod_cdf_hist = f_proj.cdf(mod_mid_hist, *fit_mid_hist)
            mod_cdf_fut  = f_proj.cdf(mod_mid_fut,  *fit_mid_hist)

            mod_mid_hist_ba = f_obs.ppf(mod_cdf_hist, *fit_mid_obs)
            mod_mid_fut_ba  = f_obs.ppf(mod_cdf_fut,  *fit_mid_obs)

            mod_hist.loc[mod_mid_hist.index, f'{var}_ba'] = mod_mid_hist_ba
            mod_fut.loc[mod_mid_fut.index,  f'{var}_ba']  = mod_mid_fut_ba

            # --- Parte UPPER (> soglia) ---
            mod_upper_hist = mod_season_hist[mod_season_hist[var] > mod_thresh_hist][var].dropna()
            mod_upper_fut  = mod_season_fut [mod_season_fut[var]  > mod_thresh_hist][var].dropna()
            obs_upper      = obs_season[obs_season[var] > obs_thresh][var].dropna()

            if len(mod_upper_hist) > 0 and len(obs_upper) > 0:
                dist_up_hist, f_proj_up, fit_up_hist = fit_best_distribution(
                    mod_upper_hist, ["gumbel_r", "expon", "weibull_max"]
                )
                dist_up_obs,  f_obs_up,  fit_up_obs = fit_best_distribution(
                    obs_upper, ["gumbel_r", "expon", "weibull_max"]
                )

                mod_cdf_hist_u = f_proj_up.cdf(mod_upper_hist, *fit_up_hist)
                mod_cdf_fut_u  = f_proj_up.cdf(mod_upper_fut,  *fit_up_hist)

                mod_upper_hist_ba = f_obs_up.ppf(mod_cdf_hist_u, *fit_up_obs)
                mod_upper_fut_ba  = f_obs_up.ppf(mod_cdf_fut_u,  *fit_up_obs)

                mod_hist.loc[mod_upper_hist.index, f'{var}_ba'] = mod_upper_hist_ba
                mod_fut.loc[mod_upper_fut.index,  f'{var}_ba']  = mod_upper_fut_ba
            else:
                dist_up_hist = np.nan
                dist_up_obs  = np.nan


            fit_summary.append({
                "Model": model_name,
                "Threshold": thresh,
                "Season": season,
                "Model_Bulk_Dist": dist_mid_hist,
                "Model_Tail_Dist": dist_up_hist,
                "Obs_Bulk_Dist": dist_mid_obs,
                "Obs_Tail_Dist": dist_up_obs
            })

        # ----- PLOT GLOBALI (HIST e FUT) PER QUESTO MODELLO & SOGLIA -----

        # HIST TOTAL
        plt.figure(figsize=(8, 5))
        sns.ecdfplot(data=mod_hist[var],         label='Original Model',  color='blue')
        sns.ecdfplot(data=mod_hist[f'{var}_ba'], label='Bias Adjusted',   color='red')
        sns.ecdfplot(data=obs[var],              label='Observed',        linestyle='dashed', color='green')
        plt.title(f'CDF HIST - {model_name} - thresh: {thresh}th')
        plt.xlabel(var)
        plt.ylabel('F(Hs)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{model_name}_CDF_ALL_HIST_thr{thresh}.png', dpi=300)
        plt.close()

        # FUT TOTAL
        plt.figure(figsize=(8, 5))
        sns.ecdfplot(data=mod_fut[var],          label='Original Model',  color='blue')
        sns.ecdfplot(data=mod_fut[f'{var}_ba'],  label='Bias Adjusted',   color='red')
        sns.ecdfplot(data=obs[var],              label='Observed',        linestyle='dashed', color='green')
        plt.title(f'CDF FUT - {model_name} - thresh: {thresh}th')
        plt.xlabel(var)
        plt.ylabel('F(Hs)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{model_name}_CDF_ALL_FUT_thr{thresh}.png', dpi=300)
        plt.close()

        # ----- Salvataggio file CSV (una volta per modello e soglia) -----
        out_hist = dir_proj_ba / f"{model_name}_historical_bias_adj_{buoy_name}_thr{thresh}.csv"
        out_fut  = dir_proj_ba / f"{model_name}_rcp85_bias_adj_{buoy_name}_thr{thresh}.csv"

        mod_hist.to_csv(out_hist)
        mod_fut.to_csv(out_fut)

        print(f"File salvati: {out_hist.name}, {out_fut.name}")


#%%  Esporta il riassunto dei fit in Excel

df_summary = pd.DataFrame(fit_summary)
excel_output = dir_proj_ba / "bias_correction_fit_summary.xlsx"
df_summary.to_excel(excel_output, index=False)

print(f"\nSUMMARY FILE CREATED: {excel_output}")
print("Numero righe summary:", len(df_summary))


# %%
# Controllo finale: valori massimi realistici degli Hs corretti
print("\n==============================")
print("VALORI MASSIMI DI HS_BA PER OGNI MODELLO")
print("==============================")

for file in sorted(dir_proj_ba.glob("*_historical_bias_adj_*.csv")):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    if 'hs_ba' in df.columns:
        max_val = df['hs_ba'].max()
        print(f"{file.stem}: {max_val:.2f} m")
    else:
        print(f"{file.stem}: nessuna colonna 'hs_ba' trovata")

print("==============================")

for file in sorted(dir_proj_ba.glob("*_rcp85_bias_adj_*.csv")):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    if 'hs_ba' in df.columns:
        max_val = df['hs_ba'].max()
        print(f"{file.stem}: {max_val:.2f} m")
    else:
        print(f"{file.stem}: nessuna colonna 'hs_ba' trovata")

# %%
