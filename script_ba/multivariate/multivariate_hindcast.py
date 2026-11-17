#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pathlib import Path
import xsdba
import matplotlib.pyplot as plt
import seaborn as sns
import cmocean

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=10)


# In[2]:


# Root folder of the data (edit this path)
p = Path('wave-climate-ITbuoys')

dir_boe =  p / 'raw/buoys-processed/'                            # buoy files: {name}_boa_processed.txt
dir_reg = p / 'raw/hindcast_MeteOcean-UniGe-ww3-10km/'           # hindcast files: {name}_ww3-10km_hindcast.txt
dir_reg_ba = p / 'bias-adj-Hs_Tp-dOTC/hindcast_MeteOcean-UniGe-ww3-10km/'  # corrected hindcast (output),
                                                                 # used as reference by multivariate_projs.ipynb
dir_pics = Path('pics')                                          # figures

dir_reg_ba.mkdir(parents=True, exist_ok=True)
dir_pics.mkdir(parents=True, exist_ok=True)


# In[3]:


# Station names (they replace the file coordinateDR.xlsx).
# Each name is used to build the file names, e.g. {name}_boa_processed.txt
stations = ['spezia', 'capomele', 'alghero', 'crotone', 'cagliari', 'mazara', 'monopoli',
            'civitavecchia', 'ortona', 'ancona', 'cetraro', 'cesenatico', 'ponza',
            'palermo', 'catania']


# In[8]:


for pp, nn in enumerate(stations):
    
    nn = nn.upper()
    print(nn)

    boa = pd.read_csv(dir_boe / f'{nn}_boa_processed.txt', index_col=0, parse_dates=True)
    boa.dropna(inplace=True)
    boa = boa.resample('1h').nearest()
    boa.loc[:, 'tp'] = boa.loc[:, 'tp'].values + np.random.uniform(low=0, high=0.5, size=len(boa.tp))
    
    reg = pd.read_csv(dir_reg / f'{nn}_ww3-10km_hindcast.txt', index_col=0, parse_dates=True)
    reg.dropna(inplace=True)
    
    df = pd.concat([boa[['hs', 'tp']], reg[['hs', 'tp']]], keys=["obs", "ww3-10km"], axis=1)
    #df = df.dropna(axis=0)
    df = df.loc['1979':'2020', :]

    boa = df['obs'].to_xarray()
    boa = boa.rename({'index':'time'})
    boa.hs.attrs['units'] = 'm'
    boa.tp.attrs['units'] = 's'

    reg_hist = df['ww3-10km'].to_xarray()
    reg_hist.hs.attrs['units'] = 'm'
    reg_hist.tp.attrs['units'] = 's'
    reg_hist = reg_hist.rename({'index':'time'})

    reg_hist_sim = df['ww3-10km'].to_xarray()
    reg_hist_sim.hs.attrs['units'] = 'm'
    reg_hist_sim.tp.attrs['units'] = 's'
    reg_hist_sim = reg_hist_sim.rename({'index':'time'})

    boa = xsdba.processing.stack_variables(boa)
    reg_hist = xsdba.processing.stack_variables(reg_hist)
    reg_hist_sim = xsdba.processing.stack_variables(reg_hist_sim)

    #ADJ = xsdba.MBCn.train(boa, reg_hist,
     #                      base_kws={"nquantiles": 50, "group": 'time'},
      #                     adj_kws={"interp": "nearest", "extrapolation": "constant"},
       #                    n_iter=20,  # perform 20 iteration
        #                   n_escore=-1,  # only send 1000 points to the escore metric
         #                  )

    #reg_hist_ba = ADJ.adjust(sim=reg_hist, ref=boa, hist=reg_hist, 
     #                       base=xsdba.EmpiricalQuantileMapping,
      #                       base_kws_vars={"hs": {"kind": "+"}, "tp": {"kind": "+"}},
       #                      adj_kws={"interp": "nearest", "extrapolation": "constant"},
        #                     )
    
    #OTC
    reg_hist_ba = xsdba.adjustment.dOTC.adjust(sim=reg_hist_sim, ref=boa, hist=reg_hist,
                                                kind={"hs": "+", "tp":"+"},  
                                                #adapt_freq_thresh={"pr": "3.5e-4 kg m-2 s-1"},  # Idem
                                            )
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), sharey=True, sharex=True)
    for pp, (da, label) in enumerate(zip((boa, reg_hist, reg_hist_ba), ("buoy", "raw www-10km", "bias-adjusted www-10km"))):
        ds = xsdba.unstack_variables(da).to_dataframe()
        x=ds.hs 
        y=ds.tp
        sns.scatterplot(x=x, y=y, s=5, color=".15", ax=axs[pp], alpha=0.5)
        sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako", ax=axs[pp])
        #sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1, ax=axs[pp])
        axs[pp].grid(linewidth=0.5, color='gray', alpha=0.5, linestyle='--',)
        axs[pp].set_xlabel('$H_s$ [m]')
        axs[pp].set_title(label)
    axs[0].set_ylabel('$T_p$ [s]')
    if nn == 'SPEZIA':
        fig.suptitle('LA SPEZIA', y=1.01)
    else:
        fig.suptitle(nn, y=1.01)
    fig.savefig(dir_pics / f'performance-scatter-multiv-{nn}.png',  dpi=300, bbox_inches='tight')

    ## Quantiles year
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    rollw_dy = 15

    colors = cmocean.cm.haline_r(np.linspace(0, 1, len(quantiles)))
    colors2 = cmocean.cm.matter(np.linspace(0, 1, len(quantiles)))
    
    dsq_boa = xsdba.unstack_variables(boa).groupby('time.dayofyear').quantile(quantiles).rolling(dayofyear=rollw_dy).mean()
    dsq_reg = xsdba.unstack_variables(reg_hist).groupby('time.dayofyear').quantile(quantiles).rolling(dayofyear=rollw_dy).mean()
    dsq_reg_ba = xsdba.unstack_variables(reg_hist_ba).groupby('time.dayofyear').quantile(quantiles).rolling(dayofyear=rollw_dy).mean()

    fig, axs = plt.subplots(1, 4, figsize=(11, 3), sharex=True)
    for qq, quantile in enumerate(quantiles):
        qq_obs = dsq_boa.sel(quantile=quantile)
        axs[0].plot(dsq_boa.dayofyear, qq_obs['hs'].data, '.', color=colors[qq], markersize=0.7)
        axs[0].set_title('raw')
        axs[1].plot(dsq_boa.dayofyear, qq_obs['hs'].data, '.', color=colors[qq], markersize=0.7)
        axs[1].set_title('bias-adjusted')
        axs[2].plot(dsq_boa.dayofyear, qq_obs['tp'].data, '.', color=colors2[qq], markersize=0.7)
        axs[2].set_title('raw')
        axs[3].plot(dsq_boa.dayofyear, qq_obs['tp'].data, '.', color=colors2[qq], markersize=0.7)
        axs[3].set_title('bias-adjusted')
    
        qq_reg = dsq_reg.sel(quantile=quantile)
        axs[0].plot(dsq_reg.dayofyear, qq_reg['hs'].data, color=colors[qq], linestyle='solid', linewidth=0.8, label=f'${int(quantile*100)}^{{th}}$')
        axs[2].plot(dsq_reg.dayofyear, qq_reg['tp'].data, color=colors2[qq], linestyle='solid', linewidth=0.8, label=f'${int(quantile*100)}^{{th}}$')
    
        qq_reg_ba = dsq_reg_ba.sel(quantile=quantile)
        axs[1].plot(dsq_reg.dayofyear, qq_reg_ba['hs'].data, color=colors[qq], linestyle='solid', linewidth=0.8, label=f'${int(quantile*100)}^{{th}}$')
        axs[3].plot(dsq_reg.dayofyear, qq_reg_ba['tp'].data, color=colors2[qq], linestyle='solid', linewidth=0.8, label=f'${int(quantile*100)}^{{th}}$')
    
    for pp in range(4):
        ax = axs[pp]
        # Convert xlabel from 1-365 to date
        ax.set_xlim([1, 365])
        days_range = pd.period_range('01-01', '12-31', freq='D')
        labels_date = [
            days_range[int((valn / ax.get_xticks()[-1]) * (days_range.size - 1))].strftime('%d/%m') for valn in ax.get_xticks()]
        ax.set_xticklabels(labels_date)
        ax.grid(linewidth=0.5, color='gray', alpha=0.5, linestyle='--',)
        ax.set_xlabel('Date')
    
        if pp <2:
            ax.set_ylabel('$H_s$ [m]')
        else:
            ax.set_ylabel('$T_p$ [s]')
    
    lines, labels = axs[0].get_legend_handles_labels()
    leg_ppal = fig.legend(lines, labels, bbox_to_anchor=(0.55, 1.06), loc='lower center', frameon=False, ncol=8,
                          title='percentile:')
    leg_ppal.get_title().set_position([-1050, -53])
    lines, labels = axs[2].get_legend_handles_labels()
    leg_ppal = fig.legend(lines, labels, bbox_to_anchor=(0.55, 1.02), loc='lower center', frameon=False, ncol=8)
    
    line1, = axs[0].plot([380, 390, 395], [1, 2, 2.1], 'black', linestyle='None', marker='.', markersize='2')
    line2, = axs[0].plot([380, 390], [1, 2], 'black', linestyle='solid', linewidth=1)
    leg_aux = fig.legend(handles=[line1, line2], labels=['Buoy', 'Hindcast'], frameon=False,
                         bbox_to_anchor=(0.5, 0.95), loc='lower center', handlelength=1.8, ncol=2, scatterpoints=3)
    
    if nn == 'SPEZIA':
        fig.suptitle('LA SPEZIA', y=1.25)
    else:
        fig.suptitle(nn)
    fig.subplots_adjust(wspace=0.35)
    fig.savefig(dir_pics / f'performance-qqyear-multiv-{nn}.png',  dpi=300, bbox_inches='tight')

    # Save the bias-adjusted series (columns: time, hs_ba, tp_ba)
    df_ba = xsdba.unstack_variables(reg_hist_ba).to_dataframe()
    df_ba = df_ba.rename(columns={'hs':'hs_ba', 'tp':'tp_ba'})
    df_ba = df_ba.round({'hs_ba': 2, 'tp_ba':2})
    df_ba.index.names = ['time']
    df_ba[['hs_ba', 'tp_ba']].to_csv(dir_reg_ba / f'{nn}_ww3-10km_hindcast.txt')
