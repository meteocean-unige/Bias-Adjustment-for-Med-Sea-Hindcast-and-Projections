# Bias Adjustment of Wave Data Along the Italian Coast

Scripts used to produce the dataset:  
"A bias-adjusted wave dataset of reanalyses, hindcasts and climate projections along the Italian coast" — [Zenodo](https://doi.org/10.5281/zenodo.17854569)

**Variables:** significant wave height `Hs` (univariate) and `Hs`–`Tp` (bivariate).  
**Sites (15):** Alghero, Ancona, Cagliari, Capo Mele, Catania, Cesenatico, Cetraro, Civitavecchia, Crotone, La Spezia, Mazara, Monopoli, Ortona, Palermo, Ponza.

**Terminology:** `obs` = reference data, `mod` = model data to correct, `_ba` = bias-adjusted.

## Contents

Scripts are in `script_ba/`:

- `univariate_era5/` — ERA5 reanalysis vs. buoy (one script per site)
- `univariate_cmems/` — CMEMS reanalysis vs. buoy (one script per site)
- `univariate_ww3-10km/` — MeteOcean-UniGe WW3 10 km hindcast vs. buoy (one script per site)
- `univariate_ww3-unst/` — MeteOcean-UniGe WW3 unstructured hindcast vs. buoy (one per site)
- `univariate_projs/` — EURO-CORDEX projections (21 RCM-GCM, baseline + RCP8.5) vs. bias-adjusted WW3 10 km hindcast (one script per site)
- `multivariate/` — `Hs`–`Tp` correction with dOTC (`xsdba`): hindcast vs. buoy (`.py`), projections vs. corrected hindcast (`.ipynb`)
- `templates/` — generic versions of the univariate scripts for new sites/data

## Method

### Univariate: DM-seasonal

Seasonal distribution mapping is applied separately for DJF, MAM, JJA, and SON. In each season, the data are split at a percentile threshold into the body and upper tail. The best distribution for each part is selected with `fitter`, and each model value is mapped as:

```text
x_ba = F_obs^-1(F_mod(x))
```

Body candidates: `gamma`, `lognorm`, `beta`, `norm`. Tail candidates: `gumbel_r`, `expon`, `weibull_max`, `expon`. The threshold for each site is set in its script.

### Projections

The reference is the bias-adjusted WW3 10 km hindcast (1979–2005). The future run is mapped with the distributions of the historical run, split at the historical threshold, so the climate-change signal is preserved. The scripts test thresholds 85, 90, and 95 (one output file per threshold).

### Bivariate

dOTC (`xsdba.adjustment.dOTC`), additive for `Hs` and `Tp`.

### Workflow

1. Correct hindcast/reanalysis against buoys.
2. Correct projections against the corrected WW3 10 km hindcast.

## Usage

```bash
pip install -r requirements.txt
```

The site scripts contain the folder paths used on the author's computer; edit them before running. In the templates, the threshold is intentionally left empty and must be set.

Figures in the multivariate scripts use LaTeX (`usetex=True`). Set it to `False` if LaTeX is not installed.
