# awaca_radiosonde

Analysis of radiosonde profiles from station **89642 — Dumont d'Urville (DDU), Antarctica**.

Data source: [Météo-France open data](https://donneespubliques.meteofrance.fr) — HR_complet format.

---

## Project structure

```
src/
  download_data_mem.py       Download and parse radiosonde archives from Météo-France
  processing.py              Low-level parsing of the HR_complet CSV format
  utils.py                   QC pipeline: stitching, cleaning, interpolation, diagnostics
  feature_engineering.py     Physics-based scalar feature extraction and constants
  cluster_profile.py         ClusterProfiles and FeatureClusterProfiles classes
  plot_radiosonde_linear.py  Linear profile plots (T, Td, wind speed, wind direction)
  plot_radiosonde_skewt.py   Skew-T log-P diagrams (MetPy)
  plot_radiosonde_ddu_map.py Station location map (Cartopy)

notebooks/
  clustering/                Profile-shape and feature-based clustering workflows
  analysis/                  Seasonal and inter-annual analysis
  exploration/               Exploratory work
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate awaca_rs
```

All dependencies (numpy, pandas, scipy, scikit-learn, matplotlib, metpy, cartopy, httpx) are declared in `environment.yml`.

---

## Usage

Each notebook is self-contained. It adds `../../src` to `sys.path` so all modules resolve automatically — no install step needed.

Typical workflow:

```python
import numpy as np
from download_data_mem import get_data
from utils import clean_and_interpolate
from cluster_profile import ClusterProfiles

# 1. Download
soundings = get_data(years=[2023, 2024], months=[6, 7, 8], hours=[0])

# 2. QC + interpolate
z_grid = np.arange(50, 5000, 25)
soundings_int, stats = clean_and_interpolate(soundings, z_grid=z_grid)

# 3. Cluster
cp = ClusterProfiles(variables=['t', 'td', 't-td', 'u', 'v'])
cp.build_X(soundings_int)
cp.treat_nan(strategy="mean")
cp.normalize_shape(['t', 'td', 't-td'])
cp.apply_pca(n_components=2)
labels = cp.fit_kmeans(k=5)
cp.plot_cluster_mean_profiles(z_grid=z_grid)
```
