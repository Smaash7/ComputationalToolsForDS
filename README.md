# Do similar flights get delayed in similar ways?
## Goal
Flight delays spread through the air network: a late aircraft delays its next flight, and congested hubs pass delays on to other airports. We test whether structurally similar flights (route, airline, distance, schedule) show **similar delay patterns**, and whether airports form **communities** with shared delay behaviour.

## Research questions
Question and Main methods 
**RQ1** (core) | Can route + airline profiles, built only from schedule/route features, be grouped into meaningful clusters? Do the clusters differ in delay level and delay causes? 
Methods: K-Means (our own implementation, validated against scikit-learn) and hierarchical clustering; Silhouette, Davies–Bouldin; Kruskal–Wallis 


**RQ2** (core) | Does the airport network contain communities with similar delays? Which airports act as bridges? 
Methods: Louvain and modularity, spectral clustering, betweenness centrality 


**RQ3** (extension) | Do the outputs of RQ1 (route cluster) and RQ2 (airport community, betweenness) help predict a 15+ minute arrival delay, compared with a per-route baseline? 
Methods: Explainable Boosting Machine (EBM, outside the course) with the RQ1 and RQ2 outputs as features; rolling monthly evaluation (train on months 1..m, test on month m+1, for m = 6..11); ROC-AUC and PR-AUC reported next to each month's base rate 

## Decisions
- Route profiles are built on months 1–6 only, with at least 30 flights per route + airline pair (about one per week).
- The airline is not a clustering feature, but it is a feature in the EBM (RQ3).
- Time-of-day bins: 5–12h, 12–18h, 18–24h, 0–5h.
- Delays are compared between clusters and communities with medians.
- Betweenness uses distance = 1 / number of flights.
- Flights with an impossible scheduled duration will be removed in the cleaning step.
- In RQ3, each month's PR-AUC is reported next to its base rate.
- RQ1 clusters and RQ2 communities are built once on months 1–6 and kept fixed in all folds (labels stay stable). Only the historical delay rates (route and cluster) are recomputed per fold from the months before the test month.
- Cleaning rule (`src/cleaning.py`, removes 41 flights; cancelled and diverted flights are kept, labelled in `status`): a flight is removed if `crs_elapsed_time` is missing or <= 0, if its implied speed (distance / scheduled duration) is above 650 mph, or if its scheduled duration is less than half or more than twice the median scheduled duration of its route (origin, dest).

## Folder structure
```
ComputationalTools/
├── data/
│   ├── raw/           flight_data_2024.csv (Kaggle download)
│   ├── processed/     flights_2024.parquet (full data, compact)
│   └── sample/        flights_2024_sample.parquet (100k rows)
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 01b_eda.ipynb  exploratory data analysis
│   ├── 01c_cleaning_network.ipynb  cleaning + airport edges (RQ2)
│   ├── 02_route_clustering_Approach_A.ipynb  RQ1: route profiles (months 1-6), K-Means (own + scikit-learn),
│   │                                         hierarchical, Kruskal-Wallis, delay causes, cluster lookup for RQ3
│   ├── 02b_route_clusters_interpretation.ipynb  RQ1: cluster names, delay medians/percentiles,
│   │                                         delay causes per cluster, hierarchical vs K-Means runtime
│   ├── 03_airport_network.ipynb    RQ2: Louvain, spectral, betweenness
│   └── 04_predicting_delays.ipynb  RQ3: model comparisons, evaluation and feature importance
├── prediction/                RQ3 pipeline, entry point and settings.config
├── src/
│   ├── config.py          paths and random seed
│   ├── profiling.py       runtime and memory measurement
│   ├── data_loading.py    CSV -> Parquet, sample, loader
│   ├── cleaning.py        cleaning rule -> flights_2024_clean.parquet
│   └── network.py         airport edge table (months 1-6) and graph; fit_communities and
│                          best_louvain give RQ3 one row per airport (community, betweenness)
├── results/
│   ├── tables/        tables used in the report (e.g. 03_spectral_mainland.csv:
│   │                  spectral vs Louvain on the mainland airports only)
│   ├── figures/       figures used in the report (03_community_map.png: airport coordinates
│   │                  from the airportsdata package, used only for this map, not in any analysis)
│   └── logs/          performance_log.csv
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer.

**Windows (PowerShell):**
```powershell
cd "C:\path\to\ComputationalTools"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ipykernel install --user --name ctds --display-name "Python (ctds)"
```

**macOS / Linux:**
```bash
cd path/to/ComputationalTools
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ctds --display-name "Python (ctds)"
```

Then select the kernel **"Python (ctds)"** in Jupyter or VS Code.

## Getting the data

1. Download the CSV from Kaggle: <https://www.kaggle.com/datasets/hrishitpatil/flight-data-2024>
2. Put it at `data/raw/flight_data_2024.csv` (or set the environment variable `FLIGHTS_CSV` to its path).
3. Run notebook 01 once. It converts the CSV to Parquet (about 1–2 minutes) and creates the sample.

## Running the notebooks
Run the notebooks in order. Notebook 01 skips the conversion if the Parquet files already exist (`FORCE = False`).
Notebook 01c then creates `data/processed/flights_2024_clean.parquet`, which notebooks 02, 02b, 03 and 04 read.
Notebook 02b reads the clusters exported by notebook 02, so run 02 first if the clustering changes.

### RQ3 prediction

After running notebooks 01, 01c, 02 and 03, run notebook `04_predicting_delays.ipynb`, or run
`python prediction/main.py` from the repository root. Both use `prediction/settings.config`.
The default fits on up to 200,000 flights per training window and evaluates up to
100,000 per month. July–August select the model and an F2 threshold subject to at least 50%
validation specificity (configurable to avoid an all-positive classifier); September–December are
chronological rolling evaluations. The model search compares expanding windows with the last
two and four months. All September–December 2024 outcomes have already been inspected during
earlier iterations, so they are exploratory for this revised model rather than a fresh independent
test. The command `python prediction/main.py --smoke` runs a much smaller
end-to-end check and writes to a separate smoke subdirectory.

The pipeline compares EBM, logistic regression and XGBoost candidates, the route-rate baseline,
and flight/RQ1/RQ2 ablations for both the selected model and the EBM. It writes metrics,
confusion counts, sampling details, validation rankings, per-feature permutation importance,
selection settings and the final fitted model to `results/rq3_pipeline_v2/`. The previous run is
preserved in `results/rq3_pipeline/`. The saved model is the last evaluation fold's model
(trained through November using its selected window), and requires the same feature engineering
and historical inputs; it is not a standalone raw-flight predictor. Historical rates use only
preceding months. Unknown routes with a prior schedule profile are assigned to the nearest
reconstructed RQ1 cluster; genuinely new routes remain unknown until history is available.
The fixed January–June structural features are retrospective training representations; the
original RQ1 profile excludes cancellations, so it is not a strictly real-time network snapshot.
Use `python -m unittest discover -s tests -p test_prediction.py` for focused checks.
Resource measurements report elapsed wall time and sampled peak resident memory of the main
process, not the memory of any child worker processes.

The original fixed-threshold results are the primary July–August-selected evaluation. A
supplementary rolling threshold uses the previous month's out-of-sample predictions and labels
to select F2, then evaluates the following month. This policy was added after inspecting the
previous results: all adaptive months in the revised experiment are exploratory rather than
independent tests. Run
`python prediction/main.py --adaptive-only` to recompute just this supplementary analysis.

The revision also tested adding day of month, scheduled arrival hour, scheduled duration,
airport identifiers, and airport historical delay rates in July/August. The full expanded set
lowered mean validation PR-AUC from 0.4111 to 0.3978 under the same depth-3/250-tree model;
the original 13-feature input was retained. A fresh year or later data is needed for an
independent evaluation after this iteration.

The data files in `data/raw/` and `data/processed/` are not committed (see `.gitignore`); every group member creates
them locally with notebooks 01 and 01c, which gives identical files. Running 01 and 01c also rewrites some tables in
`results/` and adds lines to `results/logs/performance_log.csv`: do not commit those changes unless the data step itself changed.

## Data source

Hrishit Patil, *Flight Delay Dataset — 2024*, Kaggle: `hrishitpatil/flight-data-2024`. US domestic flights in 2024, about 7 million rows and 35 columns.
