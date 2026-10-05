# Do similar flights get delayed in similar ways?
## Goal
Flight delays spread through the air network: a late aircraft delays its next flight, and congested hubs pass delays on to other airports. We test whether structurally similar flights (route, airline, distance, schedule) show **similar delay patterns**, and whether airports form **communities** with shared delay behaviour.

## Research questions
Question and Main methods 
**RQ1** (core) | Can route + airline profiles, built only from schedule/route features, be grouped into meaningful clusters? Do the clusters differ in delay level and delay causes? 
Methods: K-Means (our own implementation, validated against scikit-learn), MiniBatchKMeans, hierarchical, DBSCAN; Silhouette, Davies–Bouldin; Kruskal–Wallis 


**RQ2** (core) | Does the airport network contain communities with similar delays? Which airports act as bridges? 
Methods: Louvain and modularity, spectral clustering, betweenness centrality 


**RQ3** (extension) Do cluster and community labels help predict a 15+ minute arrival delay, compared with a per-route baseline? 
Methods: Gradient boosting (LightGBM, outside the course); train Jan–Sep, test Oct–Dec 

## Folder structure
```
ComputationalTools/
├── data/
│   ├── raw/           flight_data_2024.csv (Kaggle download)
│   ├── processed/     flights_2024.parquet (full data, compact)
│   └── sample/        flights_2024_sample.parquet (100k rows)
├── notebooks/         01_data_loading.ipynb
├── src/
│   ├── config.py          paths and random seed
│   ├── profiling.py       runtime and memory measurement
│   └── data_loading.py    CSV -> Parquet, sample, loader
├── results/
│   ├── tables/        tables used in the report
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

## Data source

Hrishit Patil, *Flight Delay Dataset — 2024*, Kaggle: `hrishitpatil/flight-data-2024`. US domestic flights in 2024, about 7 million rows and 35 columns.