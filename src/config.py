"""All file paths and the random seed. Set FLIGHTS_CSV if your CSV is not in data/raw/."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

RAW_CSV = Path(os.environ.get("FLIGHTS_CSV", DATA_DIR / "raw" / "flight_data_2024.csv"))
FLIGHTS_PARQUET = DATA_DIR / "processed" / "flights_2024.parquet"
SAMPLE_PARQUET = DATA_DIR / "sample" / "flights_2024_sample.parquet"
PERFORMANCE_LOG = RESULTS_DIR / "logs" / "performance_log.csv"

RANDOM_SEED = 42
