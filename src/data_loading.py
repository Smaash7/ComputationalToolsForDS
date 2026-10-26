"""Convert the Kaggle "Flight Delay Dataset - 2024" CSV to Parquet, sample it and load it."""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import FLIGHTS_PARQUET, RANDOM_SEED, RAW_CSV, SAMPLE_PARQUET
from src.profiling import MB, track

CATEGORY_COLS = ["op_unique_carrier", "origin", "dest", "origin_state_nm", "dest_state_nm",
                 "cancellation_code"]
INT8_COLS = ["month", "day_of_month", "day_of_week", "cancelled", "diverted"]
HHMM_COLS = ["crs_dep_time", "crs_arr_time"]  # scheduled clock time, e.g. 1435 = 14:35
FLOAT_COLS = [
    "crs_elapsed_time", "distance",
    "dep_delay", "arr_delay",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
]
USECOLS = ["fl_date"] + CATEGORY_COLS + INT8_COLS + HHMM_COLS + FLOAT_COLS

# Known only after the flight: use as targets or to describe clusters, NEVER as features.
OUTCOME_COLS = [
    "cancelled", "diverted", "cancellation_code",
    "dep_delay", "arr_delay",
    "carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay",
]
# The actual-time columns (dep_time, taxi_out, wheels_off, wheels_on, taxi_in, arr_time,
# air_time, actual_elapsed_time) are never read, so they cannot leak into a model.


def _read_dtypes():
    """dtypes for reading the CSV; numbers are read as float32 and narrowed in _clean_chunk()."""
    dtypes = {c: "float32" for c in INT8_COLS + HHMM_COLS + FLOAT_COLS}
    dtypes.update({c: "category" for c in CATEGORY_COLS})
    dtypes["fl_date"] = "string"
    return dtypes


def _clean_chunk(chunk):
    """Convert a raw chunk to the final compact dtypes."""
    for c in INT8_COLS:
        chunk[c] = chunk[c].astype("int8")
    for c in HHMM_COLS:
        chunk[c] = chunk[c].astype("Int16")
    chunk["fl_date"] = pd.to_datetime(chunk["fl_date"])
    return chunk


def _to_arrow(chunk, schema=None):
    """Convert a chunk to a pyarrow Table, with categories written as strings."""
    out = chunk.copy()
    for c in CATEGORY_COLS:
        out[c] = out[c].astype("string")
    return pa.Table.from_pandas(out, schema=schema, preserve_index=False)


def csv_to_parquet(chunksize=500_000, force=False):
    """Read the raw CSV in chunks and write one compact Parquet file (no rows dropped)."""
    if FLIGHTS_PARQUET.exists() and not force:
        print(f"{FLIGHTS_PARQUET.name} already exists -> skipping (use force=True to redo).")
        return None
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"CSV not found at {RAW_CSV}. See 'Getting the data' in the README.")
    FLIGHTS_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    mem_optimized = 0
    writer = None
    schema = None
    with track("csv_to_parquet", chunksize=chunksize) as stats:
        reader = pd.read_csv(RAW_CSV, usecols=USECOLS, dtype=_read_dtypes(), chunksize=chunksize)
        try:
            for i, chunk in enumerate(reader, start=1):
                chunk = _clean_chunk(chunk)[USECOLS]
                mem_optimized += chunk.memory_usage(deep=True).sum()
                if writer is None:
                    table = _to_arrow(chunk)
                    schema = table.schema
                    writer = pq.ParquetWriter(FLIGHTS_PARQUET, schema, compression="snappy")
                else:
                    table = _to_arrow(chunk, schema=schema)
                writer.write_table(table)
                n_rows += len(chunk)
                print(f"  chunk {i}: {n_rows:,} rows written", end="\r")
        except BaseException:
            # A half-written file would be skipped on the next run, so delete it.
            if writer is not None:
                writer.close()
                writer = None
            FLIGHTS_PARQUET.unlink(missing_ok=True)
            raise
        finally:
            if writer is not None:
                writer.close()
        print()
        stats["rows"] = n_rows

    stats.update(
        csv_size_mb=round(RAW_CSV.stat().st_size / MB, 1),
        parquet_size_mb=round(FLIGHTS_PARQUET.stat().st_size / MB, 1),
        in_memory_optimized_mb=round(mem_optimized / MB, 1),
    )
    print(f"Rows: {n_rows:,} | CSV: {stats['csv_size_mb']} MB -> Parquet: {stats['parquet_size_mb']} MB")
    print(f"In-memory size with optimized dtypes: {stats['in_memory_optimized_mb']} MB")
    return stats


def load_flights(path=FLIGHTS_PARQUET, columns=None):
    """Load a Parquet file (full data or a sample), optionally only some columns."""
    cat_cols = [c for c in CATEGORY_COLS if columns is None or c in columns]
    # read_dictionary returns categories directly, without building millions of Python strings.
    return pq.read_table(path, columns=columns, read_dictionary=cat_cols).to_pandas()


def make_sample(n=100_000, sample_path=SAMPLE_PARQUET, force=False):
    """Save a uniform random sample of n flights (fixed seed, so everyone gets the same one)."""
    if sample_path.exists() and not force:
        print(f"{sample_path.name} already exists -> skipping (use force=True to redo).")
        return None
    sample_path.parent.mkdir(parents=True, exist_ok=True)

    with track("make_sample", n=n) as stats:
        n_total = pq.ParquetFile(FLIGHTS_PARQUET).metadata.num_rows
        rng = np.random.default_rng(RANDOM_SEED)
        idx = np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))
        sample = load_flights().iloc[idx].reset_index(drop=True)
        sample.to_parquet(sample_path, index=False)
        stats["rows"] = len(sample)

    print(f"Sample: {len(sample):,} of {n_total:,} rows -> {sample_path}")
    return stats


def compare_dtype_memory(nrows=200_000):
    """Compare memory of default pandas dtypes vs. our optimized dtypes on the first nrows rows."""
    default = pd.read_csv(RAW_CSV, nrows=nrows)
    optimized = _clean_chunk(pd.read_csv(RAW_CSV, nrows=nrows, usecols=USECOLS, dtype=_read_dtypes()))
    return pd.DataFrame({
        "setup": ["all 35 columns, default dtypes",
                  "our columns, default dtypes",
                  "our columns, optimized dtypes"],
        "columns": [default.shape[1], len(USECOLS), len(USECOLS)],
        f"memory_mb_for_{nrows}_rows": [
            default.memory_usage(deep=True).sum() / MB,
            default[USECOLS].memory_usage(deep=True).sum() / MB,
            optimized.memory_usage(deep=True).sum() / MB,
        ],
    }).round(1)


if __name__ == "__main__":
    csv_to_parquet()
    make_sample()
