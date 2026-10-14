"""Airport network for RQ2: one edge per directed route, weighted by number of scheduled flights."""
from src.config import CLEAN_PARQUET, EDGES_PARQUET
from src.data_loading import load_flights
from src.profiling import track


def build_edges(df, months=range(1, 7)):
    """Directed edges origin -> dest with all scheduled flights (incl. cancelled/diverted) in `months`.
    `distance` = 1 / flights, the edge length used for betweenness (busy routes are short)."""
    sub = df[df["month"].isin(list(months))]
    edges = sub.groupby(["origin", "dest"], observed=True).size().rename("flights").reset_index()
    edges["distance"] = 1 / edges["flights"]
    return edges


def make_edges_parquet():
    with track("build_edges", months="1-6") as stats:
        edges = build_edges(load_flights(CLEAN_PARQUET, columns=["month", "origin", "dest"]))
        edges.to_parquet(EDGES_PARQUET, index=False)
        stats["edges"] = len(edges)
    return edges
