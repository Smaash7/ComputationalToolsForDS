"""Airport network for RQ2: one edge per directed route, weighted by number of scheduled flights."""
import networkx as nx
import pandas as pd

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


def build_graph(edges):
    """Undirected weighted graph: A->B and B->A are merged, flights summed, distance = 1 / flights."""
    # Write every route with the alphabetically first airport as "a", so A->B and B->A match
    undirected = pd.DataFrame()
    undirected["a"] = edges["origin"].astype(str)
    undirected["b"] = edges["dest"].astype(str)
    undirected["flights"] = edges["flights"]
    swap = undirected["a"] > undirected["b"]
    undirected.loc[swap, ["a", "b"]] = undirected.loc[swap, ["b", "a"]].to_numpy()

    # Merge A->B and B->A into one undirected edge by adding their flights
    undirected = undirected.groupby(["a", "b"], as_index=False)["flights"].sum()

    # Build the graph; busy routes get a short distance (used for betweenness)
    graph = nx.Graph()
    for row in undirected.itertuples(index=False):
        graph.add_edge(row.a, row.b, flights=int(row.flights), distance=1 / row.flights)
    return graph
