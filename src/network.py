"""Airport network for RQ2: one edge per directed route, weighted by number of scheduled flights."""
import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity

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


def best_louvain(graph, n_seeds=20):
    """Run Louvain with seeds 0..n_seeds-1 and keep the run with the highest modularity.
    Communities are sorted by size, so community 0 is the largest."""
    best_communities = None
    best_modularity = -1
    for seed in range(n_seeds):
        communities = louvain_communities(graph, weight="flights", seed=seed)
        score = modularity(graph, communities, weight="flights")
        if score > best_modularity:
            best_communities = communities
            best_modularity = score
    return sorted(best_communities, key=len, reverse=True), best_modularity


def fit_communities(flights, months, n_seeds=20):
    """RQ2 features for RQ3: build the airport network from `months` only and return one row per airport
    with its Louvain community, betweenness, degree and flights.

    `flights` needs the columns month, origin and dest. The main RQ3 setup calls this once with months 1-6 and
    keeps those communities fixed in all folds; refitting per training window is optional (a robustness check),
    and then community numbers can change between windows, so `community_name` (the busiest airport of the
    community) is easier to compare. Only airports with flights in `months` are returned: when joining to
    flights, airports not in the training months get community "unknown"."""
    graph = build_graph(build_edges(flights, months=months))

    communities, best_modularity = best_louvain(graph, n_seeds=n_seeds)
    community_of = {}
    for number, community in enumerate(communities):
        for airport in community:
            community_of[airport] = number

    flights_per_airport = dict(graph.degree(weight="flights"))
    betweenness = nx.betweenness_centrality(graph, weight="distance")
    betweenness_hops = nx.betweenness_centrality(graph)

    airports = pd.DataFrame({"airport": sorted(graph)})
    airports["community"] = airports["airport"].map(community_of)
    airports["betweenness"] = airports["airport"].map(betweenness)
    airports["betweenness_hops"] = airports["airport"].map(betweenness_hops)
    airports["degree"] = airports["airport"].map(dict(graph.degree()))
    airports["flights"] = airports["airport"].map(flights_per_airport)

    # name each community after its busiest airport
    busiest = airports.sort_values("flights", ascending=False).groupby("community")["airport"].first()
    airports["community_name"] = airports["community"].map(busiest)

    airports.attrs["modularity"] = best_modularity
    return airports
