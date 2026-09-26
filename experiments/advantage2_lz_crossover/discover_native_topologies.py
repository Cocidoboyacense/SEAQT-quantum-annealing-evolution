"""Discover 6-qubit topologies the real Advantage2_system1 chip offers
NATIVELY, across many physical regions, and catalog them for review --
before spending any QPU (anneal) time.

Different philosophy than `submit_hardware.find_native_embedding` (which
starts from an ABSTRACT target shape, e.g. ring_J, and asks minorminer to
embed it onto SOME physical qubits): here we start from a physical region
(a random connected window of 6 qubits) and ask what shape it naturally
gives, via the real chip graph's INDUCED subgraph on those qubits (every
real coupler among them, not just a spanning tree). Random windows that
land on physically DIFFERENT qubits but turn out to be the same shape
(isomorphic) automatically become that shape's different "regions" -- one
sampling process covers both the "more topologies" and "different chip
regions" asks at once, no separate embedding search needed.

Costs no QPU anneal time: only a live edgelist/properties query (cheap),
then pure-Python random sampling + graph-isomorphism bucketing.

Usage:
    python discover_native_topologies.py [--samples N] [--seed S]

Output (results/):
    native_topology_catalog.csv    one row per isomorphism class, sorted by
                                    how many distinct chip regions realize
                                    it (then by edge count) -- review this
                                    and hand-pick a subset for the real
                                    sweep.
    native_topology_instances.csv  one row per discovered instance
                                    (class_id, physical qubits, edges in
                                    that instance's own logical indexing) --
                                    pick specific regions from here.

Nothing here is auto-added to the real hardware sweep -- see this
experiment's README for the review -> `native_topologies_selected.json` ->
config.py pickup flow.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import networkx as nx

import config

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_N_SAMPLES = 20000


def build_chip_graph() -> nx.Graph:
    """The real, live qubit connectivity graph for config.SOLVER_NAME --
    NOT an idealized dwave_networkx.zephyr_graph, since that would include
    physical qubits that don't actually exist/work on this chip, defeating
    the point of "native". Fails loudly if the solver isn't reachable
    (e.g. still in maintenance) rather than silently falling back."""
    if config.SOLVER_NAME is None:
        raise RuntimeError("config.SOLVER_NAME is not set -- fill it in before running discovery")
    from dwave.system import DWaveSampler

    base = DWaveSampler(solver=config.SOLVER_NAME)
    G = nx.Graph()
    G.add_nodes_from(base.nodelist)
    G.add_edges_from(base.edgelist)
    print(f"Solver: {config.SOLVER_NAME} -- {G.number_of_nodes()} qubits, "
          f"{G.number_of_edges()} couplers")
    return G


def random_connected_window(G: nx.Graph, n: int, rng: random.Random) -> tuple[int, ...] | None:
    """One random connected window of exactly `n` physical qubits: pick a
    random start node, then repeatedly add a random node from the current
    frontier (neighbors of the window not already in it). Returns None if
    the window's growth gets stuck (frontier exhausted before reaching
    size n -- only possible if the graph has a connected component smaller
    than n, which doesn't happen on a real QPU chip but is guarded against
    anyway for the offline self-test)."""
    start = rng.choice(list(G.nodes))
    window = {start}
    frontier = set(G.neighbors(start))
    while len(window) < n:
        if not frontier:
            return None
        node = rng.choice(list(frontier))
        window.add(node)
        frontier.discard(node)
        frontier |= set(G.neighbors(node)) - window
    return tuple(sorted(window))


def logical_edges(subgraph: nx.Graph, physical_qubits: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Edges of `subgraph` (already induced on `physical_qubits`) renamed
    into logical 0..n-1 indices, via the fixed convention "sorted physical
    qubit id order" -- so the same physical qubit always gets the same
    logical index for a given instance, and the mapping is reproducible
    from `physical_qubits` alone."""
    idx = {q: i for i, q in enumerate(physical_qubits)}
    return tuple(sorted(tuple(sorted((idx[a], idx[b])) for a, b in subgraph.edges())))


class TopologyClass:
    def __init__(self, class_id: int, representative: nx.Graph):
        self.class_id = class_id
        self.representative = representative
        self.instances: list[tuple[int, ...]] = []  # physical-qubit tuples

    @property
    def num_edges(self) -> int:
        return self.representative.number_of_edges()

    @property
    def degree_sequence(self) -> list[int]:
        return sorted((d for _, d in self.representative.degree()), reverse=True)

    @property
    def is_bipartite(self) -> bool:
        return nx.is_bipartite(self.representative)

    @property
    def diameter(self) -> int:
        return nx.diameter(self.representative)


def classify_windows(G: nx.Graph, n_samples: int, n_qubits: int, seed: int) -> list[TopologyClass]:
    """Draw `n_samples` random connected windows of `n_qubits` physical
    qubits, take each window's induced subgraph, and bucket them into
    isomorphism classes. Bucketed first by a Weisfeiler-Lehman graph hash
    (fast, near-collision-free) then split with exact nx.is_isomorphic
    (cheap at n_qubits=6) to resolve any hash collisions correctly."""
    rng = random.Random(seed)
    seen_windows: set[tuple[int, ...]] = set()
    buckets: dict[str, list[TopologyClass]] = {}
    classes: list[TopologyClass] = []

    for _ in range(n_samples):
        physical = random_connected_window(G, n_qubits, rng)
        if physical is None or physical in seen_windows:
            continue
        seen_windows.add(physical)
        subG = G.subgraph(physical)
        h = nx.weisfeiler_lehman_graph_hash(subG, iterations=3)

        matched = None
        for cls in buckets.get(h, []):
            if nx.is_isomorphic(subG, cls.representative):
                matched = cls
                break
        if matched is None:
            matched = TopologyClass(len(classes), nx.Graph(subG))
            classes.append(matched)
            buckets.setdefault(h, []).append(matched)
        matched.instances.append(physical)

    return classes


def _self_test() -> None:
    """Offline correctness check against a small local graph -- doesn't
    need live chip access, so this always runs (cheap: milliseconds)
    before real discovery, and is also runnable standalone."""
    # A 4x4 grid graph: enough size/edge variety at n_qubits=6 to exercise
    # windowing + isomorphism dedup without needing the real chip.
    G = nx.grid_2d_graph(4, 4)
    G = nx.convert_node_labels_to_integers(G)

    rng = random.Random(0)
    for _ in range(200):
        w = random_connected_window(G, 6, rng)
        assert w is not None, "grid graph is well-connected; window growth should not get stuck"
        assert len(w) == 6 and len(set(w)) == 6
        assert nx.is_connected(G.subgraph(w)), f"window {w} is not connected in the source graph"

    classes = classify_windows(G, n_samples=3000, n_qubits=6, seed=0)
    assert len(classes) >= 2, "a 4x4 grid should yield more than one distinct 6-qubit shape"
    total_instances = sum(len(c.instances) for c in classes)
    assert total_instances > 0

    # Two hand-built isomorphic graphs (a path relabeled) must collapse to
    # the SAME class, not two different ones.
    path_a = nx.path_graph(6)
    path_b = nx.relabel_nodes(path_a, {i: (i * 7) % 11 for i in range(6)})
    ha = nx.weisfeiler_lehman_graph_hash(path_a, iterations=3)
    hb = nx.weisfeiler_lehman_graph_hash(path_b, iterations=3)
    assert ha == hb, "isomorphic graphs must hash the same (WL hash is isomorphism-invariant)"
    assert nx.is_isomorphic(path_a, path_b)

    print("[self-test] OK -- windowing + isomorphism-class dedup verified offline\n")


def write_catalog(classes: list[TopologyClass], G: nx.Graph) -> None:
    ring_ref = nx.cycle_graph(config.N_QUBITS)
    chain_ref = nx.path_graph(config.N_QUBITS)
    ring_hits = [c for c in classes if nx.is_isomorphic(c.representative, ring_ref)]
    chain_hits = [c for c in classes if nx.is_isomorphic(c.representative, chain_ref)]
    print(f"Ring (6-cycle) found natively: {'yes' if ring_hits else 'no'}"
          + (f" ({len(ring_hits[0].instances)} instances)" if ring_hits else ""))
    print(f"Chain (path-6) found natively: {'yes' if chain_hits else 'no'}"
          + (f" ({len(chain_hits[0].instances)} instances)" if chain_hits else ""))
    print("(Informational only -- ring/chain in the real sweep still go through the existing "
          "minorminer-based find_native_embedding regardless of whether they show up here.)\n")

    ordered = sorted(classes, key=lambda c: (-len(c.instances), -c.num_edges))

    catalog_path = RESULTS_DIR / "native_topology_catalog.csv"
    with open(catalog_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "num_edges", "degree_sequence", "is_bipartite", "diameter",
                          "n_instances", "example_physical_qubits", "example_edges_logical"])
        for cls in ordered:
            example = cls.instances[0]
            edges = logical_edges(G.subgraph(example), example)
            writer.writerow([
                cls.class_id, cls.num_edges, " ".join(map(str, cls.degree_sequence)),
                cls.is_bipartite, cls.diameter, len(cls.instances),
                " ".join(map(str, example)),
                ";".join(f"{i}-{j}" for i, j in edges),
            ])

    instances_path = RESULTS_DIR / "native_topology_instances.csv"
    with open(instances_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "instance_index", "physical_qubits", "edges_logical"])
        for cls in ordered:
            for i, physical in enumerate(cls.instances):
                edges = logical_edges(G.subgraph(physical), physical)
                writer.writerow([cls.class_id, i, " ".join(map(str, physical)),
                                  ";".join(f"{a}-{b}" for a, b in edges)])

    print(f"{len(ordered)} distinct isomorphism classes, {sum(len(c.instances) for c in ordered)} "
          f"total instances found.")
    print(f"Catalog:   {catalog_path}")
    print(f"Instances: {instances_path}")
    print("\nTop 15 classes by instance count:")
    print(f"{'class_id':>8}  {'edges':>5}  {'instances':>9}  degree_sequence")
    for cls in ordered[:15]:
        print(f"{cls.class_id:>8}  {cls.num_edges:>5}  {len(cls.instances):>9}  {cls.degree_sequence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _self_test()

    G = build_chip_graph()
    print(f"Sampling {args.samples} random connected {config.N_QUBITS}-qubit windows...")
    classes = classify_windows(G, n_samples=args.samples, n_qubits=config.N_QUBITS, seed=args.seed)
    write_catalog(classes, G)
