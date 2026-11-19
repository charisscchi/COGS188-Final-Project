from collections.abc import Mapping, Sequence
from itertools import islice
from numbers import Integral

import networkx as nx

from routing import Node, checked_cost, path_cost


def generate_candidate_routes(
    graph: Mapping[Node, Sequence[tuple[Node, float]]],
    start: Node,
    goal: Node,
    k: int = 3,
) -> list[list[Node]]:
    """
    Generate up to k distinct, loop-free routes
    between a starting point and a destination.

    Graph weights represent estimated travel time
    in seconds.
    """

    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer.")

    network = nx.DiGraph()
    network.add_nodes_from(graph)

    # Convert our existing graph into a directed graph
    for node, neighbors in graph.items():

        for neighbor, cost in neighbors:

            if network.has_edge(node, neighbor):
                raise ValueError(
                    "Parallel roads require distinct road IDs."
                )

            network.add_edge(
                node,
                neighbor,
                weight=checked_cost(cost)
            )

    if start not in network or goal not in network:
        return []

    try:
        paths = nx.shortest_simple_paths(
            network,
            start,
            goal,
            weight="weight"
        )

        return list(islice(paths, k))

    except nx.NetworkXNoPath:
        return []


def route_travel_time(
    graph: Mapping[Node, Sequence[tuple[Node, float]]], route: Sequence[Node],
) -> float:
    """Calculate a legacy junction path's cost, rejecting ambiguous roads."""
    return path_cost(graph, route)


if __name__ == "__main__":
    from traffic_aware import build_graph

    roads = [
        ("S", "A", 300, 10),
        ("A", "G", 300, 10),
        ("S", "B", 300, 10),
        ("B", "C", 400, 10),
        ("C", "G", 300, 10)
    ]

    traffic_speeds = {
        ("S", "A"): 2,
        ("A", "G"): 2
    }

    graph = build_graph(roads, traffic_speeds)

    candidates = generate_candidate_routes(
        graph,
        start="S",
        goal="G",
        k=3
    )

    print("Candidate Route Generation")
    print("=" * 40)
    print("Available routes:", len(candidates))

    for i, route in enumerate(candidates, start=1):

        cost = route_travel_time(graph, route)

        print("\nCandidate", i)
        print("Route:", " -> ".join(route))
        print("Estimated travel time:", cost, "seconds")

    # Verify route generation
    assert len(candidates) == 2
    assert candidates[0] == ["S", "B", "C", "G"]
    assert candidates[1] == ["S", "A", "G"]

    assert generate_candidate_routes(
        {"S": [], "G": []}, "S", "G"
    ) == []

    print("\nCandidate route generation test passed!")
