
from itertools import islice
import math
import networkx as nx

from traffic_aware import build_graph


def generate_candidate_routes(graph, start, goal, k=3):
    """
    Generate up to k distinct, loop-free routes
    between a starting point and a destination.

    Graph weights represent estimated travel time
    in seconds.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    network = nx.DiGraph()
    network.add_nodes_from(graph)

    # Convert our existing graph into a directed graph
    for node, neighbors in graph.items():

        for neighbor, cost in neighbors:

            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid travel cost.")

            if network.has_edge(node, neighbor):
                raise ValueError(
                    "Parallel roads require distinct road IDs."
                )

            network.add_edge(
                node,
                neighbor,
                weight=cost
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


def route_travel_time(graph, route):
    """Calculate the total cost of a given route."""

    return sum(
        dict(graph[start])[end]
        for start, end in zip(route, route[1:])
    )


if __name__ == "__main__":

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
