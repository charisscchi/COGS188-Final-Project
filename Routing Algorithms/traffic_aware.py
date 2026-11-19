
import math
from routing import astar


def build_graph(roads, traffic_speeds=None):
    """
    Build a road graph with travel time as the edge cost.

    Road lengths are measured in meters.
    Road speeds are measured in meters per second.
    Travel times are measured in seconds.
    """

    graph = {}

    if traffic_speeds is None:
        traffic_speeds = {}

    for start, end, length, normal_speed in roads:

        # Use normal speed when traffic data is unavailable
        speed = traffic_speeds.get(
            (start, end), normal_speed
        )

        # Handle invalid or missing speed observations
        if speed is None or not math.isfinite(speed) or speed <= 0:
            speed = normal_speed

        # Assume vehicles cannot exceed the normal speed limit
        speed = min(speed, normal_speed)

        # Calculate estimated road travel time
        travel_time = length / speed

        graph.setdefault(start, []).append(
            (end, travel_time)
        )

        graph.setdefault(end, [])

    return graph


if __name__ == "__main__":

    # Example roads:
    # (start, end, length in meters, normal speed in m/s)
    roads = [
        ("S", "A", 300, 10),
        ("A", "G", 300, 10),
        ("S", "B", 300, 10),
        ("B", "C", 400, 10),
        ("C", "G", 300, 10)
    ]

    # Example traffic speeds in meters per second
    traffic_speeds = {
        ("S", "A"): 2,
        ("A", "G"): 2,
        ("S", "B"): 10,
        ("B", "C"): 10,
        ("C", "G"): 10
    }

    # Lower-bound travel time estimates for this test graph
    heuristic = {
        "S": 60,
        "A": 30,
        "B": 70,
        "C": 30,
        "G": 0
    }

    # Build the two road graphs
    static_graph = build_graph(roads)

    traffic_graph = build_graph(
        roads, traffic_speeds
    )

    # Run both routing methods
    static_path, static_cost = astar(
        static_graph, "S", "G", heuristic
    )

    traffic_path, traffic_cost = astar(
        traffic_graph, "S", "G", heuristic
    )

    # Calculate how long the static route would take
    # under the actual example traffic conditions
    static_actual_cost = sum(
        dict(traffic_graph[u])[v]
        for u, v in zip(static_path, static_path[1:])
    )

    print("Static A*")
    print("-" * 40)
    print("Route:", " -> ".join(static_path))
    print("Estimated travel time:", static_cost, "seconds")
    print("Time with traffic:", static_actual_cost, "seconds")

    print("\nTraffic-aware A*")
    print("-" * 40)
    print("Route:", " -> ".join(traffic_path))
    print("Time with traffic:", traffic_cost, "seconds")

    improvement = static_actual_cost - traffic_cost

    print("\nComparison")
    print("-" * 40)
    print("Time saved:", improvement, "seconds")

    # Verify the expected results
    assert static_path == ["S", "A", "G"]
    assert traffic_path == ["S", "B", "C", "G"]
    assert static_actual_cost == 300
    assert traffic_cost == 100

    print("\nTraffic-aware A* test passed!")
