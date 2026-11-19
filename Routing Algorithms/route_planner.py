
from routing import astar
from traffic_aware import build_graph
from candidate_routes import generate_candidate_routes
from epsilon_greedy import EpsilonGreedy


def prepare_routes(
    static_graph,
    traffic_graph,
    start,
    goal,
    heuristic=None,
    k=3
):
    """
    Prepare routing decisions for all three methods.

    Inputs:
        static_graph: graph using normal road speeds
        traffic_graph: graph using observed traffic speeds
        start: taxi starting location
        goal: passenger pickup location
        heuristic: estimated remaining travel times
        k: maximum number of candidate routes

    Returns:
        Static A* route, traffic-aware A* route,
        and candidate routes for epsilon-greedy.
    """

    # Zero heuristic is a safe fallback when
    # geographical coordinates are unavailable.
    if heuristic is None:
        heuristic = {
            node: 0.0
            for node in static_graph
        }

    static_path, static_cost = astar(
        static_graph, start, goal, heuristic
    )

    traffic_path, traffic_cost = astar(
        traffic_graph, start, goal, heuristic
    )

    candidates = generate_candidate_routes(
        traffic_graph, start, goal, k
    )

    return {
        "static_route": static_path,
        "static_estimated_time": static_cost,
        "traffic_aware_route": traffic_path,
        "traffic_estimated_time": traffic_cost,
        "candidate_routes": candidates
    }


if __name__ == "__main__":

    # Controlled example road network
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

    heuristic = {
        "S": 60,
        "A": 30,
        "B": 70,
        "C": 30,
        "G": 0
    }

    static_graph = build_graph(roads)

    traffic_graph = build_graph(
        roads, traffic_speeds
    )

    results = prepare_routes(
        static_graph,
        traffic_graph,
        start="S",
        goal="G",
        heuristic=heuristic
    )

    print("Combined Routing System")
    print("=" * 50)

    print(
        "Static A*:",
        " -> ".join(results["static_route"])
    )

    print(
        "Traffic-aware A*:",
        " -> ".join(results["traffic_aware_route"])
    )

    candidates = results["candidate_routes"]

    print("\nCandidate routes:")

    for i, route in enumerate(candidates):
        print(i + 1, ":", " -> ".join(route))

    # Initialize the learning agent for this scenario
    agent = EpsilonGreedy(
        routes=candidates,
        epsilon=0.1,
        seed=42
    )

    # Hypothetical environment feedback
    # Replace with SUMO feedback during integration
    waiting_times = {
        ("S", "A", "G"): 300,
        ("S", "B", "C", "G"): 100
    }

    for episode in range(200):

        action = agent.choose_route()
        route = candidates[action]

        observed_wait = waiting_times[tuple(route)]

        agent.update(action, observed_wait)

    print(
        "\nEpsilon-greedy learned route:",
        " -> ".join(agent.best_route())
    )

    # Verify the combined system
    assert results["static_route"] == ["S", "A", "G"]

    assert results["traffic_aware_route"] == [
        "S", "B", "C", "G"
    ]

    assert agent.best_route() == [
        "S", "B", "C", "G"
    ]

    print("\nCombined routing system test passed!")
