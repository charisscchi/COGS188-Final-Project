from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from candidate_routes import generate_candidate_routes
from epsilon_greedy import EpsilonGreedy, RouteID
from road_network import Endpoint, RoadNetwork, RoadPosition, SearchNode, build_road_query
from routing import astar
from traffic_aware import TrafficDiagnostics, TrafficSnapshot, normalize_time, routing_speeds


RoutingMethod = Literal["static", "traffic_aware", "epsilon_greedy"]
TrafficProvider = Callable[[datetime], TrafficSnapshot]
PickupHeuristic = Callable[[RoadPosition, RoadPosition], float]


@dataclass(frozen=True)
class RouteOption:
    road_ids: RouteID
    estimated_time_s: float


@dataclass(frozen=True)
class RoutingResult:
    method: RoutingMethod
    status: Literal["ok", "unreachable"]
    road_ids: RouteID | None
    estimated_time_s: float | None
    candidates: tuple[RouteOption, ...]
    diagnostics: TrafficDiagnostics


def plan_route(
    network: RoadNetwork,
    start_road_id: str,
    pickup_road_id: str,
    *,
    request_time: datetime,
    method: RoutingMethod,
    start_position_m: float | None = None,
    pickup_position_m: float | None = None,
    traffic: TrafficSnapshot | TrafficProvider | None = None,
    vehicle_class: str = "passenger",
    heuristic: PickupHeuristic | None = None,
    k: int = 3,
    bandit: EpsilonGreedy | None = None,
) -> RoutingResult:
    """Plan one pickup. Invalid inputs raise ValueError.

    A heuristic receives the current position and the specific pickup position.
    The caller must establish that it lower-bounds remaining estimated seconds
    for this query's costs. The default zero heuristic is Dijkstra-equivalent.
    """
    if method not in ("static", "traffic_aware", "epsilon_greedy"):
        raise ValueError(f"Unknown routing method {method!r}.")
    if method == "epsilon_greedy" and bandit is None:
        raise ValueError("Supply an EpsilonGreedy agent for the recurring pickup context.")
    requested = normalize_time(request_time)
    start = network.position(start_road_id, start_position_m)
    pickup = network.position(pickup_road_id, pickup_position_m, at_end=True)
    if callable(traffic):
        snapshot = traffic(requested)
        if not isinstance(snapshot, TrafficSnapshot):
            raise ValueError("Traffic provider must return a TrafficSnapshot.")
    else:
        snapshot = traffic
    if snapshot is not None:
        if not isinstance(snapshot, TrafficSnapshot) or snapshot.request_time != requested:
            raise ValueError("Traffic provider must return a snapshot for this request time.")
    speeds, diagnostics = routing_speeds(network, snapshot, use_traffic=method != "static")
    query = build_road_query(network, start, pickup, speeds, vehicle_class)

    if method == "epsilon_greedy":
        paths = generate_candidate_routes(query.graph, Endpoint.START, Endpoint.PICKUP, k)
        candidates = tuple(
            RouteOption(road_ids, query.travel_time(road_ids))
            for road_ids in map(query.road_ids, paths)
        )
        if not candidates:
            return RoutingResult(method, "unreachable", None, None, (), diagnostics)
        bandit.set_candidates(option.road_ids for option in candidates)
        chosen = candidates[bandit.choose_route()]
        return RoutingResult(method, "ok", chosen.road_ids, chosen.estimated_time_s, candidates, diagnostics)

    if heuristic is not None and not callable(heuristic):
        raise ValueError("Use a callable for the pickup heuristic.")

    def estimate(state: SearchNode) -> float:
        if heuristic is None or state == Endpoint.PICKUP:
            return 0.0
        position = RoadPosition(state, 0.0) if isinstance(state, str) else start
        return heuristic(position, pickup)

    path, cost = astar(
        query.graph, Endpoint.START, Endpoint.PICKUP, estimate if heuristic is not None else None,
    )
    if path is None:
        return RoutingResult(method, "unreachable", None, None, (), diagnostics)
    return RoutingResult(method, "ok", query.road_ids(path), cost, (), diagnostics)


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
    from traffic_aware import build_graph

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
