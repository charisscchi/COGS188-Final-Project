from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Literal
from zoneinfo import ZoneInfo

from road_network import RoadNetwork, positive_number, validate_id
from routing import checked_cost, is_finite_number


NYC = ZoneInfo("America/New_York")


def normalize_time(value: datetime) -> datetime:
    """Return UTC. Interpret a naive time as NYC local time."""
    if not isinstance(value, datetime) or value != value:
        raise ValueError("A valid request or observation datetime is required.")
    if value.utcoffset() is not None:
        return value.astimezone(timezone.utc)
    first = value.replace(tzinfo=NYC, fold=0).astimezone(timezone.utc)
    second = value.replace(tzinfo=NYC, fold=1).astimezone(timezone.utc)
    if first != second:
        raise ValueError("Ambiguous or nonexistent NYC local time. Supply an explicit offset.")
    return first


@dataclass(frozen=True)
class TrafficObservation:
    speed_mps: float | None
    observed_at: datetime | None
    status: Literal["available", "missing", "closed"] = "available"

    def __post_init__(self) -> None:
        if self.status not in ("available", "missing", "closed"):
            raise ValueError("Unknown traffic observation status.")
        if self.observed_at is not None:
            object.__setattr__(self, "observed_at", normalize_time(self.observed_at))
        elif self.status != "missing":
            raise ValueError("Available observations and closures require an observation time.")


@dataclass(frozen=True)
class TrafficSnapshot:
    request_time: datetime
    observations: Mapping[str, TrafficObservation]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_time", normalize_time(self.request_time))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("A traffic source label is required.")
        observations = dict(self.observations)
        for road_id, observation in observations.items():
            validate_id(road_id)
            if not isinstance(observation, TrafficObservation):
                raise ValueError("Snapshot entries must be TrafficObservation values.")
            if observation.observed_at is not None and observation.observed_at > self.request_time:
                raise ValueError("Traffic snapshots cannot include future observations.")
        object.__setattr__(self, "observations", MappingProxyType(observations))

    def age_seconds(self, road_id: str) -> float | None:
        observation = self.observations.get(road_id)
        if observation is None or observation.observed_at is None:
            return None
        return (self.request_time - observation.observed_at).total_seconds()


@dataclass(frozen=True)
class TrafficDiagnostics:
    source: str | None
    observed_roads: tuple[str, ...]
    missing_roads: tuple[str, ...]
    invalid_speed_roads: tuple[str, ...]
    fallback_roads: tuple[str, ...]
    capped_roads: tuple[str, ...]
    closed_roads: tuple[str, ...]


def valid_speed(value: float | None) -> bool:
    return is_finite_number(value) and value > 0


def routing_speeds(
    network: RoadNetwork, snapshot: TrafficSnapshot | None, *, use_traffic: bool,
) -> tuple[dict[str, float], TrafficDiagnostics]:
    """Apply shared closures and, when enabled, use capped observed speeds."""
    observations = snapshot.observations if snapshot is not None else {}
    unknown = observations.keys() - network.roads.keys()
    if unknown:
        raise ValueError(f"Snapshot includes unknown road IDs: {sorted(unknown)!r}.")
    speeds = {}
    observed, missing, invalid, fallback, capped, closed = [], [], [], [], [], []
    for road_id, road in network.roads.items():
        observation = observations.get(road_id)
        has_speed = False
        if observation is not None and observation.status == "closed":
            closed.append(road_id)
            continue
        speed = road.normal_speed_mps
        if observation is None or observation.status == "missing":
            missing.append(road_id)
        elif not valid_speed(observation.speed_mps):
            invalid.append(road_id)
        else:
            has_speed = True
            observed.append(road_id)
            if use_traffic:
                if observation.speed_mps > speed:
                    capped.append(road_id)
                speed = min(speed, observation.speed_mps)
        if use_traffic and not has_speed:
            fallback.append(road_id)
        speeds[road_id] = speed
    return speeds, TrafficDiagnostics(
        snapshot.source if snapshot is not None else None,
        tuple(observed), tuple(missing), tuple(invalid), tuple(fallback), tuple(capped), tuple(closed),
    )


def build_graph(
    roads: Iterable[tuple[str, str, float, float]],
    traffic_speeds: Mapping[tuple[str, str], float] | None = None,
) -> dict[str, list[tuple[str, float]]]:
    """
    Build a legacy junction graph for the original routing demos.

    Road lengths are measured in meters.
    Road speeds are measured in meters per second.
    Travel times are measured in seconds.
    """

    graph = {}

    if traffic_speeds is None:
        traffic_speeds = {}

    for start, end, length, normal_speed in roads:
        length = positive_number(length, "Road length")
        normal_speed = positive_number(normal_speed, "Normal speed")

        # Use normal speed when traffic data is unavailable
        speed = traffic_speeds.get(
            (start, end), normal_speed
        )

        # Handle invalid or missing speed observations
        if not valid_speed(speed):
            speed = normal_speed

        # Assume vehicles cannot exceed the normal speed limit
        speed = min(speed, normal_speed)

        # Calculate estimated road travel time
        travel_time = checked_cost(length / speed)

        graph.setdefault(start, []).append(
            (end, travel_time)
        )

        graph.setdefault(end, [])

    return graph


if __name__ == "__main__":
    from routing import astar, path_cost

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
    static_actual_cost = path_cost(traffic_graph, static_path)

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
