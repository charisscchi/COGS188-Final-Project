"""Configurable, reproducible traffic snapshots for fictional experiments."""

from collections.abc import Iterable, Mapping
from datetime import datetime
import random
from types import MappingProxyType

from road_network import RoadNetwork, positive_number, validate_id
from traffic_aware import TrafficObservation, TrafficSnapshot, normalize_time


CONGESTION_MULTIPLIERS = MappingProxyType({
    "free_flow": 1.0,
    "light": 0.8,
    "moderate": 0.5,
    "heavy": 0.2,
})


def road_id_set(network: RoadNetwork, ids: Iterable[str], label: str) -> frozenset[str]:
    if isinstance(ids, str):
        raise ValueError(f"{label} must be a collection of directed road IDs.")
    road_ids = set()
    for road_id in ids:
        validate_id(road_id)
        if road_id not in network.roads or road_id in road_ids:
            raise ValueError(f"{label} contains an unknown or repeated road ID: {road_id!r}.")
        road_ids.add(road_id)
    return frozenset(road_ids)


class SyntheticTraffic:
    """Callable provider with fixed affected roads and request-time observations."""

    def __init__(
        self,
        network: RoadNetwork,
        level: str = "free_flow",
        *,
        affected_roads: Iterable[str] | None = None,
        affected_count: int | None = None,
        seed: int = 42,
        road_multipliers: Mapping[str, float] | None = None,
        closed_roads: Iterable[str] = (),
        missing_roads: Iterable[str] = (),
        pattern: str = "configured",
    ) -> None:
        if level not in CONGESTION_MULTIPLIERS:
            raise ValueError(f"Unknown synthetic congestion level: {level!r}.")
        if type(seed) is not int:
            raise ValueError("The synthetic seed must be an integer.")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError("A synthetic pattern label is required.")
        self.network = network
        self.source = f"preliminary synthetic / {pattern}"
        self.closed = road_id_set(network, closed_roads, "Closed roads")
        self.missing = road_id_set(network, missing_roads, "Missing roads")
        if self.closed & self.missing:
            raise ValueError("A road cannot be both explicitly closed and missing.")
        unavailable = self.closed | self.missing

        if affected_roads is not None:
            if affected_count is not None:
                raise ValueError("Specify affected road IDs or a seeded count.")
            affected = road_id_set(network, affected_roads, "Affected roads")
            if affected & unavailable:
                raise ValueError("Affected roads must have available observations.")
        else:
            count = int(level != "free_flow") if affected_count is None else affected_count
            eligible = sorted(network.roads.keys() - unavailable)
            if type(count) is not int or not 0 <= count <= len(eligible):
                raise ValueError("Affected count must fit the number of eligible roads.")
            affected = frozenset(random.Random(seed).sample(eligible, count))

        multipliers = dict.fromkeys(affected, CONGESTION_MULTIPLIERS[level])
        if road_multipliers is not None:
            for road_id, value in road_multipliers.items():
                if road_id not in network.roads or road_id in unavailable:
                    raise ValueError(f"Multiplier road {road_id!r} must have an available observation.")
                multiplier = positive_number(value, "Synthetic speed multiplier")
                if multiplier > 1:
                    raise ValueError("Synthetic speed multipliers must be at most one.")
                multipliers[road_id] = multiplier
        self.affected_roads = affected
        self.multipliers = MappingProxyType(multipliers)

    def __call__(self, request_time: datetime) -> TrafficSnapshot:
        when = normalize_time(request_time)
        observations = {}
        for road_id, road in self.network.roads.items():
            if road_id in self.closed:
                observations[road_id] = TrafficObservation(None, when, "closed")
            elif road_id in self.missing:
                observations[road_id] = TrafficObservation(None, None, "missing")
            else:
                speed = positive_number(
                    road.normal_speed_mps * self.multipliers.get(road_id, 1.0), "Synthetic speed",
                )
                observations[road_id] = TrafficObservation(speed, when)
        return TrafficSnapshot(when, observations, self.source)
