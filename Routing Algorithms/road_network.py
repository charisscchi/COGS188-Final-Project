from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from routing import checked_cost, path_cost


def positive_number(value: float, label: str) -> float:
    number = checked_cost(value, label)
    if number == 0:
        raise ValueError(f"{label} must be positive.")
    return number


def validate_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Road, junction, and vehicle class IDs must be nonempty strings.")


def class_permissions(classes: frozenset[str] | None) -> frozenset[str] | None:
    if classes is None:
        return None
    if isinstance(classes, str):
        raise ValueError("Supply a set of allowed vehicle classes.")
    allowed = frozenset(classes)
    for vehicle_class in allowed:
        validate_id(vehicle_class)
    return allowed


@dataclass(frozen=True)
class Road:
    id: str
    from_junction: str
    to_junction: str
    length_m: float
    normal_speed_mps: float
    allowed_classes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        for identifier in (self.id, self.from_junction, self.to_junction):
            validate_id(identifier)
        object.__setattr__(self, "length_m", positive_number(self.length_m, "Road length"))
        object.__setattr__(self, "normal_speed_mps", positive_number(self.normal_speed_mps, "Normal speed"))
        object.__setattr__(self, "allowed_classes", class_permissions(self.allowed_classes))


@dataclass(frozen=True)
class Turn:
    from_road: str
    to_road: str
    allowed_classes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        validate_id(self.from_road)
        validate_id(self.to_road)
        object.__setattr__(self, "allowed_classes", class_permissions(self.allowed_classes))


@dataclass(frozen=True)
class RoadPosition:
    road_id: str
    position_m: float


class RoadNetwork:
    """Directed physical roads with explicitly allowed road-to-road movements."""

    def __init__(self, roads: Iterable[Road], turns: Iterable[Turn]) -> None:
        by_id: dict[str, Road] = {}
        for road in roads:
            if road.id in by_id:
                raise ValueError(f"Duplicate road ID {road.id!r}.")
            by_id[road.id] = road
        outgoing: dict[str, dict[str, Turn]] = {road_id: {} for road_id in by_id}
        for turn in turns:
            if turn.from_road not in by_id or turn.to_road not in by_id:
                raise ValueError("Turn references an unknown road.")
            if by_id[turn.from_road].to_junction != by_id[turn.to_road].from_junction:
                raise ValueError("Turn must connect roads at a shared junction in travel direction.")
            if turn.to_road in outgoing[turn.from_road]:
                raise ValueError("Supply one turn per directed road pair with its combined permissions.")
            outgoing[turn.from_road][turn.to_road] = turn

        # Canonical ordering makes equal-cost choices reproducible for the same graph.
        self.roads: Mapping[str, Road] = MappingProxyType(dict(sorted(by_id.items())))
        self.outgoing: Mapping[str, tuple[Turn, ...]] = MappingProxyType({
            road_id: tuple(turn for _, turn in sorted(outgoing[road_id].items()))
            for road_id in self.roads
        })

    def position(self, road_id: str, position_m: float | None, *, at_end: bool = False) -> RoadPosition:
        validate_id(road_id)
        if road_id not in self.roads:
            raise ValueError(f"Unknown road ID {road_id!r}.")
        length = self.roads[road_id].length_m
        if position_m is None:
            position_m = length if at_end else 0.0
        position_m = checked_cost(position_m, "Road position")
        if position_m > length:
            raise ValueError(f"Position exceeds the length of road {road_id!r}.")
        return RoadPosition(road_id, position_m)


class Endpoint(Enum):
    START = "start position"
    PICKUP = "pickup position"


SearchNode = str | Endpoint


@dataclass
class RoadQuery:
    graph: dict[SearchNode, list[tuple[SearchNode, float]]]
    start: RoadPosition
    pickup: RoadPosition

    def road_ids(self, path: Sequence[SearchNode]) -> tuple[str, ...]:
        if len(path) < 2 or path[0] != Endpoint.START or path[-1] != Endpoint.PICKUP:
            raise ValueError("Search path must link the query start and pickup.")
        roads = path[1:-1]
        if any(not isinstance(road, str) for road in roads):
            raise ValueError("Intermediate states must be road IDs.")
        return (self.start.road_id, *roads)

    def travel_time(self, road_ids: Sequence[str]) -> float:
        if not road_ids or road_ids[0] != self.start.road_id or road_ids[-1] != self.pickup.road_id:
            raise ValueError("Route must start and end on the query roads.")
        return path_cost(self.graph, [Endpoint.START, *road_ids[1:], Endpoint.PICKUP])


def build_road_query(
    network: RoadNetwork,
    start: RoadPosition,
    pickup: RoadPosition,
    speeds_mps: Mapping[str, float],
    vehicle_class: str,
) -> RoadQuery:
    """Use road-entry states plus the query's exact start and pickup positions.

    Roads absent from the speed mapping are unavailable. Each transition pays
    for the section of the current road traveled before the next state.
    """
    validate_id(vehicle_class)
    start = network.position(start.road_id, start.position_m)
    pickup = network.position(pickup.road_id, pickup.position_m)
    usable = {
        road_id: positive_number(speeds_mps[road_id], "Effective speed")
        for road_id, road in network.roads.items()
        if road_id in speeds_mps and (
            road.allowed_classes is None or vehicle_class in road.allowed_classes
        )
    }
    graph: dict[SearchNode, list[tuple[SearchNode, float]]] = {
        Endpoint.START: [], Endpoint.PICKUP: [],
    }
    query = RoadQuery(graph, start, pickup)
    if start.road_id not in usable or pickup.road_id not in usable:
        return query

    def seconds(road_id: str, distance_m: float) -> float:
        return checked_cost(distance_m / usable[road_id])

    if start.road_id == pickup.road_id and start.position_m <= pickup.position_m:
        graph[Endpoint.START].append((
            Endpoint.PICKUP, seconds(start.road_id, pickup.position_m - start.position_m),
        ))
        return query

    def departures(road_id: str, position_m: float) -> list[tuple[SearchNode, float]]:
        successors = [
            turn.to_road for turn in network.outgoing[road_id]
            if turn.to_road in usable
            and (turn.to_road != start.road_id or pickup.road_id == start.road_id)
            and (turn.allowed_classes is None or vehicle_class in turn.allowed_classes)
        ]
        remaining = network.roads[road_id].length_m - position_m
        return [(successor, seconds(road_id, remaining)) for successor in successors]

    graph[Endpoint.START] = departures(start.road_id, start.position_m)
    for road_id in usable:
        if road_id == pickup.road_id:
            graph[road_id] = [(Endpoint.PICKUP, seconds(road_id, pickup.position_m))]
        else:
            graph[road_id] = departures(road_id, 0.0)
    return query
