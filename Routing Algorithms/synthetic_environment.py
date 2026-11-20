"""A small fictional outcome model for routing experiments, independent of planning."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import random
from types import MappingProxyType
from typing import Literal

from road_network import Road, RoadNetwork, RoadPosition, Turn, build_road_query, positive_number
from route_planner import RoutingResult
from routing import checked_cost
from synthetic_traffic import CONGESTION_MULTIPLIERS, SyntheticTraffic
from traffic_aware import TrafficSnapshot


START_ROAD = "synthetic_start"
SHORT_ROAD = "synthetic_short"
PICKUP_ROADS = ("synthetic_pickup_east", "synthetic_pickup_north")
REFERENCE_TIME = datetime(2024, 8, 1, 12, tzinfo=timezone.utc)
SPLITS = ("train", "validation", "test")


def synthetic_network() -> RoadNetwork:
    """Three corridors between two junctions and two distinct pickup roads."""
    return RoadNetwork(
        [
            Road(START_ROAD, "A", "B", 40, 10),
            Road(SHORT_ROAD, "B", "C", 90, 10),
            Road("synthetic_parallel", "B", "C", 138, 10),
            Road("synthetic_bypass_in", "B", "D", 78, 10),
            Road("synthetic_bypass_out", "D", "C", 78, 10),
            Road(PICKUP_ROADS[0], "C", "E", 60, 10),
            Road(PICKUP_ROADS[1], "C", "N", 75, 10),
        ],
        [
            Turn(START_ROAD, SHORT_ROAD),
            Turn(START_ROAD, "synthetic_parallel"),
            Turn(START_ROAD, "synthetic_bypass_in"),
            Turn("synthetic_bypass_in", "synthetic_bypass_out"),
            *[
                Turn(road, pickup)
                for road in (SHORT_ROAD, "synthetic_parallel", "synthetic_bypass_out")
                for pickup in PICKUP_ROADS
            ],
        ],
    )


def derive_seed(master: int, *parts: str | int) -> int:
    """Derive stable seed streams without relying on Python's salted hash."""
    digest = sha256(json.dumps((master, *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class OutcomeModel:
    forecast_noise: float = 0.18
    realized_speed_noise: float = 0.22
    max_road_delay_s: float = 3.0
    max_short_turn_delay_s: float = 14.0
    missing_probability: float = 0.2
    hidden_closure_probability: float = 0.12
    dispatch_delay_s: float = 2.0
    timeout_s: float = 65.0

    def __post_init__(self) -> None:
        for label in ("forecast_noise", "realized_speed_noise"):
            if checked_cost(getattr(self, label), label) >= 1:
                raise ValueError(f"{label} must be below 1.")
        for label in ("missing_probability", "hidden_closure_probability"):
            if checked_cost(getattr(self, label), label) > 1:
                raise ValueError(f"{label} must be at most 1.")
        for label in ("max_road_delay_s", "max_short_turn_delay_s", "dispatch_delay_s"):
            checked_cost(getattr(self, label), label)
        positive_number(self.timeout_s, "Pickup timeout")


@dataclass(frozen=True)
class PickupRequest:
    scenario_id: str
    start: RoadPosition
    pickup: RoadPosition
    traffic: TrafficSnapshot
    context: str


@dataclass(frozen=True)
class RealizedConditions:
    speeds_mps: Mapping[str, float]
    road_delays_s: Mapping[str, float]
    turn_delays_s: Mapping[tuple[str, str], float]
    closed_roads: frozenset[str]
    dispatch_delay_s: float
    timeout_s: float


@dataclass(frozen=True)
class Scenario:
    split: str
    pair_seed: int
    split_seed: int
    scenario_seed: int
    episode: int
    condition: str
    pattern: str
    request: PickupRequest
    realized: RealizedConditions


@dataclass(frozen=True)
class PickupOutcome:
    """The contract shared by synthetic evaluation and a future SUMO callback."""

    status: Literal["completed", "unreachable", "blocked", "timeout"]
    pickup_time_s: float | None = None
    route_travel_s: float | None = None

    def __post_init__(self) -> None:
        if self.status == "completed":
            checked_cost(self.pickup_time_s, "Pickup time")
            if self.route_travel_s is not None:
                if checked_cost(self.route_travel_s, "Route time") > self.pickup_time_s:
                    raise ValueError("Route time must fit within the request-to-pickup time.")
        elif self.status in ("unreachable", "blocked", "timeout"):
            if self.pickup_time_s is not None or self.route_travel_s is not None:
                raise ValueError("A failed pickup cannot supply completion times.")
        else:
            raise ValueError(f"Unknown pickup outcome: {self.status!r}.")


def _traffic_pair(
    network: RoadNetwork,
    when: datetime,
    condition: str,
    pattern: str,
    pickup_road: str,
    scenario_seed: int,
    model: OutcomeModel,
) -> tuple[TrafficSnapshot, RealizedConditions]:
    nominal = {road_id: 1.0 for road_id in network.roads}
    nominal[SHORT_ROAD] = CONGESTION_MULTIPLIERS[condition]
    forecast = nominal.copy()
    realized = nominal.copy()
    missing, closed, hidden_closed = (), (), ()
    roads, turns = {}, {}
    dispatch = 0.0
    if pattern == "noisy":
        observation_rng = random.Random(derive_seed(scenario_seed, "forecast"))
        realized_rng = random.Random(derive_seed(scenario_seed, "realized"))

        def vary(factor: float, spread: float, rng: random.Random) -> float:
            return min(1.0, factor * rng.uniform(1 - spread, 1 + spread))

        forecast = {road: vary(factor, model.forecast_noise, observation_rng)
                    for road, factor in nominal.items()}
        realized = {road: vary(factor, model.realized_speed_noise, realized_rng)
                    for road, factor in nominal.items()}
        if observation_rng.random() < model.missing_probability:
            missing = (SHORT_ROAD,)
        if realized_rng.random() < model.hidden_closure_probability:
            hidden_closed = (SHORT_ROAD,)
        roads = {road: realized_rng.uniform(0, model.max_road_delay_s) for road in nominal}
        turns = {(SHORT_ROAD, pickup_road): realized_rng.uniform(0, model.max_short_turn_delay_s)}
        dispatch = model.dispatch_delay_s
    elif pattern == "known_closure":
        closed = (pickup_road,)

    available = {road: factor for road, factor in forecast.items() if road not in (*missing, *closed)}
    traffic = SyntheticTraffic(
        network, condition, affected_roads=(), road_multipliers=available,
        closed_roads=closed, missing_roads=missing, pattern=pattern,
    )(when)
    outcome = RealizedConditions(
        MappingProxyType({road: network.roads[road].normal_speed_mps * factor
                          for road, factor in realized.items()}),
        MappingProxyType(roads), MappingProxyType(turns), frozenset((*closed, *hidden_closed)),
        dispatch, model.timeout_s,
    )
    return traffic, outcome


def _forecast_context(
    network: RoadNetwork, traffic: TrafficSnapshot, start: RoadPosition, pickup: RoadPosition,
) -> str:
    observation = traffic.observations[SHORT_ROAD]
    if observation.status == "available":
        ratio = observation.speed_mps / network.roads[SHORT_ROAD].normal_speed_mps
        band = next(label for boundary, label in ((0.9, "free_flow"), (0.65, "light"),
                                                   (0.35, "moderate"), (0, "heavy"))
                    if ratio >= boundary)
    else:
        band = observation.status
    return (f"{start.road_id}@{start.position_m:g}/{pickup.road_id}@{pickup.position_m:g}/"
            f"{traffic.source}/{band}")


def make_scenarios(
    network: RoadNetwork,
    levels: Sequence[str],
    split: str,
    size: int,
    pair_seed: int,
    model: OutcomeModel,
) -> tuple[Scenario, ...]:
    """Shuffle complete template cycles, then realize every episode before planning."""
    if split not in SPLITS:
        raise ValueError(f"Unknown experiment split: {split!r}.")
    if type(size) is not int or size < 1 or type(pair_seed) is not int:
        raise ValueError("Scenario size must be positive and the pair seed an integer.")
    if not levels or len(set(levels)) != len(levels) or any(
        level not in CONGESTION_MULTIPLIERS for level in levels
    ):
        raise ValueError("Supply distinct, recognized congestion levels.")

    split_seed = derive_seed(pair_seed, "scenario", split)
    rng = random.Random(split_seed)
    templates = [
        (pattern, level, pickup) for level in levels
        for pattern in ("perfect", "noisy") for pickup in PICKUP_ROADS
    ]
    templates.append(("known_closure", levels[0], PICKUP_ROADS[1]))
    order = []
    while len(order) < size:
        cycle = templates.copy()
        rng.shuffle(cycle)
        order.extend(cycle)
    scenarios = []
    for episode, (pattern, level, pickup) in enumerate(order[:size], start=1):
        scenario_id = f"{split}-{pair_seed}-{episode:04d}"
        scenario_seed = derive_seed(split_seed, episode)
        # Distinct timestamps encode split membership, without driving the RNG.
        when = REFERENCE_TIME + timedelta(days=7 * SPLITS.index(split), seconds=episode)
        traffic, realized = _traffic_pair(network, when, level, pattern, pickup, scenario_seed, model)
        start_position = network.position(START_ROAD, 10)
        pickup_position = network.position(pickup, 30)
        request = PickupRequest(
            scenario_id, start_position, pickup_position, traffic,
            _forecast_context(network, traffic, start_position, pickup_position),
        )
        scenarios.append(
            Scenario(split, pair_seed, split_seed, scenario_seed, episode, level, pattern, request, realized),
        )
    return tuple(scenarios)


def evaluate_synthetic(
    network: RoadNetwork, scenario: Scenario, result: RoutingResult,
) -> PickupOutcome:
    """Score one already-chosen route with the same hidden conditions for every policy."""
    route = result.road_ids
    if route is None:
        return PickupOutcome("unreachable")
    hidden = scenario.realized
    if hidden.closed_roads.intersection(route):
        return PickupOutcome("blocked")
    request = scenario.request
    query = build_road_query(network, request.start, request.pickup, hidden.speeds_mps, "passenger")
    drive_s = query.travel_time(route)
    road_delay = sum(hidden.road_delays_s.get(road, 0.0) for road in route)
    turn_delay = sum(hidden.turn_delays_s.get(turn, 0.0) for turn in zip(route, route[1:]))
    route_s = checked_cost(drive_s + road_delay + turn_delay, "Realized route time")
    pickup_s = checked_cost(hidden.dispatch_delay_s + route_s, "Modeled pickup time")
    if pickup_s > hidden.timeout_s:
        return PickupOutcome("timeout")
    return PickupOutcome("completed", pickup_s, route_s)
