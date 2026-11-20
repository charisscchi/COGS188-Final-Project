"""Paired preliminary synthetic experiments with separate training and evaluation."""

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean
from time import perf_counter_ns
from types import MappingProxyType

from epsilon_greedy import EpsilonGreedy, RouteID
from road_network import RoadNetwork
from route_planner import RouteSelector, RoutingMethod, RoutingResult, plan_route
from routing import checked_cost
from synthetic_environment import (
    SPLITS, OutcomeModel, PickupOutcome, PickupRequest, Scenario, derive_seed,
    evaluate_synthetic, make_scenarios, synthetic_network,
)
from synthetic_traffic import CONGESTION_MULTIPLIERS


LABEL = "preliminary synthetic"
VERSION = "directed-corridors-v1"
OutcomeCallback = Callable[[PickupRequest, RoutingResult], PickupOutcome]


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    train_episodes: int = 119
    validation_scenarios: int = 51
    test_scenarios: int = 68
    levels: tuple[str, ...] = tuple(CONGESTION_MULTIPLIERS)
    epsilons: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.50)
    k: int = 3
    first_candidate: bool = True
    model: OutcomeModel = field(default_factory=OutcomeModel)

    def __post_init__(self) -> None:
        if not self.seeds or any(type(seed) is not int for seed in self.seeds):
            raise ValueError("Supply one or more integer pair seeds.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Pair seeds must be distinct.")
        for size in (self.train_episodes, self.validation_scenarios, self.test_scenarios, self.k):
            if type(size) is not int or size < 1:
                raise ValueError("Split sizes and k must be positive integers.")
        if not self.levels or len(set(self.levels)) != len(self.levels) or any(
            level not in CONGESTION_MULTIPLIERS for level in self.levels
        ):
            raise ValueError("Supply distinct, recognized congestion levels.")
        if not self.epsilons or any(checked_cost(epsilon, "Epsilon") > 1 for epsilon in self.epsilons):
            raise ValueError("Supply epsilon values in [0, 1].")
        if len(set(self.epsilons)) != len(self.epsilons):
            raise ValueError("Epsilon values must be distinct.")

    @property
    def experiment_id(self) -> str:
        manifest = json.dumps([VERSION, asdict(self)], sort_keys=True)
        return f"synthetic-{sha256(manifest.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class EpisodeResult:
    plan: RoutingResult
    outcome: PickupOutcome
    routing_time_s: float


def run_episode(
    network: RoadNetwork,
    request: PickupRequest,
    method: RoutingMethod,
    outcome_callback: OutcomeCallback,
    *,
    selector: RouteSelector | None = None,
    k: int = 3,
    vehicle_class: str = "passenger",
) -> EpisodeResult:
    """Plan with observable inputs, then ask an evaluator for the actual outcome."""
    started = perf_counter_ns()
    plan = plan_route(
        network, request.start.road_id, request.pickup.road_id,
        request_time=request.traffic.request_time, method=method,
        start_position_m=request.start.position_m, pickup_position_m=request.pickup.position_m,
        traffic=request.traffic, k=k, bandit=selector, vehicle_class=vehicle_class,
    )
    routing_time_s = (perf_counter_ns() - started) / 1e9
    return EpisodeResult(plan, outcome_callback(request, plan), routing_time_s)


class FrozenSelector:
    """Greedy with a read-only copy of trained estimates for sampled routes."""

    def __init__(self, estimates: Mapping[RouteID, float] | None = None) -> None:
        self.estimates = MappingProxyType(dict(estimates or {}))
        self.routes: tuple[RouteID, ...] = ()
        self.mode = "unreachable"

    def set_candidates(self, routes: Iterable[Sequence[str]]) -> None:
        self.routes = tuple(tuple(route) for route in routes)

    def choose_route(self) -> int:
        sampled = [index for index, route in enumerate(self.routes) if route in self.estimates]
        if sampled:
            self.mode = "frozen_greedy"
            # max keeps the first traffic-ranked candidate at equal reward.
            return max(sampled, key=lambda index: self.estimates[self.routes[index]])
        self.mode = "first_candidate_fallback"
        return 0


@dataclass
class TrainingState:
    agent: EpsilonGreedy
    seed: int
    estimates: dict[RouteID, tuple[int, float]] = field(default_factory=dict)
    cumulative_reward: float = 0.0

    def record(self, result: EpisodeResult) -> tuple[str, float | None]:
        if result.plan.road_ids is None:
            return "unreachable", None
        initial = any(self.estimates.get(option.road_ids, (0, 0))[0] == 0
                      for option in result.plan.candidates)
        reward = None
        if result.outcome.status == "completed":
            wait = result.outcome.pickup_time_s
            self.agent.update_route(result.plan.road_ids, wait)
            reward = -wait
            self.cumulative_reward += reward
        self.estimates.update({
            route: (count, value) for route, count, value in
            zip(self.agent.routes, self.agent.counts, self.agent.q_values, strict=True)
        })
        return ("initial_trial" if initial else "epsilon_rule"), reward

    def frozen(self) -> FrozenSelector:
        return FrozenSelector({route: value for route, (count, value) in self.estimates.items() if count})


@dataclass(frozen=True)
class Episode:
    experiment_id: str
    label: str
    scenario_id: str
    pair_seed: int
    split_seed: int
    scenario_seed: int
    split: str
    episode: int
    method: str
    condition: str
    pattern: str
    context: str
    request_time: str
    epsilon: float | None
    agent_seed: int | None
    selection_mode: str
    selected_route: RouteID | None
    candidate_routes: tuple[RouteID, ...]
    candidate_count: int
    fallback_road_count: int
    estimated_travel_s: float | None
    modeled_pickup_s: float | None
    realized_route_s: float | None
    status: str
    routing_time_s: float
    reward: float | None
    cumulative_context_reward: float | None


@dataclass(frozen=True)
class LearnedRoute:
    experiment_id: str
    label: str
    pair_seed: int
    epsilon: float
    context: str
    agent_seed: int
    road_ids: RouteID
    reward_samples: int
    mean_reward: float | None


@dataclass(frozen=True)
class ExperimentRun:
    config: ExperimentConfig
    chosen_epsilon: float
    episodes: tuple[Episode, ...]
    learned_routes: tuple[LearnedRoute, ...]


def pick_epsilon(episodes: Iterable[Episode], epsilons: Sequence[float]) -> float:
    """Prefer completion rate, then completed-only mean pickup time, then lower epsilon."""
    records = [row for row in episodes if row.split == "validation" and row.method == "epsilon_greedy"]

    def score(epsilon: float) -> tuple[float, float, float]:
        rows = [row for row in records if row.epsilon == epsilon]
        if not rows:
            raise ValueError(f"Missing validation episodes for epsilon {epsilon}.")
        waits = [row.modeled_pickup_s for row in rows if row.status == "completed"]
        return (-len(waits) / len(rows), mean(waits) if waits else math.inf, epsilon)

    return min(epsilons, key=score)


def run_experiment(config: ExperimentConfig) -> ExperimentRun:
    network = synthetic_network()
    rows: list[Episode] = []
    training: dict[tuple[int, float, str], TrainingState] = {}
    experiment_id = config.experiment_id

    def record(
        scenario: Scenario, method: str, *, epsilon: float | None = None,
        state: TrainingState | None = None, learn: bool = False,
    ) -> None:
        request = scenario.request
        selector = None
        agent_seed = state.seed if state else None
        mode, reward, cumulative = "astar", None, None
        if method == "epsilon_greedy":
            if learn:
                selector = state.agent
            else:
                selector = state.frozen() if state else FrozenSelector()
        elif method == "first_candidate":
            selector = FrozenSelector()

        def score_route(_request: PickupRequest, plan: RoutingResult) -> PickupOutcome:
            return evaluate_synthetic(network, scenario, plan)

        result = run_episode(
            network, request, "epsilon_greedy" if selector is not None else method,
            score_route, selector=selector, k=config.k,
        )
        if learn:
            mode, reward = state.record(result)
            cumulative = state.cumulative_reward
        elif result.plan.road_ids is None:
            mode = "unreachable"
        elif method == "first_candidate":
            mode = "first_candidate"
        elif method == "epsilon_greedy":
            mode = selector.mode
            if mode == "first_candidate_fallback":
                mode = "unseen_context_fallback" if state is None else "unsampled_candidates_fallback"

        plan, outcome = result.plan, result.outcome
        rows.append(Episode(
            experiment_id, LABEL, request.scenario_id, scenario.pair_seed, scenario.split_seed,
            scenario.scenario_seed, scenario.split, scenario.episode, method, scenario.condition,
            scenario.pattern, request.context, request.traffic.request_time.isoformat(), epsilon,
            agent_seed, mode, plan.road_ids, tuple(option.road_ids for option in plan.candidates),
            len(plan.candidates), len(plan.diagnostics.fallback_roads), plan.estimated_time_s,
            outcome.pickup_time_s, outcome.route_travel_s, outcome.status, result.routing_time_s,
            reward, cumulative,
        ))

    def evaluate_split(scenarios: Iterable[Scenario], epsilons: Sequence[float]) -> None:
        for scenario in scenarios:
            for method in ("static", "traffic_aware"):
                record(scenario, method)
            if config.first_candidate:
                record(scenario, "first_candidate")
            for epsilon in epsilons:
                state = training.get((scenario.pair_seed, epsilon, scenario.request.context))
                record(scenario, "epsilon_greedy", epsilon=epsilon, state=state)

    for seed in config.seeds:
        for scenario in make_scenarios(network, config.levels, "train", config.train_episodes, seed, config.model):
            for epsilon in config.epsilons:
                context = scenario.request.context
                key = (seed, epsilon, context)
                if key not in training:
                    agent_seed = derive_seed(seed, "agent", context)
                    training[key] = TrainingState(EpsilonGreedy(epsilon=epsilon, seed=agent_seed), agent_seed)
                record(scenario, "epsilon_greedy", epsilon=epsilon, state=training[key], learn=True)
        scenarios = make_scenarios(network, config.levels, "validation", config.validation_scenarios, seed, config.model)
        evaluate_split(scenarios, config.epsilons)

    chosen = pick_epsilon(rows, config.epsilons)
    for seed in config.seeds:
        scenarios = make_scenarios(network, config.levels, "test", config.test_scenarios, seed, config.model)
        evaluate_split(scenarios, (chosen,))

    learned = tuple(
        LearnedRoute(experiment_id, LABEL, seed, epsilon, context, state.seed, route,
                     count, value if count else None)
        for (seed, epsilon, context), state in sorted(training.items())
        for route, (count, value) in sorted(state.estimates.items())
    )
    return ExperimentRun(config, chosen, tuple(rows), learned)


def main() -> None:
    from experiment_outputs import prepare_output_dir, write_outputs

    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description="Run paired preliminary synthetic robotaxi routing experiments.")
    parser.add_argument("--seeds", type=int, nargs="+", default=defaults.seeds)
    parser.add_argument("--train-episodes", type=int, default=defaults.train_episodes)
    parser.add_argument("--validation-scenarios", type=int, default=defaults.validation_scenarios)
    parser.add_argument("--test-scenarios", type=int, default=defaults.test_scenarios)
    parser.add_argument("--levels", choices=CONGESTION_MULTIPLIERS, nargs="+", default=defaults.levels)
    parser.add_argument("--epsilons", type=float, nargs="+", default=defaults.epsilons)
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--timeout-s", type=float, default=defaults.model.timeout_s)
    parser.add_argument("--first-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "preliminary_synthetic")
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs and config without Matplotlib.")
    args = parser.parse_args()
    try:
        config = ExperimentConfig(
            tuple(args.seeds), args.train_episodes, args.validation_scenarios, args.test_scenarios,
            tuple(args.levels), tuple(args.epsilons), args.k, args.first_candidate,
            OutcomeModel(timeout_s=args.timeout_s),
        )
        output_dir = prepare_output_dir(args.output_dir, plots=not args.no_plots)
    except ValueError as exc:
        parser.error(str(exc))

    result = run_experiment(config)
    write_outputs(result, output_dir, plots=not args.no_plots)
    print(f"Preliminary synthetic experiment {config.experiment_id}")
    print(f"Validation selected epsilon {result.chosen_epsilon:g} for frozen greedy testing.")
    print(f"Wrote {len(result.episodes):,} policy episodes to {output_dir}")


if __name__ == "__main__":
    main()
