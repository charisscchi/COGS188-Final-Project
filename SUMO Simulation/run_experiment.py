"""Compare the three upstream planners using pickup times measured by SUMO."""

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from sumo_adapter import NETWORK, OSM, ROOT, STEP_S, SumoDriver, read_network, run_binary

from epsilon_greedy import EpsilonGreedy
from experiment import FrozenSelector, TrainingState, run_episode
from route_planner import plan_route
from synthetic_environment import PickupRequest
from traffic_aware import TrafficObservation, TrafficSnapshot

CONFIG = ROOT / "scenarios.json"
RESULTS = ROOT / "results"
UPSTREAM_COMMIT = "5618dfbed7508d99b10a50c836fdaef4e31450e4"
EPOCH = datetime(2024, 8, 1, 16, tzinfo=timezone.utc)
METHODS = ("static", "traffic_aware", "epsilon_greedy")


@dataclass(frozen=True)
class Scenario:
    id: str
    split: str
    context: str
    condition: str
    seed: int
    request_s: float
    timeout_s: float
    start_edge: str
    start_m: float
    pickup_edge: str
    pickup_m: float
    bottleneck_edges: tuple[str, ...]
    forecast_factor: float
    actual_factor: float
    background: tuple[tuple[float, tuple[str, ...]], ...]

    def routing_request(self, roads):
        request_time = EPOCH + timedelta(seconds=self.request_s)
        observations = {
            edge: TrafficObservation(roads.roads[edge].normal_speed_mps * self.forecast_factor, request_time)
            for edge in self.bottleneck_edges
        }
        return PickupRequest(
            self.id, roads.position(self.start_edge, self.start_m),
            roads.position(self.pickup_edge, self.pickup_m),
            TrafficSnapshot(request_time, observations, "synthetic SUMO dispatch forecast"), self.context,
        )


def make_scenarios(config, roads, lanes):
    # Generate the same background routes before any policy runs.
    background_routes = {}
    for pickup in config["pickups"]:
        for edge_key, position_key in (("start_edge", "start_m"), ("pickup_edge", "pickup_m")):
            edge = pickup[edge_key]
            if len(lanes[edge]) != 1 or not 5 <= pickup[position_key] < lanes[edge][0].getLength() - 5:
                raise ValueError(f"{edge} must have one accessible lane and a position inside it")
        plan = plan_route(
            roads, pickup["start_edge"], pickup["pickup_edge"], request_time=EPOCH,
            method="epsilon_greedy", bandit=EpsilonGreedy(), k=config["candidate_routes"],
            start_position_m=pickup["start_m"], pickup_position_m=pickup["pickup_m"],
        )
        if len(plan.candidates) < 2:
            raise ValueError(f"{pickup['id']} must have multiple legal routes")
        background_routes[pickup["id"]] = [candidate.road_ids for candidate in plan.candidates]

    for split, count in config["requests_per_context"].items():
        rng = random.Random(config["split_seeds"][split])
        request_s = config["request_s"]
        for pickup in config["pickups"]:
            for condition, parameters in config["conditions"].items():
                context = f"{pickup['id']}/{condition}"
                for index in range(1, count + 1):
                    seed = rng.randrange(1, 2**31)
                    demand = random.Random(seed)
                    factor = demand.uniform(*parameters["actual_range"])
                    background = tuple(
                        (demand.randrange(0, int(2 * request_s / STEP_S)) * STEP_S,
                         demand.choice(background_routes[pickup["id"]]))
                        for _ in range(demand.randint(*parameters["background_range"]))
                    )
                    yield Scenario(
                        f"{split}-{pickup['id']}-{condition}-{index:02d}", split, context, condition, seed,
                        request_s, config["timeout_s"], pickup["start_edge"], pickup["start_m"],
                        pickup["pickup_edge"], pickup["pickup_m"], tuple(pickup["bottleneck_edges"]),
                        parameters["forecast_factor"], factor, background,
                    )


def evaluate(driver, roads, scenario, method, k, *, selector=None, cache=None, replay_dir=None):
    measured = None

    def callback(_request, plan):
        nonlocal measured
        if cache is not None and plan.road_ids in cache:
            measured = cache[plan.road_ids]
        else:
            measured = driver.drive_in_sumo(scenario, plan.road_ids, replay_dir=replay_dir)
            if cache is not None:
                cache[plan.road_ids] = measured
        return measured.outcome

    episode = run_episode(roads, scenario.routing_request(roads), method, callback, selector=selector, k=k)
    plan = episode.plan
    row = {
        "request_id": scenario.id, "split": scenario.split, "context": scenario.context,
        "traffic_condition": scenario.condition, "sumo_seed": scenario.seed, "method": method,
        "request_s": scenario.request_s, "departure_s": measured.depart_s, "pickup_s": measured.arrival_s,
        "status": episode.outcome.status, "pickup_wait_s": episode.outcome.pickup_time_s,
        "drive_s": episode.outcome.route_travel_s, "stopped_in_traffic_s": measured.stopped_in_traffic_s,
        "distance_m": measured.distance_m, "planning_ms": round(episode.routing_time_s * 1000, 6),
        "start_edge": scenario.start_edge, "start_pos_m": scenario.start_m,
        "pickup_edge": scenario.pickup_edge, "pickup_pos_m": scenario.pickup_m,
        "arrived_lane": measured.arrival_lane, "arrived_pos_m": measured.arrival_position_m,
        "route_edges": " ".join(plan.road_ids or ()), "estimated_drive_s": plan.estimated_time_s,
    }
    return episode, row


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows):
    completed = [row for row in rows if row["status"] == "completed"]
    return {
        "requests": len(rows), "completed": len(completed),
        "completion_rate": len(completed) / len(rows),
        "mean_pickup_wait_s": mean(row["pickup_wait_s"] for row in completed) if completed else None,
        "mean_drive_s": mean(row["drive_s"] for row in completed) if completed else None,
        "mean_stopped_in_traffic_s": mean(row["stopped_in_traffic_s"] for row in completed) if completed else None,
        "mean_planning_ms": mean(row["planning_ms"] for row in rows),
    }


def run_smoke(driver, roads, config, output):
    pickup = config["pickups"][0]
    scenario = Scenario(
        "smoke-single-taxi", "smoke", pickup["id"], "empty_network", 188,
        0, config["timeout_s"], pickup["start_edge"], pickup["start_m"], pickup["pickup_edge"], pickup["pickup_m"],
        (), 1.0, 1.0, (),
    )
    _, row = evaluate(driver, roads, scenario, "static", config["candidate_routes"], replay_dir=output)
    if row["status"] != "completed":
        raise RuntimeError(f"The single-taxi verification failed: {row}")
    write_rows(output / "measurement.csv", [row])
    print(f"Smoke: taxi departed at {row['departure_s']} s and reached {row['arrived_lane']} "
          f"at {row['pickup_s']} s ({row['arrived_pos_m']} m).", flush=True)


def run_experiment(driver, roads, config, scenarios, output):
    epsilons = config["epsilons"]
    k = config["candidate_routes"]
    states = {}
    training = []
    validation = []
    testing = []
    for scenario in scenarios:
        if scenario.split != "train":
            continue
        cache = {}
        for epsilon in epsilons:
            key = (scenario.context, epsilon)
            if key not in states:
                states[key] = TrainingState(EpsilonGreedy(epsilon=epsilon, seed=scenario.seed), scenario.seed)
            state = states[key]
            episode, row = evaluate(driver, roads, scenario, "epsilon_greedy", k, selector=state.agent, cache=cache)
            mode, reward = state.record(episode)
            training.append({**row, "epsilon": epsilon, "selection_mode": mode, "reward_s": reward})
    write_rows(output / "training.csv", training)
    print(f"Training: {len(training)} epsilon-greedy decisions using SUMO feedback.", flush=True)

    for scenario in scenarios:
        if scenario.split != "validation":
            continue
        cache = {}
        for epsilon in epsilons:
            selector = states[scenario.context, epsilon].frozen()
            _, row = evaluate(driver, roads, scenario, "epsilon_greedy", k, selector=selector, cache=cache)
            validation.append({**row, "epsilon": epsilon, "selection_mode": selector.mode})
    scores = [{"epsilon": epsilon, **summary([row for row in validation if row["epsilon"] == epsilon])}
              for epsilon in epsilons]
    best = min(scores, key=lambda score: (-score["completed"],
                                         score["mean_pickup_wait_s"] if score["completed"] else float("inf"),
                                         score["epsilon"]))
    winner = best["epsilon"]
    write_rows(output / "validation.csv", validation)
    write_rows(output / "validation_summary.csv", scores)
    print(f"Validation selected epsilon={winner:g}.", flush=True)

    for scenario in scenarios:
        if scenario.split != "test":
            continue
        selector: FrozenSelector = states[scenario.context, winner].frozen()
        for method in METHODS:
            _, row = evaluate(driver, roads, scenario, method, k,
                              selector=selector if method == "epsilon_greedy" else None)
            testing.append({**row, "selection_mode": selector.mode if method == "epsilon_greedy" else ""})
    write_rows(output / "test.csv", testing)

    groups = [("all", testing)]
    groups.extend((condition, [row for row in testing if row["traffic_condition"] == condition])
                  for condition in config["conditions"])
    report = []
    for condition, rows in groups:
        baseline = {row["request_id"]: row for row in rows if row["method"] == "static"}
        for method in METHODS:
            method_rows = [row for row in rows if row["method"] == method]
            pairs = [(row, baseline[row["request_id"]]) for row in method_rows
                     if row["status"] == baseline[row["request_id"]]["status"] == "completed"]
            delta = mean(row["pickup_wait_s"] - static["pickup_wait_s"] for row, static in pairs) if pairs else None
            report.append({"traffic_condition": condition, "method": method, **summary(method_rows),
                           "joint_completions_vs_static": len(pairs), "paired_wait_delta_vs_static_s": delta})
    write_rows(output / "summary.csv", report)

    learned = [{"context": context, "epsilon": epsilon, "route_edges": " ".join(route),
                "training_completions": count, "mean_reward_s": reward if count else None}
               for (context, epsilon), state in states.items()
               for route, (count, reward) in sorted(state.estimates.items())]
    write_rows(output / "learned_routes.csv", learned)
    return winner, len(testing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true", help="Run and save the single-taxi scenario")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    _, roads, lanes = read_network()
    driver = SumoDriver(lanes)
    run_smoke(driver, roads, config, output / "smoke")
    if args.smoke_only:
        return
    scenarios = list(make_scenarios(config, roads, lanes))
    with (output / "scenarios.jsonl").open("w", encoding="utf-8") as handle:
        for scenario in scenarios:
            handle.write(json.dumps(vars(scenario)) + "\n")
    winner, test_runs = run_experiment(driver, roads, config, scenarios, output)
    manifest = {
        "sumo_version": run_binary("sumo", "--version").stdout.splitlines()[0],
        "python_version": sys.version.split()[0], "upstream_commit": UPSTREAM_COMMIT,
        "network_edges": len(roads.roads), "legal_turns": sum(map(len, roads.outgoing.values())),
        "osm_sha256": hashlib.sha256(OSM.read_bytes()).hexdigest(),
        "network_sha256": hashlib.sha256(NETWORK.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "traffic_source": "synthetic speed limits plus generated background vehicles",
        "selected_epsilon": winner, "test_sumo_runs": test_runs,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Testing: {test_runs} SUMO runs. Results in {output}", flush=True)


if __name__ == "__main__":
    main()
