# Routing Algorithms

This UC San Diego COGS 188 project asks whether traffic-aware routing reduces passenger pickup waiting time in a simulated robotaxi system. The routing code produces route choices and travel-time estimates. SUMO will measure passenger pickup waiting time from request to pickup.

## Routing methods

| Method | How it chooses a route |
| --- | --- |
| Static A* | Minimizes estimated travel time at fixed normal road speeds. |
| Traffic-aware A* | Runs the same search with travel times based on a request-time traffic snapshot. |
| Epsilon-greedy | Selects among up to `k` legal routes ranked by estimated traffic-aware travel time. It learns route preferences from externally reported pickup waits. |

- **A*** orders states by `f(n) = g(n) + h(n)`. Here `g` is the best travel time found from the taxi to that state and `h` estimates the remaining travel time to this pickup. A cheaper path can reopen a state. An admissible heuristic satisfies `0 <= h(n) <= true minimum remaining cost`, gives zero at the goal, and uses the same units and cost model as the search. With `h = 0`, A* behaves like Dijkstra's algorithm. This is the default until a geographical lower bound can be verified. A future geometry-based callback can use projected distances and an upper speed bound once Tony verifies that the distances lower-bound the remaining travel distances under the selected graph model.
- **Travel time** is `time_s = distance_m / speed_mps`. Costs add across the exact directed roads traversed. The first and last roads contribute only the distance traveled on each. Turns currently have zero separate delay.
- **Epsilon-greedy** samples candidates with zero recorded rewards before using the ε rule. Once all active candidates have observations, it chooses a random candidate with probability ε and a highest-value candidate with probability `1 - ε`. Equal values are broken randomly. For a completed pickup, the reward is `R = -waiting_time_s` and the sample-average update is `Q(a) = Q(a) + (R - Q(a)) / N(a)`, where `N(a)` includes the current observation.

## Road network and positions

Build a `RoadNetwork(roads, turns)` from instances of `Road` and `Turn` in `road_network.py`.

- `Road(id, from_junction, to_junction, length_m, normal_speed_mps, allowed_classes=None)` describes one directed physical road. IDs are unique strings. Parallel roads have separate IDs. An explicitly modeled reverse road has its own ID.
- `Turn(from_road, to_road, allowed_classes=None)` authorizes travel from the end of the first road to the start of the second. Junctions must match and every permitted road pair needs an explicit turn. Provide one turn per pair with the combined lane-level class permissions. For roads and turns, `None` permits all classes and an empty set permits none.
- Positions are measured forward from the start of each directed road. They must lie within its length. The default taxi position is the beginning of its road and the default pickup position is the end of its road.
- For a pickup ahead on the same road, the path is that road alone and the distance is the difference in positions. Equal positions cost zero estimated travel seconds. A pickup behind the taxi requires a legal directed route that re-enters the pickup road. Such a route can contain its starting road ID twice.
- A road ID represents the state of entering that road. A virtual taxi start and a virtual pickup let the search account for partial roads. Any candidate avoids repeated road-entry states. Returning once to the start road for a pickup behind the taxi is possible because the taxi's initial position is a separate state.

## Planner interface

Use `plan_route(network, start_road_id, pickup_road_id, *, request_time, method, ...)` in `route_planner.py`.

| Input | Contract |
| --- | --- |
| `method` | `"static"`, `"traffic_aware"`, or `"epsilon_greedy"`. |
| `request_time` | A Python `datetime`. An aware input is converted to UTC. A naive input is interpreted as NYC local time. Ambiguous or nonexistent naive times raise `ValueError`. |
| `start_position_m`, `pickup_position_m` | Optional offsets along the directed roads. |
| `traffic` | A `TrafficSnapshot`, a callable receiving the request time in UTC and returning a snapshot, or `None`. Supply the same snapshot to each method if comparing them under known closures. |
| `vehicle_class` | Defaults to `"passenger"`. Use the SUMO class Tony assigns to the robotaxi. |
| `heuristic` | Optional A* callback `(current: RoadPosition, pickup: RoadPosition) -> float` estimating remaining seconds for this goal. Intermediate positions refer to road entries. The planner gives the terminal state zero. The caller is responsible for admissibility. |
| `k`, `bandit` | Positive integer candidate limit (default 3) and a caller-owned `EpsilonGreedy` agent or compatible route selector for the epsilon-greedy method. |

`RoutingResult` includes the method, `status`, ordered `road_ids`, `estimated_time_s`, `candidates`, and `diagnostics`. A result with `status == "unreachable"` has `None` for the route and estimated time. Invalid IDs, positions, traffic inputs, and method names raise `ValueError`. Candidate entries give the exact road IDs and travel-time estimate under that query's traffic costs. Equal costs are ordered reproducibly for the same pinned NetworkX version and road and turn data. The planner identifies candidates by their road tuples, so their learned values survive ranking changes.

### Minimal example with synthetic IDs

Run with the routing directory on the Python import path. The road IDs, lengths, and connections below are fictional.

```python
from datetime import datetime, timezone

from epsilon_greedy import EpsilonGreedy
from road_network import Road, RoadNetwork, Turn
from route_planner import plan_route
from traffic_aware import TrafficObservation, TrafficSnapshot

network = RoadNetwork(
    [
        Road("toy_start", "A", "B", 40, 10),
        Road("toy_main", "B", "C", 100, 10),
        Road("toy_alt", "B", "C", 140, 10),
        Road("toy_pickup", "C", "D", 60, 10),
    ],
    [
        Turn("toy_start", "toy_main"),
        Turn("toy_start", "toy_alt"),
        Turn("toy_main", "toy_pickup"),
        Turn("toy_alt", "toy_pickup"),
    ],
)
request_time = datetime(2024, 8, 1, 12, tzinfo=timezone.utc)
snapshot = TrafficSnapshot(
    request_time, {"toy_main": TrafficObservation(2.0, request_time)}, "synthetic fixture"
)
agent = EpsilonGreedy(epsilon=0.1, seed=42)
for method in ("static", "traffic_aware", "epsilon_greedy"):
    result = plan_route(
        network, "toy_start", "toy_pickup",
        request_time=request_time, method=method, traffic=snapshot,
        start_position_m=10, pickup_position_m=20, bandit=agent,
    )
    print(result.method, result.road_ids, result.estimated_time_s)
```

Keep one agent per comparable pickup context, for example the same taxi start, pickup, and traffic regime. Keep each chosen tuple with its request so that feedback can call `agent.update_route(chosen_road_ids, observed_wait_s)` even after ranking changes. `choose_route()` and `update(index, wait)` are also available for immediate feedback in the current candidate order. Use one completed training trial at a time during first sampling. Repeated selection before feedback can revisit an untried candidate. With one feasible candidate it is always chosen. With zero feasible candidates the planner returns `unreachable` and leaves the agent unchanged.

Record failed pickups and timeouts as separate outcomes. Skip their reward updates until the team defines a consistent failure-penalty policy. An action with no completed pickup can stay uninitialized, so the experiment protocol should set retry or timeout rules before training. No SUMO rewards have been collected by this implementation.

## Traffic snapshots and coverage

A `TrafficSnapshot(request_time, observations, source)` maps directed road IDs to `TrafficObservation(speed_mps, observed_at, status)`. Status is `"available"`, `"missing"`, or `"closed"`. `snapshot.age_seconds(road_id)` gives age at the request when the observation has a timestamp. The snapshot rejects future observations and must match the planner's request time. Providers are responsible for their own observation freshness policy.

- A finite positive observed speed is capped at the road's normal speed. Missing or invalid observations fall back to normal speed. The planner reports which roads were observed, missing, invalid, capped, closed, or served by fallback. These diagnostics cover the supplied network.
- A closure removes a road from all three methods when the same snapshot is supplied. This treats closures as shared hard feasibility constraints. Static A* otherwise uses only normal speeds in its objective.
- `load_traffic_data(path)` reads the historical CSV and keeps raw speeds and source fields such as `STATUS`. Invalid or ambiguous timestamps are dropped. NYC local timestamps and explicit-offset ISO timestamps are normalized to UTC. If multiple rows share a link and timestamp, the last CSV row wins.
- `get_traffic_snapshot(frame, request_time, max_age_minutes=15)` returns the latest row per DOT link in the inclusive lookback through the request. Invalid recent measurements are retained at this stage so they do not silently revive an older measurement.
- `historical_snapshot(..., road_to_link, meters_per_second_per_unit=...)` adapts those rows. It requires a caller-supplied verified unit conversion and a mapping from each directed SUMO road ID to one DOT link. One DOT link may cover several roads. An optional `accept_observation(row)` applies verified source quality rules. An invalid or rejected latest row is marked missing. DOT status codes remain uninterpreted by default. [NYC DOT dataset documentation](https://data.cityofnewyork.us/Transportation/DOT-Traffic-Speeds/i4gi-tjb9)
- The local FHV sample has **0 of 11 requests** with any positive raw speed in the default lookback, even before mapping or quality checks. The first positive-speed record is August 1, 2024 at 00:03:03 NYC time, after every sampled request. Historical evaluation needs overlapping requests, verified units and source quality rules, and a directional crosswalk. The fixture unit conversion test uses explicitly labeled km/h data.
- `SyntheticTraffic` constructs snapshots directly in m/s through the same provider interface. Future SUMO experiments need shared underlying demand and controls so each policy experiences comparable traffic.

## Verification from the repository root

Use Python **3.14.4** to reproduce the local checks. The pinned packages require Python 3.12 or newer and exclude 3.14.1 through the NetworkX version. Python 3.14.4 is the version tested here. Routing dependencies are pinned in `requirements.txt`. Pandas installs NumPy transitively. The timezone package supplies an IANA database fallback.

```bash
python -m pip install -r "Routing Algorithms/requirements.txt"
python -B -m unittest discover -s "Routing Algorithms" -p "test_*.py" -v
python -B "Routing Algorithms/routing.py"
python -B "Routing Algorithms/traffic_aware.py"
python -B "Routing Algorithms/epsilon_greedy.py"
python -B "Routing Algorithms/candidate_routes.py"
python -B "Routing Algorithms/route_planner.py"
python -B "Routing Algorithms/traffic_data.py"
```

Use `traffic_data.py --help` for configurable CSV paths, lookback, and sample snapshot times. The tests preserve the eight original cases and add seeded comparisons against NetworkX Dijkstra, exact directed-road cost checks, traffic adapter tests, and bandit regressions. The existing demos use their original junction fixtures. `prepare_routes` and `build_graph` remain available for those fixtures and assume unique junction-to-junction connections for candidate generation. Use `plan_route` with `RoadNetwork` for directed road integrations.

## Phase 2 preliminary synthetic experiments

### What the model represents

The fictional fixture has seven one-way roads, three corridors between the same two junctions, and two pickup roads. The shortest corridor is a 90 m road. Its alternatives are a 138 m parallel road and a 156 m two-road bypass. All normal speeds are 10 m/s. The taxi starts 10 m into its road, and each pickup is 30 m into its road. Legal turns are explicit. A known-closure scenario makes the pickup road unavailable to every method.

`SyntheticTraffic` supports explicit or seeded selection of affected directed roads, per-road overrides, missing observations, and closures. The experiment applies a nominal multiplier to the short road only. Free-flow, light, moderate, and heavy use 1.00, 0.80, 0.50, and 0.20 of normal speed, respectively. These are proposed synthetic settings. The runner uses three patterns.

| Pattern | Planner observes | Evaluator uses |
| --- | --- | --- |
| Perfect information | Exact scenario speeds | The same speeds, with zero added delays. This checks that a completed traffic-aware route has the predicted cost. |
| Noisy forecast | Independently perturbed speeds (uniform multiplier within ±18%, capped at normal speed), and a 20% chance the short-road observation is missing | Speeds perturbed within ±22% of nominal and capped at normal speed, a 12% chance of a hidden short-road closure, uniform delays of 0–3 s per traversed road, an extra 0–14 s for turning from the short road onto the pickup road, and 2 s of dispatch delay. |
| Known closure | The closed pickup road | The same closure, which makes the pickup unreachable. |

The synthetic evaluator adds `distance_m / realized_speed_mps` over the chosen road segments, then adds the realized road and turn delays. The modeled pickup time adds the dispatch delay to that route travel time. The evaluator receives a scenario that was generated before any policy planned. The planner receives the request, legal network, and departure-time snapshot. Forecast and realization randomness come from separate streams. A missing or invalid speed uses normal speed for planning. Only an explicitly observed closure changes shared planning feasibility. The model includes no dispatch queue, vehicle interactions, live updates mid-route, pickup dwell, or lane-level simulation.

### Training and held-out evaluation

Each seed runs the same template mix across three independent streams. The default cycle combines four congestion levels, two forecast patterns, and two pickup roads, plus one known closure, for 17 templates. Cycles are shuffled. The held-out set uses fresh stochastic draws from the same templates and the same network, so it measures performance on new realizations of familiar scenario types. Each policy receives the same request and underlying conditions for each matched scenario.

- **Training** keeps one agent per seed, ε value, and observable context. The context uses start and pickup positions, the forecast source label, and the short-road speed band derived from the snapshot. Missing speed has its own band. The nominal congestion label, hidden speeds, delays, and hidden closure never enter the context. Each candidate uses its exact road tuple as its identity. A candidate with no successful feedback remains in initial sampling, which precedes exploration at the configured ε. Each completed observation updates `R = -pickup_time_s` and `Q ← Q + (R − Q) / N`. The logs separate initial trials from subsequent selections using the ε rule.
- **Validation** compares frozen policies trained with ε in `{0, 0.05, 0.10, 0.20, 0.50}`. Select the highest completion rate, then the lowest completed-only mean pickup time, then the lowest ε. This rule uses validation alone.
- **Testing** freezes rewards and chooses the highest learned value among sampled candidates available on that request. Traffic rank breaks ties. If no available candidate has a learned value or the context is unseen, use the first traffic-ranked candidate. The optional first-candidate baseline uses exactly the same candidate generator and `k` as the learned selector. It provides a direct comparison against traffic ranking alone.

### Failures and interpretation

The outcome is `completed`, `unreachable` (planner finds no feasible route), `blocked` (a selected road is closed in the realization), or `timeout` (modeled pickup would take more than 65 s under the saved configuration). Failures count in completion and failure denominators. Pickup-time statistics use completed requests. Paired time differences use scenarios that both compared methods completed, and status pairs retain all requests. Failed training attempts leave the reward blank and add no sample. The mean reward therefore estimates negative waiting time **conditional on success**. A route that rarely completes but succeeds quickly can be ranked above a reliable, slower route. Persistently failing candidates can also keep an agent in its initial sampling stage.

Keep this as the primary completed-wait analysis and examine completion alongside it. For a future sensitivity analysis, retrain under a predeclared deadline-loss definition that assigns the common 65 s observation horizon to any noncompletion and actual pickup time to completions. This would represent the cost of failing to complete within the study window, rather than an observed wait. Compare it with the current success-only policy on paired, separately held-out realizations, using training and validation for any revised policy choices. The saved experiment uses the original success-only reward.

### Verified preliminary synthetic results

The saved run `synthetic-8074b600213b` has five paired seeds `11, 22, 33, 44, 55`, 119 training episodes per ε and seed, 51 validation requests and 68 test requests per seed, and `k = 3`. ε = 0.20 completed 239 of 255 validation requests. Completion for ε = 0, 0.05, 0.10, and 0.50 was 236, 236, 238, and 238 of 255, respectively. The corresponding completed-only validation means, in ascending ε order, were 22.93, 22.93, 22.89, 22.88, and 22.92 s. The counts are close enough that the selected ε should be understood as the winner of this validation run.

| Held-out policy | Completed / 340 | Unreachable | Blocked | Timeout | Mean / median / P90 (s), completed only |
| --- | ---: | ---: | ---: | ---: | --- |
| Static A* | 279 (82.06%) | 20 | 17 | 24 | 30.51 / 27.96 / 51.00 |
| Traffic-aware A* | 307 (90.29%) | 20 | 8 | 5 | 24.44 / 19.80 / 35.04 |
| First traffic-ranked candidate | 307 (90.29%) | 20 | 8 | 5 | 24.44 / 19.80 / 35.04 |
| Frozen ε-greedy | 319 (93.82%) | 20 | 1 | 0 | 23.08 / 19.80 / 29.83 |

Among jointly completed requests, traffic-aware A* averaged 6.43 s faster than static A* (279 pairs). Frozen ε-greedy averaged 8.16 s faster than static A* (279 pairs) and 1.52 s faster than traffic-aware A* or the first-candidate baseline (307 pairs). Frozen ε-greedy was slower on 25 of those 307 pairs and tied on 215. Traffic-aware A* was slower than static on two of their 279 jointly completed requests. All four methods took 15 s on the 40 perfect-information free-flow controls. The traffic-aware and first-candidate policies selected the same route or both found no route in all 340 held-out cases. The fixture repeats a small network, gives the short route extra hidden delay, and exposes only five independent seed runs. These choices limit what can be inferred about new networks, different traffic distributions, or actual SUMO pickup waits. These are descriptive comparisons with no uncertainty interval.

### Reproduction and outputs

Python 3.14.4 with the routing requirements reproduces the tests and CSVs. Plotting uses the existing Matplotlib 3.11.2 installation. For CSV-only runs in environments without it, add `--no-plots`. Choose a new or empty output directory inside `Routing Algorithms`.

```bash
.venv/bin/python -B -m unittest discover -s "Routing Algorithms" -p "test_*.py" -v
.venv/bin/python -B "Routing Algorithms/route_planner.py"
.venv/bin/python -B "Routing Algorithms/experiment.py" \
  --seeds 11 22 33 44 55 --train-episodes 119 \
  --validation-scenarios 51 --test-scenarios 68 \
  --levels free_flow light moderate heavy --epsilons 0 0.05 0.10 0.20 0.50 \
  --k 3 --timeout-s 65 --first-candidate \
  --output-dir "Routing Algorithms/preliminary_synthetic_replay"
```

| File | Contents |
| --- | --- |
| [config.json](preliminary_synthetic/config.json) | Exact parameters, master and derived split seeds, runtime versions, and the selected ε. |
| [episodes.csv](preliminary_synthetic/episodes.csv) | 6,375 policy episodes, with scenario IDs, seeds, selected and available routes, timing, outcomes, and training rewards. |
| [summary.csv](preliminary_synthetic/summary.csv) | Counts and pickup-time summaries by split, method, ε, congestion level, and pattern. Includes validation comparisons. |
| [paired_test.csv](preliminary_synthetic/paired_test.csv) | 1,360 matched comparison rows with both outcomes and, when both complete, the time difference in seconds. Negative values favor the method named first. |
| [route_counts.csv](preliminary_synthetic/route_counts.csv) | Candidate availability and route selection counts by split, seed, policy, and context. |
| [learned_routes.csv](preliminary_synthetic/learned_routes.csv) | Per-route successful reward sample counts and frozen mean rewards after training. |
| [preliminary_plots.png](preliminary_synthetic/preliminary_plots.png) | Six panels showing training curves, performance by congestion level, validation by ε, held-out pickup distributions, completion and failure, and paired differences. Overlapping traffic-aware and first-candidate traces reflect equal outcomes in this run. |

Scenario and agent seeds are derived from the master seed with SHA-256 and separate labels. The manifest records split seeds and the episode log provides per-scenario and agent seeds. Routing time includes planning and route selection, and varies across replays. It excludes outcome evaluation and reward updates.

## SUMO handoff and model limits

1. Export each usable directed SUMO edge as a `Road(id, from_junction, to_junction, length_m, normal_speed_mps, allowed_classes)`. Export legal edge-pair transitions as `Turn(from_road, to_road, allowed_classes)`, then build `RoadNetwork(roads, turns)`. Tony supplies the actual edges, permissions, and the rules for internal junction edges.
2. At dispatch, locate the taxi and pickup on valid directed edges and record their forward position offsets in meters. Construct one request-time snapshot for the paired policies using only available traffic information. The request timestamp and simulation clock need an agreed mapping. For a taxi already moving, retain its current edge as the first route edge and apply SUMO's route-change requirements. [SUMO road networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html), [TraCI vehicle API](https://sumo.dlr.de/daily/pydoc/traci/_vehicle.html)
3. Use the Phase 1 `plan_route` example above or the `run_episode` wrapper below, which calls that same planner with the road IDs, positions, and snapshot. Share request and starting conditions across policy runs and reset the simulation to the matching starting state for each run.

```python
from epsilon_greedy import EpsilonGreedy
from experiment import FrozenSelector, run_episode
from synthetic_environment import PickupOutcome, PickupRequest

# Keep this dictionary outside the request loop.
agents = {}

def compare_pickup(network, request_id, start_edge, start_m, pickup_edge, pickup_m,
                   snapshot, context, drive_in_sumo, vehicle_class, *,
                   learn=False, frozen_values=None):
    request = PickupRequest(
        request_id,
        network.position(start_edge, start_m),
        network.position(pickup_edge, pickup_m),
        snapshot,
        context,
    )
    if learn:
        if context not in agents:
            agents[context] = EpsilonGreedy(epsilon=0.20, seed=42)
        selector = agents[context]
    else:
        selector = FrozenSelector(frozen_values)

    def measure(request, plan):
        if plan.road_ids is None:
            return PickupOutcome("unreachable")
        # Tony implements this callback using a matched SUMO initial state.
        # It returns a PickupOutcome from observed timestamps or a failure status.
        return drive_in_sumo(request, list(plan.road_ids))

    for method in ("static", "traffic_aware", "epsilon_greedy"):
        episode = run_episode(
            network, request, method, measure, vehicle_class=vehicle_class,
            selector=selector if method == "epsilon_greedy" else None,
        )
        route_ids = episode.plan.road_ids
        outcome = episode.outcome
        if learn and method == "epsilon_greedy" and outcome.status == "completed":
            selector.update_route(route_ids, outcome.pickup_time_s)
        print(method, route_ids, outcome.status, outcome.pickup_time_s)
```

For training, pass `learn=True`. For frozen evaluation, pass `frozen_values` as a mapping from road tuples to mean rewards estimated from completed training pickups for that context. An empty mapping uses the first-candidate fallback. Tony's callback should subtract the request timestamp from the pickup timestamp, convert the difference to seconds, and return `PickupOutcome("completed", waiting_s, route_travel_s)`. It can also report `PickupOutcome("blocked")` or `PickupOutcome("timeout")` with empty times. If simulation is asynchronous, store the selected tuple with its request and update the agent only when completion feedback arrives.

**Tony** still needs to confirm lane feasibility, edge lengths and units, vehicle class, pickup edge/lane/position, route acceptance, traffic availability at dispatch, timestamp mapping, and precise pickup and timeout events. The geographical A* heuristic remains at its zero default pending an admissible lower bound for the actual network. **Krasi** still needs to verify DOT units and status codes, locate passenger requests with overlapping traffic observations, and provide a directional DOT-link-to-SUMO-edge crosswalk with coverage and ambiguity checks. The fixture results establish software behavior and preliminary synthetic trade-offs. Actual SUMO pickup waits and historical NYC integration remain unverified.

## AI attribution

OpenAI ChatGPT assisted with the initial routing code, tests, data scripts, documentation, and integration guidance. OpenAI Codex assisted with Phase 1 review, implementation, debugging, regression tests, and documentation. In Phase 2, Codex assisted with the synthetic provider and outcome model, the experimental runner and plots, regression tests, result verification, and documentation. Charis independently ran the regression suite. Review and disclose the actual AI contributions in the course-required format and placement for submitted code and report material. The team still needs to confirm the course-specific requirements. [AI Attribution Toolkit](https://aiattribution.github.io/)
