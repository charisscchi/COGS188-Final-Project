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
| `k`, `bandit` | Positive integer candidate limit (default 3) and caller-owned `EpsilonGreedy` agent for the epsilon-greedy method. |

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
- A small synthetic provider can construct a snapshot directly using m/s. Phase 1 implements the planner input contract. Future experiments need corresponding traffic conditions in SUMO so all policies experience the same underlying demand or controls.

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

## SUMO handoff and model limits

Tony can call `plan_route` now with a verified network export or a test fixture. The network adapter still needs the actual public SUMO edge IDs, legal lane connections and class permissions, start edge and position, and legal pickup edge, lane, and position. For an already moving taxi, coordinate the call with SUMO's route-change rules and supply the current edge as the first edge. Use the same timestamp convention for traffic snapshots and the simulation clock. [SUMO road networks](https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html), [TraCI vehicle API](https://sumo.dlr.de/daily/pydoc/traci/_vehicle.html)

This model assumes a taxi can move between eligible lanes within an edge as needed for the next legal turn. The final integration must verify lane feasibility, internal intersection handling, route acceptance, pickup events, and failure outcomes. Costs use a single departure-time snapshot throughout the route. They omit separate turn delays, pickup dwell, and vehicle interactions. Estimated travel time is a planner input and diagnostic. Simulated request-to-pickup time is the evaluation outcome and the source of successful bandit rewards.

## AI attribution

OpenAI ChatGPT assisted with the initial routing code, tests, data scripts, documentation, and integration guidance. OpenAI Codex assisted with the Phase 1 code review, implementation, debugging, regression tests, and README. The recorded local checks cover routing software behavior with fixtures. The team should review the changes and follow COGS 188's required attribution format and placement for submitted code and report material. The course-specific wording and submission requirements still need confirmation. [AI Attribution Toolkit](https://aiattribution.github.io/)
