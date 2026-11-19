
# Routing Algorithms

## Overview

This folder contains the routing methods developed for
our simulated robotaxi system. The goal is to compare
different approaches to reducing passenger pickup
waiting time under varying traffic conditions.

## Routing Methods

### Static A*

Uses fixed road speeds to estimate travel times and
select a minimum-cost route.

### Traffic-aware A*

Uses observed traffic speeds to calculate road costs.
Missing or invalid observations currently fall back
to normal road speeds.

### Epsilon-greedy

Selects among candidate routes using an
exploration-exploitation strategy. The agent updates
its estimated rewards based on negative passenger
waiting time.

## File Structure

| File | Description |
|---|---|
| inspect_data.py | Examines the NYC datasets |
| routing.py | Core A* implementation |
| traffic_aware.py | Traffic-dependent graph costs |
| epsilon_greedy.py | Route selection and reward updates |
| test_routing.py | Unit tests for the routing methods |
| traffic_data.py | Processes historical traffic observations |
| candidate_routes.py | Generates alternative routes |
| route_planner.py | Combined routing interface |

## Running the Code

Run the following commands from the repository root:

    python -m pip install -r "Routing Algorithms/requirements.txt"

    python "Routing Algorithms/test_routing.py"

    python "Routing Algorithms/traffic_data.py"

    python "Routing Algorithms/route_planner.py"

## Integration with SUMO

The prepare_routes() function in route_planner.py
accepts static and traffic-dependent road graphs,
a starting location, and a pickup location.

It returns the static A* route, traffic-aware A*
route, estimated travel times, and candidate routes
for epsilon-greedy selection.

The epsilon-greedy agent receives actual waiting
times after each simulation and updates its reward
estimates accordingly.

The shared road graph must preserve valid road
connections, individual road IDs, and legal turns.

## Initial Testing

All eight routing unit tests passed.

The individual routing methods, candidate route
generation, and combined routing interface were
also tested using a small controlled road network.

These tests verify the initial implementation.
They do not represent actual NYC simulation results.

## Current Limitations

- DOT traffic links have not yet been matched
  to SUMO road IDs.
- The current passenger dataset contains only
  11 raw trip records.
- Nonpositive traffic speeds are temporarily
  excluded during traffic data preparation.
- Traffic speed units must be verified before
  conversion to simulation units.
- Traffic snapshots use observations from the
  preceding 15 minutes.
- The zero-heuristic fallback behaves like Dijkstra's
  algorithm rather than heuristic-guided A*.
- Epsilon-greedy requires repeated comparable
  scenarios and actual simulation feedback.

## AI Attribution

OpenAI ChatGPT was used to generate the initial
Python implementations, test cases, data-processing
scripts, documentation, and integration guidance.

The code was run locally, and the initial tests
passed. Further code review and SUMO integration
are still required.

Attribution framework:
https://aiattribution.github.io/
