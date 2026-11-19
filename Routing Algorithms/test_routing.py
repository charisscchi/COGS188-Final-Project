import math
import random
import unittest
from datetime import datetime, timedelta, timezone
from io import StringIO

import networkx as nx

from candidate_routes import generate_candidate_routes, route_travel_time
from epsilon_greedy import EpsilonGreedy
from road_network import Road, RoadNetwork, Turn, build_road_query
from route_planner import plan_route, prepare_routes
from routing import astar
from traffic_aware import TrafficObservation, TrafficSnapshot, build_graph, routing_speeds
from traffic_data import get_traffic_snapshot, historical_snapshot, load_traffic_data


class TestRouting(unittest.TestCase):

    def setUp(self):
        self.roads = [
            ("S", "A", 300, 10),
            ("A", "G", 300, 10),
            ("S", "B", 300, 10),
            ("B", "C", 400, 10),
            ("C", "G", 300, 10)
        ]

        self.heuristic = {
            "S": 60,
            "A": 30,
            "B": 70,
            "C": 30,
            "G": 0
        }

    def test_static_astar(self):
        graph = build_graph(self.roads)

        path, cost = astar(
            graph, "S", "G", self.heuristic
        )

        self.assertEqual(path, ["S", "A", "G"])
        self.assertAlmostEqual(cost, 60)

    def test_traffic_aware_astar(self):
        speeds = {
            ("S", "A"): 2,
            ("A", "G"): 2
        }

        graph = build_graph(self.roads, speeds)

        path, cost = astar(
            graph, "S", "G", self.heuristic
        )

        self.assertEqual(path, ["S", "B", "C", "G"])
        self.assertAlmostEqual(cost, 100)

    def test_no_route(self):
        graph = {"S": [], "G": []}
        heuristic = {"S": 0, "G": 0}

        path, cost = astar(
            graph, "S", "G", heuristic
        )

        self.assertIsNone(path)
        self.assertTrue(math.isinf(cost))

    def test_same_start_and_goal(self):
        graph = build_graph(self.roads)

        path, cost = astar(
            graph, "G", "G", self.heuristic
        )

        self.assertEqual(path, ["G"])
        self.assertEqual(cost, 0)

    def test_invalid_speed(self):
        speeds = {
            ("S", "A"): 0,
            ("A", "G"): -5
        }

        graph = build_graph(self.roads, speeds)

        path, cost = astar(
            graph, "S", "G", self.heuristic
        )

        self.assertEqual(path, ["S", "A", "G"])
        self.assertAlmostEqual(cost, 60)

    def test_missing_traffic(self):
        graph = build_graph(self.roads, {})

        path, cost = astar(
            graph, "S", "G", self.heuristic
        )

        self.assertEqual(path, ["S", "A", "G"])
        self.assertAlmostEqual(cost, 60)

    def test_reward_update(self):
        agent = EpsilonGreedy(
            routes=[["S", "A", "G"]],
            epsilon=0.1,
            seed=42
        )

        agent.update(0, 100)
        agent.update(0, 200)

        self.assertEqual(agent.counts[0], 2)
        self.assertAlmostEqual(agent.q_values[0], -150)

    def test_epsilon_greedy_learning(self):
        routes = [
            ["S", "A", "G"],
            ["S", "B", "C", "G"]
        ]

        agent = EpsilonGreedy(
            routes, epsilon=0.1, seed=42
        )

        waiting_times = [300, 100]

        for _ in range(200):
            action = agent.choose_route()
            agent.update(action, waiting_times[action])

        self.assertEqual(
            agent.best_route(),
            ["S", "B", "C", "G"]
        )

        self.assertEqual(sum(agent.counts), 200)


class TestAStarRegression(unittest.TestCase):
    def test_reopens_a_node_with_an_inconsistent_but_admissible_heuristic(self):
        graph = {"S": [("A", 4), ("B", 1)], "B": [("A", 1)], "A": [("G", 9)], "G": []}
        heuristic = {"S": 0, "A": 0, "B": 8, "G": 0}
        # A is initially expanded at cost 4. Going through B later improves it to 2.
        path, cost = astar(graph, "S", "G", heuristic)
        self.assertEqual(path, ["S", "B", "A", "G"])
        self.assertEqual(cost, route_travel_time(graph, path))
        self.assertEqual(cost, 11)

    def test_heap_ties_do_not_compare_node_ids(self):
        graph = {"S": [(1, 1), ("A", 1)], 1: [(None, 1)], "A": [(None, 1)], None: []}
        path, cost = astar(graph, "S", None)
        self.assertEqual(path, ["S", 1, None])
        self.assertEqual(cost, 2)

    def test_unknown_endpoints_and_invalid_heuristics(self):
        graph = {"S": [("G", 1)]}  # A node appearing only as a successor is known.
        self.assertEqual(astar(graph, "S", "G"), (["S", "G"], 1))
        for start, goal in (("?", "G"), ("S", "?"), ("?", "?")):
            with self.subTest(start=start, goal=goal), self.assertRaisesRegex(ValueError, "known"):
                astar(graph, start, goal)
        for heuristic in ({"S": 0}, {"S": 0, "G": 1}, {"S": float("nan"), "G": 0}):
            with self.subTest(heuristic=heuristic), self.assertRaises(ValueError):
                astar(graph, "S", "G", heuristic)

    def test_invalid_weights_and_ambiguous_legacy_scoring(self):
        for bad in (-1, float("nan"), float("inf"), "10", True):
            graph = {"S": [("G", bad)], "G": []}
            with self.subTest(cost=bad):
                with self.assertRaises(ValueError):
                    astar(graph, "S", "G")
                with self.assertRaises(ValueError):
                    generate_candidate_routes(graph, "S", "G")
        graph = {"S": [("G", 2), ("G", 5)], "G": []}
        for route in ([], ["S", "G"], ["G", "S"]):
            with self.subTest(route=route), self.assertRaises(ValueError):
                route_travel_time(graph, route)
        with self.assertRaises(ValueError):
            generate_candidate_routes(graph, "S", "G")

    def test_seeded_directed_graphs_against_dijkstra(self):
        rng = random.Random(188)
        for case in range(30):
            nodes = range(8)
            reference = nx.DiGraph()
            reference.add_nodes_from(nodes)
            reference.add_weighted_edges_from(
                (source, dest, rng.randrange(0, 20))
                for source in nodes for dest in nodes if source != dest and rng.random() < 0.25
            )
            graph = {
                node: [(neighbor, edge["weight"]) for neighbor, edge in reference[node].items()]
                for node in nodes
            }
            for goal in (0, 3, 7):
                distances = nx.single_source_dijkstra_path_length(reference.reverse(copy=False), goal)
                # Pick lower bounds independently so consistency is not assumed.
                heuristic = {node: rng.randint(0, distances.get(node, 0)) for node in nodes}
                for start in nodes:
                    with self.subTest(case=case, start=start, goal=goal):
                        path, cost = astar(graph, start, goal, heuristic)
                        self.assertEqual(cost, distances.get(start, math.inf))
                        if start in distances:
                            self.assertEqual((path[0], path[-1]), (start, goal))
                            self.assertEqual(route_travel_time(graph, path), cost)
                        else:
                            self.assertIsNone(path)


REQUEST = datetime(2024, 8, 1, 12, tzinfo=timezone.utc)


def synthetic_network(*, equal_cost=False, reverse_input=False):
    """Invented fixture road IDs and turns. No geographic or SUMO mapping."""
    roads = [
        Road("toy_start", "A", "B", 40, 10),
        Road("toy_fast", "B", "C", 100, 10),
        Road("toy_alt", "B", "C", 100 if equal_cost else 140, 10),
        Road("toy_forbidden", "B", "C", 1, 10),
        Road("toy_bus_only", "B", "C", 10, 10, frozenset({"bus"})),
        Road("toy_turn_limited", "B", "C", 20, 10),
        Road("toy_pickup", "C", "D", 60, 10),
        Road("toy_disconnected", "U", "V", 10, 10),
    ]
    turns = [
        Turn("toy_start", "toy_fast"),
        Turn("toy_start", "toy_alt"),
        Turn("toy_start", "toy_bus_only"),
        Turn("toy_start", "toy_turn_limited", frozenset({"bus"})),
        *[Turn(road.id, "toy_pickup") for road in roads[1:6]],
    ]
    if reverse_input:
        roads.reverse()
        turns.reverse()
    return RoadNetwork(roads, turns)


class TestDirectedRoadPlanning(unittest.TestCase):
    def setUp(self):
        self.network = synthetic_network()

    def plan(self, method="static", start="toy_start", pickup="toy_pickup", **options):
        return plan_route(self.network, start, pickup, request_time=REQUEST, method=method, **options)

    def test_individual_road_costs_positions_and_forbidden_turns(self):
        result = self.plan(start_position_m=10, pickup_position_m=20)
        self.assertEqual(result.road_ids, ("toy_start", "toy_fast", "toy_pickup"))
        self.assertEqual(result.estimated_time_s, (30 + 100 + 20) / 10)
        query = build_road_query(
            self.network, self.network.position("toy_start", 10),
            self.network.position("toy_pickup", 20),
            {road_id: road.normal_speed_mps for road_id, road in self.network.roads.items()},
            "passenger",
        )
        self.assertEqual(query.travel_time(result.road_ids), result.estimated_time_s)
        for road_id in ("toy_forbidden", "toy_bus_only", "toy_turn_limited"):
            with self.subTest(road=road_id), self.assertRaisesRegex(ValueError, "transition"):
                query.travel_time(("toy_start", road_id, "toy_pickup"))

    def test_vehicle_class_permissions(self):
        bus = self.plan(vehicle_class="bus")
        self.assertEqual(bus.road_ids, ("toy_start", "toy_bus_only", "toy_pickup"))
        restricted_pickup = self.plan(pickup="toy_bus_only")
        self.assertEqual(restricted_pickup.status, "unreachable")

    def test_disconnected_reverse_and_invalid_locations(self):
        for start, pickup in (("toy_pickup", "toy_start"), ("toy_start", "toy_disconnected")):
            result = self.plan(start=start, pickup=pickup)
            self.assertEqual((result.status, result.road_ids, result.estimated_time_s),
                             ("unreachable", None, None))
        for options in (
            {"start": "unknown"}, {"pickup": "unknown"}, {"pickup_position_m": 61},
            {"start_position_m": -1}, {"start_position_m": float("nan")}, {"method": "unknown"},
            {"method": "epsilon_greedy"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.plan(**options)

    def test_same_road_ahead_equal_and_behind(self):
        for pickup, expected in ((30, 2), (10, 0)):
            result = self.plan(pickup="toy_start", start_position_m=10, pickup_position_m=pickup)
            self.assertEqual(result.road_ids, ("toy_start",))
            self.assertEqual(result.estimated_time_s, expected)
        behind = self.plan(pickup="toy_start", start_position_m=25, pickup_position_m=5)
        self.assertEqual(behind.status, "unreachable")

    def test_same_road_behind_with_legal_reentry(self):
        network = RoadNetwork(
            [Road("toy_out", "A", "B", 50, 10), Road("toy_return", "B", "A", 100, 10)],
            [Turn("toy_out", "toy_return"), Turn("toy_return", "toy_out")],
        )
        for method in ("static", "traffic_aware", "epsilon_greedy"):
            with self.subTest(method=method):
                result = plan_route(
                    network, "toy_out", "toy_out", method=method, request_time=REQUEST,
                    start_position_m=40, pickup_position_m=10, bandit=EpsilonGreedy(),
                )
                self.assertEqual(result.road_ids, ("toy_out", "toy_return", "toy_out"))
                self.assertEqual(result.estimated_time_s, (10 + 100 + 10) / 10)

    def test_goal_specific_heuristic_receives_road_positions(self):
        calls = []

        def lower_bound(state, pickup):
            calls.append((state, pickup))
            return 0

        guided = self.plan(start_position_m=4, pickup_position_m=50, heuristic=lower_bound)
        default = self.plan(start_position_m=4, pickup_position_m=50)
        self.assertEqual(guided, default)
        self.assertEqual(calls[0][0], self.network.position("toy_start", 4))
        self.assertTrue(all(pickup == self.network.position("toy_pickup", 50) for _, pickup in calls))

    def test_invalid_road_attributes_and_turns(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Road("toy", "A", "B", value, 10)
                with self.assertRaises(ValueError):
                    Road("toy", "A", "B", 10, value)
                with self.assertRaises(ValueError):
                    build_graph([("A", "B", 10, value)])
        roads = list(self.network.roads.values())
        for turn in (Turn("toy_start", "unknown"), Turn("toy_fast", "toy_start")):
            with self.subTest(turn=turn), self.assertRaises(ValueError):
                RoadNetwork(roads, [turn])
        with self.assertRaises(ValueError):
            RoadNetwork([roads[0], roads[0]], [])
        turn = Turn("toy_start", "toy_fast")
        with self.assertRaises(ValueError):
            RoadNetwork(roads, [turn, turn])

    def test_candidate_counts_uniqueness_costs_and_identity(self):
        agent = EpsilonGreedy()
        none = self.plan("epsilon_greedy", pickup="toy_disconnected", bandit=agent)
        self.assertEqual((none.status, none.candidates, agent.routes), ("unreachable", (), ()))
        one = self.plan("epsilon_greedy", pickup="toy_fast", bandit=agent)
        self.assertEqual(len(one.candidates), 1)
        self.assertEqual(one.road_ids, one.candidates[0].road_ids)
        many = self.plan("epsilon_greedy", k=5, bandit=agent)
        self.assertEqual([option.estimated_time_s for option in many.candidates], [20, 24])
        routes = tuple(option.road_ids for option in many.candidates)
        self.assertEqual(len(set(routes)), 2)
        self.assertEqual([route[1] for route in routes], ["toy_fast", "toy_alt"])
        self.assertEqual(agent.routes, routes)
        with self.assertRaises(ValueError):
            self.plan("epsilon_greedy", bandit=agent, k=1.5)

    def test_equal_cost_candidate_order_is_independent_of_network_input_order(self):
        def candidate_result(reverse_input, k):
            return plan_route(
                synthetic_network(equal_cost=True, reverse_input=reverse_input),
                "toy_start", "toy_pickup", method="epsilon_greedy", request_time=REQUEST,
                bandit=EpsilonGreedy(seed=8), k=k,
            )

        forward = candidate_result(False, 3)
        backward = candidate_result(True, 3)
        self.assertEqual(forward, backward)
        self.assertEqual([option.estimated_time_s for option in forward.candidates], [20, 20])
        self.assertEqual(candidate_result(True, 1).candidates, forward.candidates[:1])

    def test_candidates_omit_a_detour_that_repeats_a_road(self):
        network = RoadNetwork(
            [
                Road("toy_start", "A", "B", 10, 10),
                Road("toy_forward", "B", "C", 10, 10),
                Road("toy_back", "C", "B", 10, 10),
                Road("toy_pickup", "C", "D", 10, 10),
            ],
            [
                Turn("toy_start", "toy_forward"), Turn("toy_forward", "toy_back"),
                Turn("toy_back", "toy_forward"), Turn("toy_forward", "toy_pickup"),
            ],
        )
        result = plan_route(network, "toy_start", "toy_pickup", request_time=REQUEST,
                            method="epsilon_greedy", bandit=EpsilonGreedy(), k=5)
        self.assertEqual([option.road_ids for option in result.candidates],
                         [("toy_start", "toy_forward", "toy_pickup")])

    def test_no_congestion_parity_and_explicit_closures(self):
        empty = TrafficSnapshot(REQUEST, {}, "synthetic test")
        static = self.plan()
        aware = self.plan("traffic_aware", traffic=empty)
        self.assertEqual((static.road_ids, static.estimated_time_s),
                         (aware.road_ids, aware.estimated_time_s))
        self.assertEqual(len(aware.diagnostics.fallback_roads), len(self.network.roads))
        closure = TrafficObservation(None, REQUEST, "closed")
        for method in ("static", "traffic_aware", "epsilon_greedy"):
            with self.subTest(method=method):
                traffic = TrafficSnapshot(REQUEST, {"toy_fast": closure}, "synthetic test")
                result = self.plan(method, traffic=traffic, bandit=EpsilonGreedy())
                self.assertEqual(result.road_ids[1], "toy_alt")
                self.assertEqual(result.diagnostics.closed_roads, ("toy_fast",))
                self.assertNotIn("toy_fast", result.diagnostics.fallback_roads)
                blocked = TrafficSnapshot(REQUEST, {"toy_pickup": closure}, "synthetic test")
                self.assertEqual(self.plan(method, traffic=blocked, bandit=EpsilonGreedy()).status,
                                 "unreachable")

    def test_combined_policies_learning_and_costs(self):
        agent = EpsilonGreedy(epsilon=0, seed=9)
        congestion = TrafficSnapshot(
            REQUEST, {"toy_fast": TrafficObservation(2, REQUEST)}, "synthetic test",
        )
        provider_calls = []

        def provider(when):
            provider_calls.append(when)
            return congestion

        static = self.plan(traffic=congestion)
        aware = self.plan("traffic_aware", traffic=provider)
        self.assertEqual(provider_calls, [REQUEST])
        self.assertEqual((static.road_ids[1], static.estimated_time_s), ("toy_fast", 20))
        self.assertEqual((aware.road_ids[1], aware.estimated_time_s), ("toy_alt", 24))
        selected = set()
        for _ in range(2):
            result = self.plan("epsilon_greedy", traffic=congestion, bandit=agent)
            selected.add(result.road_ids)
            expected = {option.road_ids: option.estimated_time_s for option in result.candidates}
            self.assertEqual(result.estimated_time_s, expected[result.road_ids])
            agent.update_route(result.road_ids, 60 if result.road_ids[1] == "toy_fast" else 30)
        self.assertEqual(len(selected), 2)
        # Empty traffic reverses the ordering, but the learned choice still uses the same road IDs.
        result = self.plan("epsilon_greedy", bandit=agent)
        self.assertEqual(result.road_ids, aware.road_ids)
        self.assertEqual([option.estimated_time_s for option in result.candidates], [20, 24])
        self.assertEqual(agent.counts, [1, 1])
        self.assertEqual(agent.q_values, [-60, -30])

    def test_legacy_combined_demo_interface(self):
        graph = {"S": [("G", 5)], "G": []}
        result = prepare_routes(graph, graph, "S", "G")
        self.assertEqual(result["static_route"], result["traffic_aware_route"])
        self.assertEqual(result["candidate_routes"], [["S", "G"]])


def traffic_fixture():
    # Fictional export rows used to exercise the DOT adapter. The test defines their units.
    return load_traffic_data(StringIO(
        "LINK_ID,SPEED,DATA_AS_OF,STATUS\n"
        "0001,36,2024 Aug 01 07:45:00 AM,test\n"
        "0001,18,2024 Aug 01 07:55:00 AM,test\n"
        "0001,7.2,2024-08-01T11:55:00+00:00,test\n"
        "0001,7.2,2024-08-01T11:55:00+00:00,test\n"
        "0001,50,2024 Aug 01 08:01:00 AM,test\n"
        "0002,36,2024 Aug 01 07:44:00 AM,test\n"
        "0003,36,2024 Aug 01 07:48:00 AM,test\n"
        "0003,0,2024 Aug 01 07:58:00 AM,test\n"
        "0004,-8,2024 Aug 01 07:59:00 AM,test\n"
        "0005,bad,2024 Aug 01 07:59:00 AM,test\n"
        "0006,10,invalid timestamp,test\n"
        "0007,10,2024 Nov 03 01:30:00 AM,test\n"
        "0008,10,2024 Mar 10 02:30:00 AM,test\n"
        "0009,10,2024 Aug 01 07:45:00 AM,test\n"
    ))


class TestTrafficInputs(unittest.TestCase):
    def test_source_timezones_duplicate_policy_age_and_lookback(self):
        traffic = traffic_fixture()
        self.assertEqual(len(traffic), 9)
        self.assertEqual(str(traffic["DATA_AS_OF"].dt.tz), "UTC")
        naive = get_traffic_snapshot(traffic, datetime(2024, 8, 1, 8))
        aware = get_traffic_snapshot(traffic, REQUEST)
        self.assertTrue(naive.equals(aware))
        self.assertFalse(aware["DATA_AS_OF"].gt(REQUEST).any())
        self.assertEqual(aware.set_index("LINK_ID").loc["0001", "SPEED"], 7.2)
        self.assertNotIn("0002", set(aware["LINK_ID"]))
        self.assertIn("0009", set(aware["LINK_ID"]))
        wider = get_traffic_snapshot(traffic, "08/01/2024 08:00:00 AM", max_age_minutes=16)
        self.assertIn("0002", set(wider["LINK_ID"]))
        self.assertNotIn("0001", set(get_traffic_snapshot(traffic, REQUEST, 4)["LINK_ID"]))
        self.assertTrue(get_traffic_snapshot(traffic, REQUEST, 0).empty)
        for bad in ("invalid", "2024-11-03T01:30:00", "2024-03-10T02:30:00"):
            with self.subTest(time=bad), self.assertRaises(ValueError):
                get_traffic_snapshot(traffic, bad)
        with self.assertRaises(ValueError):
            get_traffic_snapshot(traffic, REQUEST, -1)

    def test_historical_fixture_conversion_matches_synthetic_mps(self):
        # Fixture source is km/h, so exactly 1 km/h = 1000 m / 3600 s.
        historical = historical_snapshot(
            traffic_fixture(), REQUEST, {"toy_fast": "0001"},
            meters_per_second_per_unit=1000 / 3600,
        )
        self.assertAlmostEqual(historical.observations["toy_fast"].speed_mps, 2.0)
        self.assertEqual(historical.age_seconds("toy_fast"), 300)
        synthetic = TrafficSnapshot(
            REQUEST, {"toy_fast": TrafficObservation(2, REQUEST - timedelta(minutes=5))},
            "synthetic test",
        )
        results = [
            plan_route(synthetic_network(), "toy_start", "toy_pickup", request_time=REQUEST,
                       method="traffic_aware", traffic=snapshot)
            for snapshot in (historical, synthetic)
        ]
        self.assertEqual(results[0].road_ids, results[1].road_ids)
        self.assertEqual(results[0].estimated_time_s, results[1].estimated_time_s)
        self.assertEqual(routing_speeds(synthetic_network(), historical, use_traffic=True)[0],
                         routing_speeds(synthetic_network(), synthetic, use_traffic=True)[0])
        with self.assertRaises(ValueError):
            historical_snapshot(traffic_fixture(), REQUEST, {}, meters_per_second_per_unit=0)

    def test_invalid_latest_rows_and_explicit_quality_rule(self):
        snapshot = historical_snapshot(
            traffic_fixture(), REQUEST,
            {"toy_start": "0003", "toy_fast": "0004", "toy_pickup": "0005", "toy_alt": "0001"},
            meters_per_second_per_unit=1, accept_observation=lambda row: row["STATUS"] == "accepted",
        )
        self.assertTrue(all(observation.status == "missing" for observation in snapshot.observations.values()))
        self.assertEqual(snapshot.age_seconds("toy_start"), 120)
        self.assertEqual(snapshot.age_seconds("toy_alt"), 300)

    def test_empty_historical_snapshot_is_reported_as_fallback(self):
        early = REQUEST - timedelta(days=1)
        traffic = traffic_fixture()
        self.assertTrue(get_traffic_snapshot(traffic, early).empty)
        snapshot = historical_snapshot(traffic, early, {"toy_fast": "0001"}, meters_per_second_per_unit=1)
        network = synthetic_network()
        result = plan_route(network, "toy_start", "toy_pickup", request_time=early,
                            method="traffic_aware", traffic=snapshot)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.diagnostics.source, "historical DOT")
        self.assertEqual(result.diagnostics.observed_roads, ())
        self.assertEqual(set(result.diagnostics.fallback_roads), set(network.roads))

    def test_speed_caps_and_invalid_speeds_fall_back(self):
        network = synthetic_network()
        observations = {
            "toy_start": TrafficObservation(100, REQUEST),
            "toy_fast": TrafficObservation(0, REQUEST),
            "toy_alt": TrafficObservation(-5, REQUEST),
            "toy_pickup": TrafficObservation(float("nan"), REQUEST),
            "toy_bus_only": TrafficObservation(float("inf"), REQUEST),
            "toy_forbidden": TrafficObservation("unknown", REQUEST),
        }
        snapshot = TrafficSnapshot(REQUEST, observations, "synthetic test")
        speeds, details = routing_speeds(network, snapshot, use_traffic=True)
        self.assertEqual(speeds, {road_id: road.normal_speed_mps for road_id, road in network.roads.items()})
        self.assertEqual(details.capped_roads, ("toy_start",))
        self.assertEqual(set(details.invalid_speed_roads), set(observations) - {"toy_start"})
        self.assertEqual(set(details.fallback_roads), set(network.roads) - {"toy_start"})

    def test_future_or_mismatched_snapshot_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "future"):
            TrafficSnapshot(REQUEST, {"toy_fast": TrafficObservation(2, REQUEST + timedelta(seconds=1))},
                            "synthetic test")
        network = synthetic_network()
        for snapshot in (
            TrafficSnapshot(REQUEST - timedelta(minutes=1), {}, "synthetic test"),
            TrafficSnapshot(REQUEST, {"wrong_network": TrafficObservation(2, REQUEST)}, "synthetic test"),
            lambda when: None,
        ):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                plan_route(network, "toy_start", "toy_pickup", request_time=REQUEST,
                           method="traffic_aware", traffic=snapshot)


class TestBanditRegression(unittest.TestCase):
    def test_mandatory_first_trials_even_at_zero_epsilon(self):
        agent = EpsilonGreedy([["a"], ["b"], ["c"]], epsilon=0, seed=99)
        choices = []
        for _ in range(3):
            action = agent.choose_route()
            choices.append(action)
            agent.update(action, action * 10)
        self.assertEqual(set(choices), {0, 1, 2})
        self.assertEqual(agent.counts, [1, 1, 1])
        self.assertEqual({agent.choose_route() for _ in range(20)}, {0})

    def test_full_exploration_and_seeded_tie_breaks(self):
        def selections(epsilon, tied):
            agent = EpsilonGreedy([["a"], ["b"]], epsilon=epsilon, seed=132)
            agent.update(0, 10)
            agent.update(1, 10 if tied else 100)
            return [agent.choose_route() for _ in range(50)]

        for epsilon, tied in ((1, False), (0, True)):
            with self.subTest(epsilon=epsilon, tied=tied):
                sequence = selections(epsilon, tied)
                self.assertEqual(sequence, selections(epsilon, tied))
                self.assertEqual(set(sequence), {0, 1})

    def test_invalid_inputs_do_not_change_estimates(self):
        for epsilon in (-1, 1.1, float("nan"), float("inf"), "0.1"):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                EpsilonGreedy([["a"]], epsilon=epsilon)
        agent = EpsilonGreedy([["a"]])
        for action in (-1, 1, True, 0.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                agent.update(action, 10)
        for wait in (-1, float("nan"), float("inf"), "10"):
            with self.subTest(wait=wait), self.assertRaises(ValueError):
                agent.update_route(("a",), wait)
        self.assertEqual((agent.counts, agent.q_values), ([0], [0]))

    def test_copy_routes_and_retain_rewards_after_reorder(self):
        routes = [["a"], ["b"]]
        agent = EpsilonGreedy(routes, epsilon=0)
        routes[0][0] = "mutation"
        routes.append(["c"])
        self.assertEqual(agent.routes, (("a",), ("b",)))
        agent.update(0, 80)
        agent.update(1, 10)
        agent.set_candidates([["b"], ["a"]])
        self.assertEqual(agent.choose_route(), 0)
        agent.update_route(("a",), 40)
        self.assertEqual(agent.q_values, [-10, -60])
        agent.set_candidates([["a"]])
        self.assertEqual(agent.choose_route(), 0)
        agent.set_candidates([["b"], ["a"]])
        self.assertEqual(agent.q_values, [-10, -60])
        self.assertEqual(agent.counts, [1, 2])

    def test_no_candidates_and_reject_duplicate_routes(self):
        with self.assertRaises(ValueError):
            EpsilonGreedy([])
        with self.assertRaises(ValueError):
            EpsilonGreedy([["a"], ["a"]])
        empty = EpsilonGreedy()
        with self.assertRaises(RuntimeError):
            empty.choose_route()


if __name__ == "__main__":
    unittest.main(verbosity=2)
