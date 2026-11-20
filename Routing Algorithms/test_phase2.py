import csv
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from experiment import ExperimentConfig, FrozenSelector, pick_epsilon, run_episode, run_experiment
from experiment_outputs import paired_test_rows, prepare_output_dir, route_counts, summarize, write_outputs
from route_planner import plan_route
from synthetic_environment import (
    SHORT_ROAD, OutcomeModel, PickupOutcome, evaluate_synthetic, make_scenarios, synthetic_network,
)
from synthetic_traffic import CONGESTION_MULTIPLIERS, SyntheticTraffic
from test_routing import synthetic_network as provider_network
from traffic_data import historical_snapshot, load_traffic_data


REQUEST_TIME = datetime(2024, 8, 1, 12, tzinfo=timezone.utc)


class TestSyntheticTraffic(unittest.TestCase):
    def setUp(self):
        self.network = provider_network()

    def plan(self, policy, traffic):
        return plan_route(
            self.network, "toy_start", "toy_pickup",
            request_time=REQUEST_TIME, method=policy, traffic=traffic,
        )

    def test_all_multipliers_apply_only_to_selected_directed_road(self):
        for level, factor in zip(CONGESTION_MULTIPLIERS, (1, 0.8, 0.5, 0.2), strict=True):
            with self.subTest(level=level):
                snapshot = SyntheticTraffic(self.network, level, affected_roads=["toy_fast"])(REQUEST_TIME)
                self.assertEqual(snapshot.request_time, REQUEST_TIME)
                self.assertIn("preliminary synthetic", snapshot.source)
                self.assertEqual(snapshot.age_seconds("toy_fast"), 0)
                self.assertEqual(snapshot.observations["toy_fast"].speed_mps, 10 * factor)
                for road_id in self.network.roads.keys() - {"toy_fast"}:
                    self.assertEqual(snapshot.observations[road_id].speed_mps, 10)

    def test_seeded_pattern_matches_across_network_input_orders(self):
        options = dict(level="heavy", affected_count=3, seed=427)
        first = SyntheticTraffic(self.network, **options)
        second = SyntheticTraffic(provider_network(reverse_input=True), **options)
        self.assertEqual(len(first.affected_roads), 3)
        self.assertEqual(first(REQUEST_TIME), second(REQUEST_TIME))
        self.assertEqual(first(REQUEST_TIME), first(REQUEST_TIME))

    def test_closures_are_distinct_from_missing_and_reduced_speeds(self):
        provider = SyntheticTraffic(
            self.network, "heavy", affected_roads=["toy_fast"],
            closed_roads=["toy_alt"], missing_roads=["toy_start"],
        )
        snapshot = provider(REQUEST_TIME)
        self.assertEqual(snapshot.observations["toy_alt"].status, "closed")
        self.assertEqual(snapshot.observations["toy_start"].status, "missing")
        self.assertEqual(snapshot.observations["toy_fast"].status, "available")
        result = self.plan("traffic_aware", provider)
        self.assertEqual(result.road_ids[1], "toy_fast")
        self.assertIn("toy_start", result.diagnostics.fallback_roads)
        self.assertNotIn("toy_alt", result.diagnostics.fallback_roads)

    def test_invalid_parameters(self):
        for options in (
            {"level": "rush-hour"}, {"affected_count": 99}, {"affected_count": -1},
            {"affected_count": True}, {"affected_count": 2, "affected_roads": ["toy_fast"]},
            {"seed": 1.2}, {"affected_roads": "toy_fast"}, {"affected_roads": ["unknown"]},
            {"affected_roads": ["toy_fast", "toy_fast"]}, {"pattern": ""},
            {"closed_roads": ["toy_fast"], "missing_roads": ["toy_fast"]},
            {"road_multipliers": {"toy_fast": 0}}, {"road_multipliers": {"toy_fast": 2}},
            {"road_multipliers": {"toy_fast": float("nan")}},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SyntheticTraffic(self.network, **options)

    def test_no_congestion_parity_and_localized_detour(self):
        free = SyntheticTraffic(self.network)
        static = self.plan("static", free)
        aware = self.plan("traffic_aware", free)
        self.assertEqual((static.road_ids, static.estimated_time_s),
                         (aware.road_ids, aware.estimated_time_s))
        heavy = SyntheticTraffic(self.network, "heavy", affected_roads=["toy_fast"])
        self.assertEqual(self.plan("static", heavy).road_ids, static.road_ids)
        self.assertEqual(self.plan("traffic_aware", heavy).road_ids[1], "toy_alt")

    def test_historical_and_synthetic_providers_share_the_planner_contract(self):
        fictional_data = load_traffic_data(StringIO(
            "LINK_ID,SPEED,DATA_AS_OF\nfixture-link,2,2024-08-01T12:00:00Z\n"
        ))

        def historical(when):
            return historical_snapshot(
                fictional_data, when, {"toy_fast": "fixture-link"}, meters_per_second_per_unit=1,
            )

        synthetic = SyntheticTraffic(self.network, "heavy", affected_roads=["toy_fast"])
        first = self.plan("traffic_aware", historical)
        second = self.plan("traffic_aware", synthetic)
        self.assertEqual((first.road_ids, first.estimated_time_s),
                         (second.road_ids, second.estimated_time_s))
        self.assertNotEqual(first.diagnostics.source, second.diagnostics.source)


class TestSyntheticEnvironment(unittest.TestCase):
    def setUp(self):
        self.network = synthetic_network()
        self.levels = tuple(CONGESTION_MULTIPLIERS)

    def scenarios(self, *, split="test", model=OutcomeModel(), size=17):
        return make_scenarios(self.network, self.levels, split, size, 11, model)

    def route(self, scenario, method):
        return run_episode(
            self.network, scenario.request, method,
            lambda request, route: evaluate_synthetic(self.network, scenario, route),
        )

    def test_balanced_seeded_templates_and_independent_split_streams(self):
        self.assertEqual(self.scenarios(), self.scenarios())
        splits = [self.scenarios(split=split) for split in ("train", "validation", "test")]
        self.assertEqual(len({rows[0].split_seed for rows in splits}), 3)
        self.assertEqual(len({item.scenario_seed for rows in splits for item in rows}), 51)
        for rows in splits:
            self.assertEqual(len({(row.condition, row.pattern, row.request.pickup) for row in rows}), 17)
        with self.assertRaises(ValueError):
            self.scenarios(size=0)
        with self.assertRaises(ValueError):
            OutcomeModel(realized_speed_noise=1)

    def test_perfect_control_agrees_with_prediction_and_free_flow_parity(self):
        for scenario in self.scenarios():
            if scenario.pattern != "perfect":
                continue
            with self.subTest(condition=scenario.condition):
                aware = self.route(scenario, "traffic_aware")
                static = self.route(scenario, "static")
                self.assertEqual(aware.plan.estimated_time_s, aware.outcome.pickup_time_s)
                if scenario.condition == "free_flow":
                    self.assertEqual(aware.plan.road_ids, static.plan.road_ids)
                    self.assertEqual(aware.outcome, static.outcome)
                if scenario.condition == "heavy":
                    self.assertNotEqual(aware.plan.road_ids, static.plan.road_ids)
                    self.assertLess(aware.outcome.pickup_time_s, static.outcome.pickup_time_s)

    def test_hidden_state_changes_outcomes_without_changing_planning(self):
        scenario = next(row for row in self.scenarios() if row.pattern == "perfect")
        original = self.route(scenario, "traffic_aware")
        changed = replace(scenario, realized=replace(scenario.realized, dispatch_delay_s=3))
        new = self.route(changed, "traffic_aware")
        self.assertEqual(new.plan, original.plan)
        self.assertAlmostEqual(new.outcome.pickup_time_s - original.outcome.pickup_time_s, 3)
        self.assertEqual(new.outcome.route_travel_s, original.outcome.route_travel_s)
        self.assertEqual(self.route(changed, "traffic_aware").outcome, new.outcome)

    def test_noisy_missing_forecast_does_not_leak_nominal_condition_to_bandit(self):
        scenarios = self.scenarios(model=OutcomeModel(missing_probability=1))
        matches = [row for row in scenarios if row.pattern == "noisy"
                   and row.request.pickup == scenarios[0].request.pickup]
        self.assertEqual(len({row.condition for row in matches}), 4)
        self.assertEqual(len({row.request.context for row in matches}), 1)
        self.assertEqual(len({row.request.traffic.source for row in matches}), 1)

    def test_outcomes_are_policy_independent_for_the_same_route(self):
        matched = 0
        for scenario in self.scenarios():
            aware = self.route(scenario, "traffic_aware")
            static = self.route(scenario, "static")
            if static.plan.road_ids == aware.plan.road_ids and static.plan.road_ids:
                self.assertEqual(static.outcome, aware.outcome)
                matched += 1
            self.assertEqual(self.route(scenario, "traffic_aware").outcome, aware.outcome)
        self.assertGreater(matched, 0)

    def test_known_closure_hidden_closure_and_timeout_have_no_wait_reward(self):
        scenarios = self.scenarios()
        closed = next(row for row in scenarios if row.pattern == "known_closure")
        self.assertEqual(self.route(closed, "static").outcome, PickupOutcome("unreachable"))
        free = next(row for row in scenarios if row.pattern == "perfect" and row.condition == "free_flow")
        blocked = replace(free, realized=replace(free.realized, closed_roads=frozenset({SHORT_ROAD})))
        self.assertEqual(self.route(blocked, "static").outcome, PickupOutcome("blocked"))
        timed = replace(free, realized=replace(free.realized, timeout_s=1))
        self.assertEqual(self.route(timed, "static").outcome, PickupOutcome("timeout"))
        with self.assertRaises(ValueError):
            PickupOutcome("timeout", 0)

    def test_outcome_callback_can_accept_external_waits(self):
        results = []

        def feedback(actual_request, plan):
            results.append((actual_request, plan))
            return PickupOutcome("completed", 47, 42)

        request = next(row.request for row in self.scenarios() if row.pattern == "perfect")
        episode = run_episode(self.network, request, "traffic_aware", feedback)
        self.assertEqual(results, [(request, episode.plan)])
        self.assertEqual(episode.outcome.pickup_time_s, 47)
        self.assertNotEqual(episode.plan.estimated_time_s, episode.outcome.pickup_time_s)

    def test_route_selector_rejects_invalid_indices_before_scoring(self):
        class InvalidSelector(FrozenSelector):
            def choose_route(self):
                return -1

        request = next(row.request for row in self.scenarios() if row.pattern == "perfect")
        with self.assertRaisesRegex(ValueError, "candidate index"):
            run_episode(self.network, request, "epsilon_greedy",
                        lambda _request, _plan: self.fail("Invalid route should not be scored"),
                        selector=InvalidSelector())


class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ExperimentConfig(seeds=(11,), train_episodes=68, validation_scenarios=17,
                                      test_scenarios=17, epsilons=(0.0, 0.2))
        cls.experiment = run_experiment(cls.config)

    def test_bandit_reuses_values_by_context_and_keeps_candidate_identity(self):
        train = [row for row in self.experiment.episodes if row.split == "train" and row.epsilon == 0]
        by_context = {}
        for row in train:
            by_context.setdefault(row.context, []).append(row)
        comparable = next(rows for rows in by_context.values()
                          if rows[0].pattern == "perfect" and rows[0].condition == "free_flow")
        self.assertEqual([row.selection_mode for row in comparable],
                         ["initial_trial", "initial_trial", "initial_trial", "epsilon_rule"])
        self.assertEqual(len({row.selected_route for row in comparable[:3]}), 3)
        self.assertEqual(comparable[-1].reward, max(row.reward for row in comparable[:3]))
        self.assertEqual(comparable[-1].cumulative_context_reward, sum(row.reward for row in comparable))
        self.assertEqual(len({row.agent_seed for row in comparable}), 1)

        known = ("x", "y")
        unseen = ("x", "z")
        values = {known: -40.0}
        frozen = FrozenSelector(values)
        values[known] = 0.0
        frozen.set_candidates([unseen, known])
        self.assertEqual(frozen.choose_route(), 1)
        self.assertEqual(frozen.estimates[known], -40.0)
        frozen.set_candidates([unseen])
        self.assertEqual((frozen.choose_route(), frozen.mode), (0, "first_candidate_fallback"))

    def test_frozen_test_is_matched_has_no_updates_and_is_independent_of_test_size(self):
        test = [row for row in self.experiment.episodes if row.split == "test"]
        self.assertEqual({row.epsilon for row in test if row.method == "epsilon_greedy"},
                         {self.experiment.chosen_epsilon})
        self.assertTrue(all(row.reward is None and row.cumulative_context_reward is None for row in test))
        self.assertFalse(any(row.selection_mode in {"initial_trial", "epsilon_rule"} for row in test))
        for scenario_id in {row.scenario_id for row in test}:
            matched = [row for row in test if row.scenario_id == scenario_id]
            self.assertEqual(len(matched), 4)
            self.assertEqual(len({(row.scenario_seed, row.context, row.request_time) for row in matched}), 1)
        changed_test_size = run_experiment(replace(self.config, test_scenarios=23))
        self.assertEqual(changed_test_size.chosen_epsilon, self.experiment.chosen_epsilon)
        self.assertEqual(
            [replace(row, experiment_id="") for row in changed_test_size.learned_routes],
            [replace(row, experiment_id="") for row in self.experiment.learned_routes],
        )

    def test_failed_trials_do_not_count_as_reward_samples(self):
        training = [row for row in self.experiment.episodes if row.split == "train"]
        self.assertTrue(any(row.status != "completed" for row in training))
        self.assertTrue(all(row.reward is None for row in training if row.status != "completed"))
        self.assertEqual(sum(row.reward_samples for row in self.experiment.learned_routes),
                         sum(row.status == "completed" for row in training))

    def test_validation_criterion_prefers_completion_and_ignores_test(self):
        row = next(row for row in self.experiment.episodes if row.method == "epsilon_greedy")
        rows = (
            replace(row, split="validation", epsilon=0, status="completed", modeled_pickup_s=50),
            replace(row, split="validation", epsilon=0, status="completed", modeled_pickup_s=60),
            replace(row, split="validation", epsilon=0.2, status="completed", modeled_pickup_s=1),
            replace(row, split="validation", epsilon=0.2, status="timeout", modeled_pickup_s=None),
            replace(row, split="test", epsilon=0.2, status="completed", modeled_pickup_s=0),
        )
        self.assertEqual(pick_epsilon(rows, [0.2, 0]), 0)

    def test_summary_and_pair_differences_keep_failure_denominators(self):
        row = next(row for row in self.experiment.episodes if row.split == "test" and row.method == "static")
        rows = tuple(
            replace(row, scenario_id=str(index), status=status, modeled_pickup_s=wait)
            for index, (status, wait) in enumerate((
                ("completed", 10), ("completed", 20), ("completed", 40), ("timeout", None),
            ))
        )
        pooled = summarize(rows)[0]
        self.assertEqual(pooled["completion_rate"], 0.75)
        self.assertEqual(pooled["failure_rate"], 0.25)
        self.assertEqual(pooled["timeout_rate"], 0.25)
        self.assertAlmostEqual(pooled["mean_pickup_s"], 70 / 3)
        self.assertEqual(pooled["median_pickup_s"], 20)
        self.assertAlmostEqual(pooled["p90_pickup_s"], 36)
        alternatives = (
            replace(rows[0], method="traffic_aware", modeled_pickup_s=8),
            replace(rows[3], method="traffic_aware", status="completed", modeled_pickup_s=11),
        )
        first, failure = paired_test_rows((*rows, *alternatives))
        self.assertEqual(first["delta_pickup_s"], -2)
        self.assertEqual((failure["method_status"], failure["reference_status"]), ("completed", "timeout"))
        self.assertIsNone(failure["delta_pickup_s"])

    def test_optional_first_candidate_counts_and_new_context_fallback(self):
        counts = route_counts(self.experiment.episodes)
        self.assertTrue(any(row["available_as_candidate"] > row["selected"] for row in counts))
        self.assertTrue(any(row["split"] == "test" and row["method"] == "first_candidate" for row in counts))
        short_run = run_experiment(replace(
            self.config, train_episodes=1, test_scenarios=17, first_candidate=False,
        ))
        self.assertTrue(any(row.selection_mode == "unseen_context_fallback" for row in short_run.episodes))
        self.assertFalse(any(row.method == "first_candidate" for row in short_run.episodes))

    def test_reproduces_serialized_outputs_except_routing_time(self):
        other = run_experiment(self.config)
        root = Path(__file__).resolve().parent
        with TemporaryDirectory(dir=root) as tmp:
            first = prepare_output_dir(Path(tmp) / "first", plots=False)
            second = prepare_output_dir(Path(tmp) / "second", plots=False)
            write_outputs(self.experiment, first, plots=False)
            write_outputs(other, second, plots=False)
            for file in first.iterdir():
                with self.subTest(file=file.name):
                    comparison = second / file.name
                    if file.name in ("episodes.csv", "summary.csv"):
                        def values_without_timing(path):
                            with path.open(newline="") as stream:
                                return [
                                    {key: value for key, value in row.items() if "routing_time" not in key}
                                    for row in csv.DictReader(stream)
                                ]
                        self.assertEqual(values_without_timing(file), values_without_timing(comparison))
                    else:
                        self.assertEqual(file.read_bytes(), comparison.read_bytes())
            with self.assertRaises(ValueError):
                prepare_output_dir(first, plots=False)
            with self.assertRaises(ValueError):
                prepare_output_dir(root.parent / "invalid-output", plots=False)


if __name__ == "__main__":
    unittest.main()
