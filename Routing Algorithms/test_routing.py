
import unittest
import math

from routing import astar
from traffic_aware import build_graph
from epsilon_greedy import EpsilonGreedy


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
