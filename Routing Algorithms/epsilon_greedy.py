import random
from collections.abc import Iterable, Sequence
from numbers import Integral

from routing import checked_cost


RouteID = tuple[str, ...]


class EpsilonGreedy:
    """A bandit for one recurring pickup context, keyed by exact road sequence."""

    def __init__(
        self, routes: Iterable[Sequence[str]] | None = None, epsilon: float = 0.1, seed: int = 42,
    ) -> None:
        self.epsilon = checked_cost(epsilon, "Epsilon")
        if self.epsilon > 1:
            raise ValueError("Epsilon must be between 0 and 1.")
        self.rng = random.Random(seed)
        self._counts: dict[RouteID, int] = {}
        self._values: dict[RouteID, float] = {}
        self._routes: tuple[RouteID, ...] = ()
        if routes is not None:
            self.set_candidates(routes)

    def set_candidates(self, routes: Iterable[Sequence[str]]) -> None:
        """Replace the active choices while retaining estimates by route identity."""
        candidates = []
        for route in routes:
            if isinstance(route, str):
                raise ValueError("A candidate must be a sequence of road IDs.")
            route_id = tuple(route)
            if not route_id or any(not isinstance(road, str) or not road.strip() for road in route_id):
                raise ValueError("Each candidate needs a nonempty sequence of road IDs.")
            candidates.append(route_id)
        if not candidates:
            raise ValueError("At least one route is required.")
        if len(set(candidates)) != len(candidates):
            raise ValueError("Candidate routes must be unique.")
        self._routes = tuple(candidates)
        for route in self._routes:
            self._counts.setdefault(route, 0)
            self._values.setdefault(route, 0.0)

    @property
    def routes(self) -> tuple[RouteID, ...]:
        return self._routes

    @property
    def q_values(self) -> list[float]:
        return [self._values[route] for route in self.routes]

    @property
    def counts(self) -> list[int]:
        return [self._counts[route] for route in self.routes]

    def choose_route(self) -> int:
        """Return an index in the current candidate order, without updating rewards."""
        if not self.routes:
            raise RuntimeError("Set feasible candidates before choosing a route.")
        untried = [index for index, count in enumerate(self.counts) if count == 0]
        if untried:
            return self.rng.choice(untried)
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.routes))

        values = self.q_values
        best_value = max(values)
        best_actions = [index for index, value in enumerate(values) if value == best_value]
        return self.rng.choice(best_actions)

    def update(self, action: int, waiting_time: float) -> None:
        """Record immediate feedback using the current candidate order."""
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < len(self.routes):
            raise ValueError("Action must be a valid current candidate index.")
        self.update_route(self.routes[action], waiting_time)

    def update_route(self, route: Sequence[str], waiting_time: float) -> None:
        """Record feedback by road sequence, even if the active ranking changed."""
        route_id = tuple(route)
        if isinstance(route, str) or route_id not in self._counts:
            raise ValueError("Feedback must identify a known candidate route.")
        reward = -checked_cost(waiting_time, "Pickup waiting time")
        self._counts[route_id] += 1
        self._values[route_id] += (reward - self._values[route_id]) / self._counts[route_id]

    def best_route(self) -> list[str]:
        if not self.routes or not all(self.counts):
            raise RuntimeError("Test all candidate routes before selecting the best.")
        best = max(self.routes, key=self._values.__getitem__)
        return list(best)


if __name__ == "__main__":

    # Candidate routes from our example road network
    routes = [
        ["S", "A", "G"],
        ["S", "B", "C", "G"]
    ]

    # Hypothetical waiting times for controlled testing
    waiting_times = [300, 100]

    agent = EpsilonGreedy(
        routes=routes,
        epsilon=0.1,
        seed=42
    )

    episodes = 200
    total_waiting_time = 0

    for episode in range(episodes):

        # Select a candidate route
        action = agent.choose_route()

        # Simulated feedback for this controlled example
        waiting_time = waiting_times[action]

        # Learn from the selected route
        agent.update(action, waiting_time)

        total_waiting_time += waiting_time

    print("Epsilon-Greedy Results")
    print("-" * 40)
    print("Epsilon:", agent.epsilon)
    print("Training episodes:", episodes)

    for i, route in enumerate(routes):

        print("\nRoute", i + 1)
        print("Path:", " -> ".join(route))
        print("Times selected:", agent.counts[i])
        print("Estimated reward:", agent.q_values[i])

    print("\nFinal Results")
    print("-" * 40)
    print("Best route:", " -> ".join(agent.best_route()))

    print(
        "Average waiting time:",
        round(total_waiting_time / episodes, 2),
        "seconds"
    )

    # Verify the learning results
    assert sum(agent.counts) == episodes
    assert agent.q_values == [-300.0, -100.0]
    assert agent.best_route() == ["S", "B", "C", "G"]

    print("\nEpsilon-greedy test passed!")
