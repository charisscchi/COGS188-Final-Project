
import random


class EpsilonGreedy:

    def __init__(self, routes, epsilon=0.1, seed=42):

        if not routes:
            raise ValueError("At least one route is required.")

        if not 0 <= epsilon <= 1:
            raise ValueError("Epsilon must be between 0 and 1.")

        self.routes = routes
        self.epsilon = epsilon
        self.rng = random.Random(seed)

        # Estimated reward for each candidate route
        self.q_values = [0.0] * len(routes)

        # Number of times each route has been tested
        self.counts = [0] * len(routes)

    def choose_route(self):

        # Exploration: randomly select a route
        if self.rng.random() < self.epsilon:
            return self.rng.randrange(len(self.routes))

        # Exploitation: select the highest estimated reward
        best_value = max(self.q_values)

        best_actions = [
            i for i, value in enumerate(self.q_values)
            if value == best_value
        ]

        # Random tie-breaking
        return self.rng.choice(best_actions)

    def update(self, action, waiting_time):

        # Shorter passenger waiting time gives a higher reward
        reward = -waiting_time

        self.counts[action] += 1

        # Incremental sample-average update
        self.q_values[action] += (
            reward - self.q_values[action]
        ) / self.counts[action]

    def best_route(self):

        if not all(self.counts):
            raise RuntimeError(
                "Test all candidate routes before selecting the best."
            )

        best_action = max(
            range(len(self.routes)),
            key=lambda i: self.q_values[i]
        )

        return self.routes[best_action]


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
