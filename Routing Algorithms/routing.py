import heapq
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from itertools import count
from numbers import Real
from typing import TypeVar

Node = TypeVar("Node", bound=Hashable)


def is_finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def checked_cost(value: float, label: str = "Travel cost") -> float:
    if not is_finite_number(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return float(value)


def astar(
    graph: Mapping[Node, Sequence[tuple[Node, float]]],
    start: Node,
    goal: Node,
    heuristic: Mapping[Node, float] | Callable[[Node], float] | None = None,
) -> tuple[list[Node] | None, float]:
    """Minimize g + h using costs in seconds, or return (None, inf).

    The heuristic must lower-bound the remaining cost for this goal and
    return zero at the goal. The default zero heuristic is Dijkstra-equivalent.
    A better g value reopens a state, including under inconsistent heuristics.
    """
    known_nodes = set(graph)
    for neighbors in graph.values():
        for neighbor, cost in neighbors:
            known_nodes.add(neighbor)
            checked_cost(cost)
    if start not in known_nodes or goal not in known_nodes:
        raise ValueError("Start and goal must be known graph nodes.")

    def estimate(node: Node) -> float:
        if heuristic is None:
            return 0.0
        try:
            value = heuristic(node) if callable(heuristic) else heuristic[node]
        except KeyError as exc:
            raise ValueError(f"Missing heuristic for {node!r}.") from exc
        return checked_cost(value, "Heuristic")

    if estimate(goal) != 0:
        raise ValueError("The heuristic at the goal must be zero.")

    sequence = count()
    frontier = [(estimate(start), 0.0, next(sequence), start)]
    best_cost = {start: 0.0}
    parent = {}

    while frontier:
        _, current_cost, _, current = heapq.heappop(frontier)

        # Skip outdated entries
        if current_cost > best_cost[current]:
            continue

        # Reconstruct the route when the goal is reached
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1], current_cost

        # Explore neighboring roads
        for neighbor, travel_time in graph.get(current, ()):
            new_cost = checked_cost(current_cost + float(travel_time))

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = current

                estimated_cost = checked_cost(new_cost + estimate(neighbor), "Search priority")

                heapq.heappush(
                    frontier,
                    (estimated_cost, new_cost, next(sequence), neighbor)
                )

    # No valid route exists
    return None, float("inf")


def path_cost(
    graph: Mapping[Node, Sequence[tuple[Node, float]]], path: Sequence[Node],
) -> float:
    """Score a nonempty path whose transitions are present and unambiguous."""
    if not path:
        raise ValueError("An empty path has no travel time.")
    if len(path) == 1 and path[0] not in graph:
        raise ValueError("Path contains an unknown node.")
    total = 0.0
    for start, end in zip(path, path[1:]):
        weights = [cost for neighbor, cost in graph.get(start, ()) if neighbor == end]
        if len(weights) != 1:
            raise ValueError(f"Missing or ambiguous transition {start!r} -> {end!r}.")
        total = checked_cost(total + checked_cost(weights[0]))
    return total


# Initial test with a small example road network
if __name__ == "__main__":

    graph = {
        "S": [("A", 30), ("B", 30)],
        "A": [("G", 30)],
        "B": [("C", 40)],
        "C": [("G", 30)],
        "G": []
    }

    heuristic = {
        "S": 60,
        "A": 30,
        "B": 70,
        "C": 30,
        "G": 0
    }

    path, cost = astar(graph, "S", "G", heuristic)

    print("Static A* Results")
    print("-" * 40)
    print("Route:", " -> ".join(path))
    print("Total travel time:", cost, "seconds")

    # Verify the algorithm's result
    assert path == ["S", "A", "G"]
    assert cost == 60

    print("Static A* test passed!")
