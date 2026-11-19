
import heapq


def astar(graph, start, goal, heuristic):
    # Priority queue: (estimated total cost, current cost, node)
    frontier = [(heuristic[start], 0, start)]

    # Best known travel cost to each node
    best_cost = {start: 0}

    # Store previous nodes to reconstruct the route
    parent = {start: None}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)

        # Skip outdated entries
        if current_cost > best_cost[current]:
            continue

        # Reconstruct the route when the goal is reached
        if current == goal:
            path = []
            node = goal

            while node is not None:
                path.append(node)
                node = parent[node]

            return path[::-1], current_cost

        # Explore neighboring roads
        for neighbor, travel_time in graph.get(current, []):
            new_cost = current_cost + travel_time

            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                parent[neighbor] = current

                estimated_cost = new_cost + heuristic[neighbor]

                heapq.heappush(
                    frontier,
                    (estimated_cost, new_cost, neighbor)
                )

    # No valid route exists
    return None, float("inf")


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
