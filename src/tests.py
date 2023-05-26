import hde
import numpy as np

# Test bfs distance function
# Example graph represented as a list of NumPy arrays
graph = [
    np.array([1, 2]),    # Node 0, neighbors: 1, 2
    np.array([0, 3, 4]), # Node 1, neighbors: 0, 3, 4
    np.array([0, 5]),    # Node 2, neighbors: 0, 5
    np.array([1]),       # Node 3, neighbor: 1
    np.array([1, 5]),    # Node 4, neighbors: 1, 5
    np.array([2, 4])     # Node 5, neighbors: 2, 4
]

# Call BFS to compute shortest distance between nodes 0 and 5
shortest_distance = hde.bfs_shortest_distance(graph, 0, 5)
print("Shortest distance between nodes 0 and 5:", shortest_distance)