import networkx as nx
import numpy as np
import random


def random_walks(graph: nx.Graph, num_walks: int, walk_length: int) -> np.ndarray:
    """Perform random walks on an unweighted graph.

    Args:
        graph (nx.Graph): The graph.
        num_walks (int): The number of random walks for each node.
        walk_length (int): The number of nodes in a random walk.

    Returns:
        np.ndarray: The random walks, shape (n_nodes * num_walks, walk_length)
    """
    nodes = list(graph.nodes())
    result = np.zeros((len(nodes) * num_walks, walk_length))

    for i, node in enumerate(nodes):
        print(f"Random walks for node {i + 1}/{len(nodes)}", end="\r")
        for j in range(num_walks):
            current = node
            for k in range(walk_length):
                current = random.choice(list(graph.neighbors(current)))
                result[i * num_walks + j, k] = current

    return result


def compute_spectral_embeddings(graph: nx.Graph, dim: int) -> np.ndarray:
    """Perform spectral clustering on the graph and compute low-dimensional node representations.
    Does not normalize the Laplacian.

    Args:
        graph (nx.Graph): The graph.
        dim (int): The dimension of representations. This corresponds to the number of eigenvectors used.

    Returns:
        np.ndarray: Node representations (sorted by node ID, ascending), shape (num_nodes, dim).
    """
    adjacency_matrix = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes))

    # make sure the matrix is symmetric
    assert (adjacency_matrix == adjacency_matrix.T).all()

    laplacian_matrix = np.diag(np.sum(adjacency_matrix, axis=1)) - adjacency_matrix
    eigen_results = np.linalg.eigh(laplacian_matrix)
    result = eigen_results[1][:, :dim]

    return result
