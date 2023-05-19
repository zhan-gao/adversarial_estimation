import numpy as np
import networkx as nx
from scipy.special import expit



# Generate graphs using stochastic block model
def generate_network_sbm(n, lambda_values):
    # Check if the input lambda_values matches with the input n
    assert len(lambda_values) == n, "Length of lambda_values must be equal to n"

    # Create a lambda matrix by adding the lambda_values vectors
    lambda_matrix = np.add.outer(lambda_values, lambda_values)

    # Generate a matrix of e_ij values from a logit distribution
    upper_triangular_indices = np.triu_indices(n, k=1)  # indices for the upper triangle excluding diagonal
    upper_triangle_values = expit(np.random.normal(size=len(upper_triangular_indices[0])))  # values for the upper triangle

    e_matrix = np.zeros((n, n))  # initialize an empty e_matrix
    e_matrix[upper_triangular_indices] = upper_triangle_values  # fill in the upper triangle
    e_matrix = e_matrix + e_matrix.T  # add the transpose to make e_matrix symmetric

    # Fill in the adjacency matrix
    adjacency_matrix = np.greater(lambda_matrix, e_matrix).astype(int)

    # Set the diagonal to zero
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix


n_nodes = 50  # Number of nodes in the graph
n_graphs = 100  # Number of graphs to generate

graphs = []
labels = []

for i in range(n_graphs):

    # Create initial graph with n_edges edges
    graph = nx.barabasi_albert_graph(n_nodes, n_edges)
    label = np.random.randint(2)
    
    # Add random edges to the graph
    n_add_edges = np.random.randint(10, 20)
    for j in range(n_add_edges):
        node1 = np.random.randint(n_nodes)
        node2 = np.random.randint(n_nodes)
        if not graph.has_edge(node1, node2):
            graph.add_edge(node1, node2)
    
    # Append graph and label to the list
    graphs.append(nx.adjacency_matrix(graph).todense())
    labels.append(label)