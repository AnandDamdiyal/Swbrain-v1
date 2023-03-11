import networkx as nx

def load_graph_from_adjacency(adjacency_matrix):
    """
    Loads a graph from an adjacency matrix.
    """
    return nx.from_numpy_matrix(adjacency_matrix)

def load_graph_from_edgelist(edgelist_file):
    """
    Loads a graph from an edgelist file.
    """
    return nx.read_edgelist(edgelist_file)

def calculate_degree_centrality(graph):
    """
    Calculates the degree centrality for each node in the graph.
    """
    return nx.degree_centrality(graph)

def calculate_betweenness_centrality(graph):
    """
    Calculates the betweenness centrality for each node in the graph.
    """
    return nx.betweenness_centrality(graph)

def calculate_eigenvector_centrality(graph):
    """
    Calculates the eigenvector centrality for each node in the graph.
    """
    return nx.eigenvector_centrality(graph)

def calculate_clustering_coefficient(graph):
    """
    Calculates the clustering coefficient for each node in the graph.
    """
    return nx.clustering(graph)
