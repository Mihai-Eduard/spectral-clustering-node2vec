from typing import List, Dict
import pandas as pd
import networkx as nx


def get_edges(data: pd.DataFrame, nodes) -> List:
    """ Given the dataframe with articles and lists return the set of edges
        Args:
        data (pd.DataFrame): The medium dataset
        nodes: dict (node_id: article title)
    Returns:
            edges (List[tuple]): List of edges"""
    edges = []
    buckets = {}
    for index, row in data.iterrows():
        if index not in nodes:
            continue
        for current_list in row["list"].split("; "):
            if current_list not in buckets:
                buckets[current_list] = []
            buckets[current_list].append(index)
    print(f'Number of buckets: {len(buckets)}.')

    for key in buckets:
        similar_nodes = buckets[key]
        for i in range(len(similar_nodes)):
            for j in range(i + 1, len(similar_nodes)):
                edges.append((similar_nodes[i], similar_nodes[j]))
    return edges


def get_nodes(data: pd.DataFrame) -> Dict:
    """ Given the dataframe with articles and lists return the set of nodes
        Args:
        data (pd.DataFrame): The medium dataset
    Returns:
        nodes: dict (node_id: article title)"""
    nodes = {}
    for index, row in data.iterrows():
        if index not in nodes:
            nodes[index] = row["title"]
    print(len(nodes))
    return nodes


def form_graph(data: pd.DataFrame) -> nx.Graph:
    """Forms graph from medium article dataset.
    Args:
        data (pd.DataFrame): The medium dataset
    Returns:
        G (nx.Graph): The graph.
    """
    texts = [x[0] + " " + x[1] for x in zip(data.title, data.subtitle)]
    nodes = get_nodes(data)
    edges = get_edges(data, nodes)
    graph = nx.Graph()
    graph.add_nodes_from(list(nodes.keys()))
    graph.add_edges_from(edges)
    return graph
