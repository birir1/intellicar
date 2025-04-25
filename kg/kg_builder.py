import networkx as nx

def build_knowledge_graph(entities, relationships):
    """
    Builds and returns a simple knowledge graph using NetworkX.
    
    :param entities: List of node labels (e.g., sensors, ECUs, attacks)
    :param relationships: List of tuples (source, relation, target)
    :return: NetworkX DiGraph object
    """
    G = nx.DiGraph()

    for entity in entities:
        G.add_node(entity)

    for src, relation, tgt in relationships:
        G.add_edge(src, tgt, relation=relation)

    return G
