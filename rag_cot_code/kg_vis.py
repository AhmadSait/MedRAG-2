import networkx as nx

def load_knowledge_graph(graph_path):
    """
    从 GraphML 文件加载知识图谱，并返回 NetworkX 的图对象
    """
    G = nx.read_graphml(graph_path)
    print(f"加载知识图谱成功，节点数: {len(G.nodes)}")
    return G

def print_all_entities(G):
    """
    遍历图中的所有节点，并打印节点（Entity）的信息
    如果节点包含 'description' 属性，则一起打印
    """
    for node, data in G.nodes(data=True):
        description = data.get('description', '')
        if description:
            print(f"Entity: {node} - {description}")
        else:
            print(f"Entity: {node}")

if __name__ == "__main__":
    # 请将 graph_path 修改为你的 GraphML 文件路径
    graph_path = "/home/plusai/Documents/github/LightRAG/examples/dickens/graph_chunk_entity_relation.graphml"
    G = load_knowledge_graph(graph_path)
    print_all_entities(G)
