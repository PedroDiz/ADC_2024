import csv
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict, deque
from community import community_louvain
import time
import random


# Function to read the edges from the CSV file and build the graph
def build_graph_from_csv(file_path):
    # Create an empty graph
    graph = nx.Graph()

    # Open and read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # Skip the header
        next(reader)
        # Add edges to the graph
        for row in reader:
            id_1, id_2 = int(row[0]), int(row[1])  # Convert IDs to integers
            graph.add_edge(id_1, id_2)  # Add an edge between id_1 and id_2

    return graph


def average_degree_in_graph(G):
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    return avg_degree

def calculate_graph_metrics(G):
    if nx.is_connected(G):
       avg_path_length = nx.average_shortest_path_length(G)
       diameter = nx.diameter(G)
    else:
       avg_path_length = float('inf')
       diameter = float('inf')

    clustering_coeff = nx.average_clustering(G)
    highest_degree_node = max(G.nodes, key=G.degree)
    avg_degree = average_degree_in_graph(G)
    return avg_path_length, diameter, clustering_coeff, highest_degree_node, avg_degree

def get_degree_frequency(G):
    degrees = [G.degree(n) for n in G.nodes]
    degree_count = Counter(degrees)
    x, y = zip(*degree_count.items())
    return x,y

def calculate_density(subgraph):
    n = subgraph.number_of_nodes()
    if n <= 1:
        return 0  # Avoid division by zero
    m = subgraph.number_of_edges()
    return (2 * m) / (n * (n - 1))


def analyze_single_community(graph, community_nodes, partition):

    intra_edges = 0
    inter_edges = 0
    node_community_edges = defaultdict(set)

    for node in community_nodes:
        for neighbor in graph.neighbors(node):
            if neighbor in community_nodes:
                intra_edges += 1  # Edge within the community
            else:
                inter_edges += 1  # Edge connecting to other communities
                community_id = partition.get(neighbor, None)
                node_community_edges[node].add(community_id)

    # Adjust intra-community edges count for undirected graphs
    if not graph.is_directed():
        intra_edges //= 2

    # Identify bridge nodes
    bridge_nodes = 0
    for node in node_community_edges:
        # Check if the node connects to any other community
        if len(node_community_edges[node]) > 0:
            bridge_nodes += 1


    return intra_edges, inter_edges, bridge_nodes


def calculate_internal_degree(subgraph):
    dictionary = {}
    for node in subgraph.nodes:
        sum = 0
        for _ in subgraph.neighbors(node):
            sum += 1
        dictionary[node] = sum

    return dictionary


def calculate_external_degree(graph, subgraph):
    dictionary = {}
    for node in subgraph.nodes:
        sum = 0
        for neighbor in graph.neighbors(node):
            if neighbor not in subgraph.nodes:
                sum += 1
        dictionary[node] = sum
    return dictionary


def calculate_conductance(G, S):
    s_bar = set(G.nodes()) - S.nodes()

    # Calculate cut size: number of edges between S and S_bar
    cut_size = 0
    for u in S.nodes():
        for v in G.neighbors(u):
            if v in s_bar:
                cut_size += 1

    # Print cut size
    print(f"Cut size: {cut_size}")

    # Calculate volume of S (sum of degrees of nodes in S)
    vol_S = 0
    for u in S:
        vol_S += G.degree[u]


    print(f"Volume of S: {vol_S}")

    # Calculate volume of S_bar (sum of degrees of nodes in S_bar)
    vol_S_bar = 0
    for u in s_bar:
        vol_S_bar += G.degree[u]

    print(f"Volume of S_bar: {vol_S_bar}")
    # Avoid division by zero
    if min(vol_S, vol_S_bar) == 0:
        return 0.0  # Conductance is 0 if S or S_bar is disconnected

    # Conductance formula
    return cut_size / min(vol_S, vol_S_bar)


def targeted_attacks(G):
    N = G.number_of_nodes()
    print(f"Number of nodes: {N}")
    C = G.copy()
    targeted_attack_core_proportions = []
    targeted_attack_connected_components = []
    for i in range(100):

        core = next(nx.connected_components(C))
        core_proportion = len(core) / N
        ncc = nx.number_connected_components(C)

        targeted_attack_core_proportions.append(core_proportion)
        targeted_attack_connected_components.append(ncc)
        if C.number_of_nodes() > 1:
            # re-sort
            nodes_sorted_by_degree = sorted(C.nodes, key=C.degree, reverse=True)
            node_to_remove = nodes_sorted_by_degree[0]  # get the first <- Largest degree node
            C.remove_nodes_from([node_to_remove])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(list(range(100)), targeted_attack_core_proportions)
    ax1.set_title("Targeted attack")
    ax1.set_xlabel("Number of nodes removed")
    ax1.set_ylabel("Proportion of nodes in core")

    ax2.plot(list(range(100)), targeted_attack_connected_components)
    ax2.set_title("Targeted attack")
    ax2.set_xlabel("Number of nodes removed")
    ax2.set_ylabel("N. Connected Components")

    plt.tight_layout()
    plt.show()


def modified_bfs_partition(graph, percentage=0.2, seed =None, seed_strategy="random"):
    total_nodes = len(graph.nodes())
    target_size = int(total_nodes * percentage)

    # Choose seed node if not explicitly provided
    if seed is None:
        if seed_strategy == "random":
            seed = random.choice(list(graph.nodes()))
        elif seed_strategy == "lowest_degree":
            seed = min(graph.degree, key=lambda x: x[1])[0]  # Get node with the lowest degree
        else:
            raise ValueError("Invalid seed_strategy. Use 'random' or 'lowest_degree'.")

    print(f"Using seed node: {seed}")

    queue = deque([seed])
    visited = set()

    while queue and len(visited) < target_size:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor not in visited and len(visited) < target_size:
                    queue.append(neighbor)

    # Create a subgraph with the visited nodes
    partition_subgraph = graph.subgraph(visited).copy()
    return partition_subgraph


def random_sampling_partition(graph, percentage=0.2):
    total_nodes = len(graph.nodes())
    target_size = int(total_nodes * percentage)

    # Randomly sample nodes (convert graph.nodes() to a list)
    sampled_nodes = random.sample(list(graph.nodes()), target_size)

    # Create and return a subgraph with the sampled nodes
    subgraph = graph.subgraph(sampled_nodes).copy()  # .copy() ensures a new graph object
    return subgraph


def degree_based_partition(graph, percentage=0.2):
    """
    Partition the graph by selecting nodes with the highest degrees.
    """
    total_nodes = len(graph.nodes())
    target_size = int(total_nodes * percentage)
    highest_degree_node = max(graph.nodes, key=graph.degree)
    degree = graph.degree(highest_degree_node)
    print(f"Selected seed node: {highest_degree_node}")
    print(f"Seed node degree: {degree}")

    # Sort nodes by degree in descending order
    sorted_nodes = sorted(graph.nodes, key=graph.degree, reverse=True)[:target_size]

    # Print out the nodes

    print(f"Selected nodes: {sorted_nodes}")
    subgraph = graph.subgraph(sorted_nodes).copy()

    return subgraph


def export_partition_to_gephi(graph, file_name):
    nx.write_gexf(graph, file_name)
    print(f"Partitioned graph saved to {file_name}")

def gather_and_print_metrics(graph):
    graph_density = calculate_density(graph)
    graph_avg_shortest_path, graph_diameter, graph_clustering_coeff, graph_highest_degree_node, graph_avg_degree = calculate_graph_metrics(graph)
    graph_assortativity = nx.degree_assortativity_coefficient(graph)

    print("Graph information:")
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Average Degree: {graph_avg_degree}")
    print(f"Assortativity: {graph_assortativity}")
    print(f"Clustering Coefficient: {graph_clustering_coeff}")
    print(
        f"Highest Degree Node: {graph_highest_degree_node} (Degree: {graph.degree(graph_highest_degree_node)})")
    print(f"Average Path Length: {graph_avg_shortest_path}")
    print(f"Diameter: {graph_diameter}")
    print(f"Density: {graph_density:.4f}")

def extract_communities_subgraphs(graph, partition):
    communities = {}
    for node, community_id in partition.items():
        if community_id not in communities:
            communities[community_id] = []
        communities[community_id].append(node)

    return [graph.subgraph(nodes) for nodes in communities.values()]


def plot_ws_properties(n, k, p_values):
    clustering_coeffs = []
    path_lengths = []

    for p in p_values:
        ws_graph = nx.watts_strogatz_graph(n, k, p)
        clustering_coeffs.append(nx.average_clustering(ws_graph))
        if nx.is_connected(ws_graph):
            path_lengths.append(nx.average_shortest_path_length(ws_graph))
        else:
            path_lengths.append(float('inf'))

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, clustering_coeffs, label="Clustering Coefficient", marker='o')
    plt.plot(p_values, path_lengths, label="Average Path Length", marker='x')
    plt.xscale('log')
    plt.xlabel("Rewiring Probability (p)")
    plt.ylabel("Metric Value")
    plt.title("Clustering Coefficient and Average Path Length vs. Rewiring Probability")
    plt.legend()
    plt.grid(True)
    plt.show()

def analyze_communities(subgraphs, graph_partition, partition) :
    for i, subgraph in enumerate(subgraphs):
        print(f"\nCommunity {i + 1}:")

        # Calculate internal and external degrees for each node
        internal_degree = calculate_internal_degree(subgraph)
        external_degree = calculate_external_degree(graph_partition, subgraph)

        Nc = subgraph.number_of_nodes()
        separation = sum(external_degree.values()) / Nc if Nc > 0 else 0  # External links per node

        # Calculate density
        density = calculate_density(subgraph)

        # Calculate conductance
        conductance = calculate_conductance(graph_partition, subgraph)

        # Calculate intra, inter edges and bridge nodes
        intra_edges, inter_edges, bridge_nodes = analyze_single_community(graph_partition, subgraph.nodes, partition)

        # Calculate assortativity
        assortativity = nx.degree_assortativity_coefficient(subgraph)

        # Compute metrics
        avg_path_length, diameter, clustering_coeff, highest_degree_node, avg_degree = calculate_graph_metrics(subgraph)

        print(f"  - Number of Nodes: {subgraph.number_of_nodes()}")
        print(f"  - Number of Edges: {subgraph.number_of_edges()}")
        print(f"  - Number of intra-community edges: {intra_edges}")
        print(f"  - Number of inter-community edges: {inter_edges}")
        print(f"  - Number of bridge nodes: {bridge_nodes}")
        print(f"  - Average Degree: {avg_degree}")
        print(f"  - Assortativity: {assortativity}")
        print(f"  - Average Path Length: {avg_path_length}")
        print(f"  - Diameter: {diameter}")
        print(f"  - Clustering Coefficient: {clustering_coeff}")
        print(f"  - Highest Degree Node: {highest_degree_node} (Degree: {subgraph.degree(highest_degree_node)})")
        print(f"  - Density: {density:.4f}")
        print(f"  - Conductance: {conductance:.4f}")
        print(f"  - Separation: {separation:.4f}")
        print(f"  - Internal Degree: {internal_degree}")
        print(f"  - External Degree: {external_degree}")


def create_and_analyze_regular_models():
    er_graph = nx.erdos_renyi_graph(7540, 0.0033)
    plot_degree_distribution(er_graph)
    gather_and_print_metrics(er_graph)

    plot_degree_distribution(er_graph)

    print("Watts Strogatz Graph")
    ws_graph = nx.watts_strogatz_graph(7540, 24, 0.1)
    gather_and_print_metrics(ws_graph)
    plot_degree_distribution(ws_graph)

    print("Barabasi Albert Graph")
    ba_graph = nx.barabasi_albert_graph(7540, 12)
    gather_and_print_metrics(ba_graph)
    plot_degree_distribution(ba_graph)


def calculate_top_10_centralities(graph):
    degree_centrality = nx.degree_centrality(graph)
    eigenvector_centrality = nx.eigenvector_centrality(graph)
    closeness_centrality = nx.closeness_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph)

    sorted_nodes = sorted(graph.nodes, key=graph.degree, reverse=True)[:10]

    for node in sorted_nodes:
        print(f"Node {node}:")
        print(f"  - Degree Centrality: {degree_centrality[node]:.4f}")
        print(f"  - Eigenvector Centrality: {eigenvector_centrality[node]:.4f}")
        print(f"  - Closeness Centrality: {closeness_centrality[node]:.4f}")
        print(f"  - Betweenness Centrality: {betweenness_centrality[node]:.4f}")


def plot_degree_distribution(graph):
    # Get the degree of each node in the graph
    degrees = [deg for node, deg in graph.degree()]

    # Calculate the degree distribution
    degree_count = {}
    for degree in degrees:
        degree_count[degree] = degree_count.get(degree, 0) + 1

    # Sort the degree distribution by degree
    sorted_degrees = sorted(degree_count.items())
    degree_values, frequency = zip(*sorted_degrees)

    # Plot degree distribution with logarithmic axes
    plt.figure(figsize=(8, 6))
    plt.loglog(degree_values, frequency, marker='o', linestyle='None', color='b')

    plt.xlabel('Degree (Log Scale)', fontsize = 18)
    plt.ylabel('Frequency (Log Scale)', fontsize = 18)
    plt.title('Degree Distribution', fontsize = 20)
    plt.grid(True)
    plt.show()


# Path to the CSV file
csv_file_path = 'musae_git_edges.csv'

# Build the graph
graph = build_graph_from_csv(csv_file_path)

graph_partition_bfs_lowest_degree = modified_bfs_partition(graph, seed_strategy="lowest_degree")
partition = community_louvain.best_partition(graph_partition_bfs_lowest_degree)
bfs_lowest_degree_subgraphs = extract_communities_subgraphs(graph_partition_bfs_lowest_degree, partition)
analyze_communities(bfs_lowest_degree_subgraphs, graph_partition_bfs_lowest_degree, partition)
