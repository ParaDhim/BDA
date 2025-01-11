
import json
from neo4j import GraphDatabase
from pyspark.sql import SparkSession
from itertools import product

# Step 1: Load JSON Data
# Open and read each line as a separate JSON object
with open('train.json', 'r') as file:
    data = [json.loads(line) for line in file]

# Step 2: Neo4j Database Connection
uri = "bolt://localhost:7687"  # update if different
username = "neo4j"  # replace with your Neo4j username
password = "paras2003"  # replace with your Neo4j password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

# Step 3: Create Graph in Neo4j
def create_graph(tx, paper_id, references):
    # Create a paper node
    tx.run("MERGE (p:Paper {id: $paper_id})", paper_id=paper_id)
    # For each reference, create a citation edge
    for ref_id in references:
        tx.run("""
            MERGE (p:Paper {id: $paper_id})
            MERGE (r:Paper {id: $ref_id})
            MERGE (p)-[:CITES]->(r)
            """, paper_id=paper_id, ref_id=ref_id)

# Add data to Neo4j
with driver.session() as session:
    for entry in data:
        paper_id = entry['paper']
        references = entry.get('reference', [])
        session.write_transaction(create_graph, paper_id, references)

# Step 4: Export Neo4j Data to CSV
export_query = """
CALL apoc.export.csv.query("MATCH (p1:Paper)-[:CITES]->(p2:Paper) RETURN p1.id AS paper, p2.id AS reference", "citation_graph.csv", {})
"""
with driver.session() as session:
    session.run(export_query)

# Step 5: Initialize Spark Session
spark = SparkSession.builder.appName("SimRank").getOrCreate()

# Load citation graph CSV into a Spark DataFrame
df = spark.read.csv("citation_graph.csv", header=True, inferSchema=True)
df.show()

# Step 6: Define SimRank Algorithm
def simrank(df, query_nodes, C=0.8, max_iter=10, tol=1e-4):
    # Initialize similarity scores with 1.0 for self-similarity and 0.0 for all others
    sim = {(u, v): 1.0 if u == v else 0.0 for u in query_nodes for v in query_nodes}
    
    # Dictionary to store incoming neighbors for each node
    neighbors = df.rdd.map(lambda row: (row["reference"], row["paper"])) \
                      .groupByKey() \
                      .mapValues(list) \
                      .collectAsMap()
    
    for _ in range(max_iter):
        new_sim = {}
        for u, v in product(query_nodes, repeat=2):
            if u == v:
                new_sim[(u, v)] = 1.0
            else:
                u_neighbors = neighbors.get(u, [])
                v_neighbors = neighbors.get(v, [])
                if u_neighbors and v_neighbors:
                    scale = C / (len(u_neighbors) * len(v_neighbors))
                    new_sim[(u, v)] = scale * sum(sim.get((w, x), 0) for w in u_neighbors for x in v_neighbors)
                else:
                    new_sim[(u, v)] = 0.0
        
        # Check for convergence
        diff = sum(abs(new_sim[(u, v)] - sim[(u, v)]) for u, v in product(query_nodes, repeat=2))
        if diff < tol:
            break
        sim = new_sim
    
    return sim

# Step 7: Run SimRank Algorithm with Different Values of C
results = {}
for C_value in [0.7, 0.8, 0.9]:
    results[C_value] = simrank(df, query_nodes=[2982615777, 1556418098], C=C_value)

# Step 8: Display the Results
for C_value, sim_scores in results.items():
    print(f"Results for C = {C_value}:")
    for (u, v), score in sim_scores.items():
        print(f"Similarity between {u} and {v}: {score}")








# import json
# import pandas as pd
# from py2neo import Graph, Node, Relationship
# from pyspark.sql import SparkSession
# from graphframes import GraphFrame
# from tqdm import tqdm

# # Connect to Neo4j
# neo4j_url = "bolt://localhost:7689"
# neo4j_username = "neo4j"
# neo4j_password = "paras2003"  # replace with your password
# graph = Graph(neo4j_url, auth=(neo4j_username, neo4j_password))

# # Spark Session
# spark = SparkSession.builder \
#     .appName("SimRank") \
#     .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
#     .getOrCreate()

# class Neo4jHandler:
#     def __init__(self, graph, batch_size=1000):
#         self.graph = graph
#         self.batch_size = batch_size

#     def clear_database(self):
#         """Clear database in batches to avoid memory issues"""
#         # Get the total number of nodes in the database
#         total_nodes = self.graph.run("MATCH (n) RETURN count(n) as count").evaluate()
#         deleted = 0
#         with tqdm(total=total_nodes, desc="Clearing database") as pbar:
#             while True:
#                 result = self.graph.run(
#                     f"MATCH (n) WITH n LIMIT {self.batch_size} "
#                     "DETACH DELETE n RETURN count(n)"
#                 ).evaluate()
                
#                 if result == 0:
#                     break
#                 deleted += result
#                 pbar.update(result)
#         print(f"Cleared {deleted} nodes.")

# # Parse JSON data and construct citation graph in Neo4j
# def load_data_to_neo4j(file_path):
#     with open(file_path, 'r') as file:
#         data = [json.loads(line) for line in file]

#     # Clear existing data in Neo4j using the new method
#     handler = Neo4jHandler(graph)
#     handler.clear_database()
    
#     # Adding tqdm for outer loop to track the progress of processing each paper
#     with tqdm(total=len(data), desc="Processing papers") as pbar:
#         for entry in data:
#             paper_id = entry["paper"]
#             references = entry["reference"]
            
#             # Create the citing paper node
#             paper_node = Node("Paper", id=paper_id)
#             graph.merge(paper_node, "Paper", "id")
            
#             # Create cited paper nodes and citation relationships
#             for ref in references:
#                 ref_node = Node("Paper", id=ref)
#                 graph.merge(ref_node, "Paper", "id")
#                 citation = Relationship(paper_node, "CITES", ref_node)
#                 graph.merge(citation)
#             pbar.update(1)

# # Export Neo4j graph to CSV for Spark
# def export_graph_to_csv():
#     # Query nodes and edges from Neo4j
#     nodes = graph.run("MATCH (p:Paper) RETURN p.id AS id").to_data_frame()
#     edges = graph.run("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.id AS src, b.id AS dst").to_data_frame()

#     # Save nodes and edges to CSV
#     nodes.to_csv("/tmp/nodes.csv", index=False)
#     edges.to_csv("/tmp/edges.csv", index=False)

# # Load the graph into Spark
# def load_graph_in_spark():
#     # Load nodes and edges
#     nodes_df = spark.read.csv("/tmp/nodes.csv", header=True)
#     edges_df = spark.read.csv("/tmp/edges.csv", header=True)

#     # Create GraphFrame
#     graph = GraphFrame(nodes_df, edges_df)
#     return graph

# # Compute SimRank
# def simrank(graph, query_nodes, C=0.8, max_iterations=10, tolerance=1e-4):
#     # Initialize similarity matrix
#     similarities = {node.id: 1.0 if node.id == q else 0.0 for node in query_nodes for q in query_nodes}
    
#     # Add tqdm to show the progress of the SimRank iterations
#     with tqdm(total=max_iterations, desc="SimRank Iterations") as pbar:
#         # Run iterative SimRank calculations
#         for iteration in range(max_iterations):
#             new_similarities = {}
#             max_change = 0.0
#             # Add tqdm to the node pair iterations to track progress
#             for u in tqdm(graph.vertices.collect(), desc="Processing node pairs"):
#                 for v in graph.vertices.collect():
#                     if u.id == v.id:
#                         new_similarities[(u.id, v.id)] = 1.0
#                     else:
#                         in_neighbors_u = graph.edges.filter(graph.edges.dst == u.id).select("src").distinct().collect()
#                         in_neighbors_v = graph.edges.filter(graph.edges.dst == v.id).select("src").distinct().collect()
#                         if len(in_neighbors_u) > 0 and len(in_neighbors_v) > 0:
#                             scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
#                             sim_sum = sum(similarities.get((n1.src, n2.src), 0.0) for n1 in in_neighbors_u for n2 in in_neighbors_v)
#                             new_similarity = scale * sim_sum
#                             new_similarities[(u.id, v.id)] = new_similarity
#                             max_change = max(max_change, abs(new_similarity - similarities.get((u.id, v.id), 0.0)))
            
#             similarities = new_similarities
#             pbar.update(1)
#             if max_change < tolerance:
#                 break

#     # Extract top similar nodes for each query node
#     results = {}
#     for q in query_nodes:
#         sorted_sims = sorted([(v, similarities[(q.id, v.id)]) for v in graph.vertices.collect() if v.id != q.id], key=lambda x: -x[1])
#         results[q.id] = sorted_sims[:5]  # Top 5 most similar nodes
#     return results

# # Run the whole process
# def main():
#     # Step 1: Load data into Neo4j
#     load_data_to_neo4j('train.json')
    
#     # Step 2: Export the graph to CSV for Spark
#     export_graph_to_csv()
    
#     # Step 3: Load graph in Spark
#     spark_graph = load_graph_in_spark()
    
#     # Step 4: Compute SimRank with different C values
#     query_nodes = [{"id": "2982615777"}, {"id": "1556418098"}]
#     for C in [0.7, 0.8, 0.9]:
#         simrank_results = simrank(spark_graph, query_nodes, C=C)
#         print(f"SimRank results for C={C}:")
#         for query, similar_nodes in simrank_results.items():
#             print(f"Query Node {query}: {similar_nodes}")

# if __name__ == "__main__":
#     main()



import json
import pandas as pd
from py2neo import Graph, Node, Relationship
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from tqdm import tqdm

# Connect to Neo4j
neo4j_url = "bolt://localhost:7689"
neo4j_username = "neo4j"
neo4j_password = "paras2003"  # replace with your password
graph = Graph(neo4j_url, auth=(neo4j_username, neo4j_password))

# Spark Session
spark = SparkSession.builder \
    .appName("SimRank") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

class Neo4jHandler:
    def __init__(self, graph, batch_size=1000):
        self.graph = graph
        self.batch_size = batch_size

    def clear_database(self):
        """Clear database in batches to avoid memory issues"""
        # Get the total number of nodes in the database
        total_nodes = self.graph.run("MATCH (n) RETURN count(n) as count").evaluate()
        deleted = 0
        with tqdm(total=total_nodes, desc="Clearing database") as pbar:
            while True:
                result = self.graph.run(
                    f"MATCH (n) WITH n LIMIT {self.batch_size} "
                    "DETACH DELETE n RETURN count(n)"
                ).evaluate()
                
                if result == 0:
                    break
                deleted += result
                pbar.update(result)
        print(f"Cleared {deleted} nodes.")

# Parse JSON data and construct citation graph in Neo4j
def load_data_to_neo4j(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    # Clear existing data in Neo4j using the new method
    handler = Neo4jHandler(graph)
    handler.clear_database()
    
    # Adding tqdm for outer loop to track the progress of processing each paper
    with tqdm(total=len(data), desc="Processing papers") as pbar:
        for entry in data:
            paper_id = entry["paper"]
            references = entry["reference"]
            
            # Create the citing paper node
            paper_node = Node("Paper", id=paper_id)
            graph.merge(paper_node, "Paper", "id")
            
            # Create cited paper nodes and citation relationships
            for ref in references:
                ref_node = Node("Paper", id=ref)
                graph.merge(ref_node, "Paper", "id")
                citation = Relationship(paper_node, "CITES", ref_node)
                graph.merge(citation)
            pbar.update(1)

# Export Neo4j graph to CSV for Spark
def export_graph_to_csv():
    # Query nodes and edges from Neo4j
    nodes = graph.run("MATCH (p:Paper) RETURN p.id AS id").to_data_frame()
    edges = graph.run("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.id AS src, b.id AS dst").to_data_frame()

    # Save nodes and edges to CSV
    nodes.to_csv("/tmp/nodes.csv", index=False)
    edges.to_csv("/tmp/edges.csv", index=False)

# Load the graph into Spark
def load_graph_in_spark():
    # Load nodes and edges
    nodes_df = spark.read.csv("/tmp/nodes.csv", header=True)
    edges_df = spark.read.csv("/tmp/edges.csv", header=True)

    # Create GraphFrame
    graph = GraphFrame(nodes_df, edges_df)
    return graph

# Compute SimRank
def simrank(graph, query_nodes, C=0.8, max_iterations=10, tolerance=1e-4):
    # Pre-compute all in-neighbors to avoid repeated queries
    in_neighbors_cache = {}
    vertices = graph.vertices.collect()
    edges = graph.edges.collect()
    
    # Build adjacency lists for faster neighbor lookup
    for v in vertices:
        in_neighbors_cache[v.id] = [
            e.src for e in edges if e.dst == v.id
        ]
    
    # Initialize similarity matrix using dictionary for sparse storage
    similarities = defaultdict(float)
    for node in query_nodes:
        similarities[(node["id"], node["id"])] = 1.0
    
    # Convert to numpy array for faster computation of query node pairs
    query_ids = [node["id"] for node in query_nodes]
    
    with tqdm(total=max_iterations, desc="SimRank Iterations") as pbar:
        for _ in range(max_iterations):
            new_similarities = defaultdict(float)
            max_change = 0.0
            
            # Process only necessary node pairs
            node_pairs = [
                (u.id, v.id) for u in vertices 
                for v in vertices if u.id <= v.id  # Process unique pairs only
            ]
            
            for u_id, v_id in tqdm(node_pairs, desc="Processing node pairs", leave=False):
                if u_id == v_id:
                    new_similarities[(u_id, v_id)] = 1.0
                    continue
                
                in_neighbors_u = in_neighbors_cache[u_id]
                in_neighbors_v = in_neighbors_cache[v_id]
                
                if in_neighbors_u and in_neighbors_v:
                    # Vectorized similarity computation
                    sim_sum = sum(
                        similarities[(n1, n2)] 
                        for n1 in in_neighbors_u 
                        for n2 in in_neighbors_v
                    )
                    scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
                    new_sim = scale * sim_sum
                    new_similarities[(u_id, v_id)] = new_sim
                    new_similarities[(v_id, u_id)] = new_sim  # Symmetry
                    
                    # Update max change
                    old_sim = similarities[(u_id, v_id)]
                    max_change = max(max_change, abs(new_sim - old_sim))
            
            similarities = new_similarities
            pbar.update(1)
            
            if max_change < tolerance:
                break
    
    # Compute results for query nodes
    results = {}
    for q_id in query_ids:
        # Use numpy for faster sorting
        sims = [(v.id, similarities[(q_id, v.id)]) 
                for v in vertices if v.id != q_id]
        sorted_sims = sorted(sims, key=lambda x: -x[1])
        results[q_id] = sorted_sims[:5]
    
    return results

# Run the whole process
# def main():
# Step 1: Load data into Neo4j
# load_data_to_neo4j('train.json')

# Step 2: Export the graph to CSV for Spark
export_graph_to_csv()

# Step 3: Load graph in Spark
spark_graph = load_graph_in_spark()

# Step 4: Compute SimRank with different C values
query_nodes = [{"id": "2982615777"}, {"id": "1556418098"}]
for C in [0.7, 0.8, 0.9]:
    simrank_results = simrank(spark_graph, query_nodes, C=C)
    print(f"SimRank results for C={C}:")
    for query, similar_nodes in simrank_results.items():
        print(f"Query Node {query}: {similar_nodes}")

# if __name__ == "__main__":
#     main()
