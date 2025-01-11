import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from py2neo import Graph, Node, Relationship
from pyspark.sql import SparkSession
from graphframes import GraphFrame

# Connect to Neo4j
neo4j_url = "bolt://localhost:7689"
neo4j_username = "neo4j"
neo4j_password = "paras2003"
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

def load_data_to_neo4j(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]

    handler = Neo4jHandler(graph)
    handler.clear_database()
    
    with tqdm(total=len(data), desc="Processing papers") as pbar:
        for entry in data:
            paper_id = entry["paper"]
            references = entry["reference"]
            
            paper_node = Node("Paper", id=paper_id)
            graph.merge(paper_node, "Paper", "id")
            
            for ref in references:
                ref_node = Node("Paper", id=ref)
                graph.merge(ref_node, "Paper", "id")
                citation = Relationship(paper_node, "CITES", ref_node)
                graph.merge(citation)
            pbar.update(1)

def export_graph_to_csv():
    nodes = graph.run("MATCH (p:Paper) RETURN p.id AS id").to_data_frame()
    edges = graph.run("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.id AS src, b.id AS dst").to_data_frame()
    nodes.to_csv("nodes.csv", index=False)
    edges.to_csv("edges.csv", index=False)

def load_graph_in_spark():
    nodes_df = spark.read.csv("nodes.csv", header=True)
    edges_df = spark.read.csv("edges.csv", header=True)
    return GraphFrame(nodes_df, edges_df)
"""Approach-1"""
def simrank(graph, query_nodes, C=0.8, max_iterations=10, tolerance=1e-4):
    in_neighbors_cache = {}
    vertices = graph.vertices.collect()
    edges = graph.edges.collect()
    
    for v in vertices:
        in_neighbors_cache[v.id] = [e.src for e in edges if e.dst == v.id]
    
    similarities = defaultdict(float)
    for node in query_nodes:
        similarities[(node["id"], node["id"])] = 1.0
    
    query_ids = [node["id"] for node in query_nodes]
    
    with tqdm(total=max_iterations, desc="SimRank Iterations") as pbar:
        for _ in range(max_iterations):
            new_similarities = defaultdict(float)
            max_change = 0.0
            
            node_pairs = [
                (u.id, v.id) for u in vertices 
                for v in vertices if u.id <= v.id
            ]
            
            for u_id, v_id in tqdm(node_pairs, desc="Processing node pairs", leave=False):
                if u_id == v_id:
                    new_similarities[(u_id, v_id)] = 1.0
                    continue
                
                in_neighbors_u = in_neighbors_cache[u_id]
                in_neighbors_v = in_neighbors_cache[v_id]
                
                if in_neighbors_u and in_neighbors_v:
                    sim_sum = sum(
                        similarities[(n1, n2)] 
                        for n1 in in_neighbors_u 
                        for n2 in in_neighbors_v
                    )
                    scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
                    new_sim = scale * sim_sum
                    new_similarities[(u_id, v_id)] = new_sim
                    new_similarities[(v_id, u_id)] = new_sim
                    
                    old_sim = similarities[(u_id, v_id)]
                    max_change = max(max_change, abs(new_sim - old_sim))
            
            similarities = new_similarities
            pbar.update(1)
            
            if max_change < tolerance:
                break
    
    results = {}
    for q_id in query_ids:
        sims = [(v.id, similarities[(q_id, v.id)]) for v in vertices if v.id != q_id]
        sorted_sims = sorted(sims, key=lambda x: -x[1])
        results[q_id] = sorted_sims[:5]
    
    return results

# Step 1: Load data into Neo4j
load_data_to_neo4j('train.json')

# Step 2: Export the graph to CSV for Spark
export_graph_to_csv()

# Step 3: Load graph in Spark
spark_graph = load_graph_in_spark()

# Step 4: Compute SimRank
query_nodes = [{"id": "2982615777"}, {"id": "1556418098"}]
for C in [0.7, 0.8, 0.9]:
    simrank_results = simrank(spark_graph, query_nodes, C=C)
    print(f"SimRank results for C={C}:")
    for query, similar_nodes in simrank_results.items():
        print(f"Query Node {query}: {similar_nodes}")
        
        





"""Approach-2"""

# import json
# from collections import defaultdict
# from tqdm import tqdm
# import pandas as pd
# from py2neo import Graph, Node, Relationship
# from pyspark.sql import SparkSession
# from graphframes import GraphFrame

# # Connect to Neo4j
# neo4j_url = "bolt://localhost:7689"
# neo4j_username = "neo4j"
# neo4j_password = "paras2003"
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

# def load_data_to_neo4j(file_path):
#     with open(file_path, 'r') as file:
#         data = [json.loads(line) for line in file]

#     handler = Neo4jHandler(graph)
#     handler.clear_database()
    
#     with tqdm(total=len(data), desc="Processing papers") as pbar:
#         for entry in data:
#             paper_id = entry["paper"]
#             references = entry["reference"]
            
#             paper_node = Node("Paper", id=paper_id)
#             graph.merge(paper_node, "Paper", "id")
            
#             for ref in references:
#                 ref_node = Node("Paper", id=ref)
#                 graph.merge(ref_node, "Paper", "id")
#                 citation = Relationship(paper_node, "CITES", ref_node)
#                 graph.merge(citation)
#             pbar.update(1)

# def export_graph_to_csv():
#     nodes = graph.run("MATCH (p:Paper) RETURN p.id AS id").to_data_frame()
#     edges = graph.run("MATCH (a:Paper)-[:CITES]->(b:Paper) RETURN a.id AS src, b.id AS dst").to_data_frame()
#     nodes.to_csv("nodes.csv", index=False)
#     edges.to_csv("edges.csv", index=False)

# def load_graph_in_spark():
#     nodes_df = spark.read.csv("graph_nodes.csv", header=True)
#     edges_df = spark.read.csv("graph_edges.csv", header=True)
#     return GraphFrame(nodes_df, edges_df)

# import csv
# from neo4j import GraphDatabase

# def export_graph_from_neo4j(uri="neo4j://localhost:7687", 
#                             username="neo4j", 
#                             password="paras2003",
#                             nodes_output_file="graph_nodes.csv",
#                             edges_output_file="graph_edges.csv"):
#     """
#     Export Neo4j citation graph data into separate CSV files for nodes and edges without requiring APOC.
    
#     Parameters:
#     -----------
#     uri : str
#         Neo4j connection URI
#     username : str
#         Neo4j username
#     password : str
#         Neo4j password
#     nodes_output_file : str
#         Path to output CSV file for nodes
#     edges_output_file : str
#         Path to output CSV file for edges
#     """
#     driver = GraphDatabase.driver(uri, auth=(username, password))
    
#     try:
#         with driver.session() as session:
#             # Export nodes
#             nodes_query = """
#             MATCH (p:Paper)
#             RETURN p.id AS id
#             """
#             nodes_result = session.run(nodes_query)
            
#             with open(nodes_output_file, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 # Write header
#                 writer.writerow(['id'])
#                 # Write data rows
#                 for record in nodes_result:
#                     writer.writerow([record['id']])
#             print(f"Successfully exported nodes to {nodes_output_file}")
            
#             # Export edges
#             edges_query = """
#             MATCH (p1:Paper)-[:CITES]->(p2:Paper)
#             RETURN p1.id AS source, p2.id AS target
#             """
#             edges_result = session.run(edges_query)
            
#             with open(edges_output_file, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 # Write header
#                 writer.writerow(['src', 'dst'])
#                 # Write data rows
#                 for record in edges_result:
#                     writer.writerow([record['source'], record['target']])
#             print(f"Successfully exported edges to {edges_output_file}")
            
#     except Exception as e:
#         print(f"Error exporting graph: {str(e)}")
#     finally:
#         driver.close()

# # Call the function to export nodes and edges



# from collections import defaultdict
# from tqdm import tqdm

# def simrank(graph, query_nodes, C=0.8, max_iterations=10, tolerance=1e-4):
#     in_neighbors_cache = {}
#     vertices = graph.vertices.collect()
#     edges = graph.edges.collect()
#     for v in vertices:
#         in_neighbors_cache[v.id] = [e.src for e in edges if e.dst == v.id]
    
#     similarities = defaultdict(float)
#     for node in query_nodes:
#         similarities[(node["id"], node["id"])] = 1.0
#     query_ids = [node["id"] for node in query_nodes]
#     print("st")
#     with tqdm(total=max_iterations, desc="SimRank Iterations") as pbar:
#         for _ in range(max_iterations):
#             new_similarities = defaultdict(float)
#             max_change = 0.0
#             node_pairs = [
#                 (u.id, v.id) for u in vertices
#                 for v in vertices if u.id <= v.id
#             ]
#             for u_id, v_id in tqdm(node_pairs, desc="Processing node pairs", leave=False):
#                 if u_id == v_id:
#                     new_similarities[(u_id, v_id)] = 1.0
#                     continue
#                 in_neighbors_u = in_neighbors_cache[u_id]
#                 in_neighbors_v = in_neighbors_cache[v_id]
#                 if in_neighbors_u and in_neighbors_v:
#                     sim_sum = sum(
#                         similarities[(n1, n2)]
#                         for n1 in in_neighbors_u
#                         for n2 in in_neighbors_v
#                     )
#                     scale = C / (len(in_neighbors_u) * len(in_neighbors_v))
#                     new_sim = scale * sim_sum
#                     new_similarities[(u_id, v_id)] = new_sim
#                     new_similarities[(v_id, u_id)] = new_sim
#                     old_sim = similarities[(u_id, v_id)]
#                     max_change = max(max_change, abs(new_sim - old_sim))
#             similarities = new_similarities
#             pbar.update(1)
#             if max_change < tolerance:
#                 break
    
#     results = {}
#     for q_id in query_ids:
#         sims = [(v.id, similarities[(q_id, v.id)]) for v in vertices if v.id != q_id]
#         sorted_sims = sorted(sims, key=lambda x: -x[1])
#         results[q_id] = sorted_sims[:5]
#     return results

# # Step 1: Load data into Neo4j
# # load_data_to_neo4j('train.json')

# # Step 2: Export the graph to CSV for Spark
# # export_graph_to_csv()
# export_graph_from_neo4j()

# # Step 3: Load graph in Spark
# spark_graph = load_graph_in_spark()

# # Step 4: Compute SimRank
# query_nodes = [{"id": "2982615777"}, {"id": "1556418098"}]
# for C in tqdm([0.7, 0.8, 0.9],"Computing Simrank"):
#     simrank_results = simrank(spark_graph, query_nodes, C=C)
#     print(f"SimRank results for C={C}:")
#     for query, similar_nodes in simrank_results.items():
#         print(f"Query Node {query}: {similar_nodes}")


"""Approach-3"""

# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# import pandas as pd
# import os
# from datetime import datetime
# from tqdm.auto import tqdm

# def cache_in_neighbors(df):
#     """Cache in-neighbors for all nodes to avoid repeated queries."""
#     in_neighbors_df = df.groupBy('target').agg(
#         F.collect_list('source').alias('in_neighbors')
#     ).cache()
#     return {row['target']: row['in_neighbors'] for row in in_neighbors_df.collect()}

# def compute_simrank_similarity(a, b, in_neighbors_dict, C, max_iterations=10, tolerance=1e-4):
#     """Compute SimRank similarity between two nodes."""
#     if a == b:
#         return 1.0
        
#     in_neighbors_a = in_neighbors_dict.get(a, [])
#     in_neighbors_b = in_neighbors_dict.get(b, [])
    
#     if not in_neighbors_a or not in_neighbors_b:
#         return 0.0
    
#     # Initialize similarity matrix for in-neighbors
#     sim_matrix = {}
#     for na in in_neighbors_a:
#         for nb in in_neighbors_b:
#             if na == nb:
#                 sim_matrix[(na, nb)] = 1.0
#             else:
#                 sim_matrix[(na, nb)] = 0.0
    
#     # Iterate until convergence
#     for _ in range(max_iterations):
#         new_sim_matrix = {}
#         max_diff = 0.0
        
#         for na in in_neighbors_a:
#             for nb in in_neighbors_b:
#                 if na == nb:
#                     new_sim_matrix[(na, nb)] = 1.0
#                     continue
                    
#                 in_na = in_neighbors_dict.get(na, [])
#                 in_nb = in_neighbors_dict.get(nb, [])
                
#                 if not in_na or not in_nb:
#                     new_sim_matrix[(na, nb)] = 0.0
#                     continue
                
#                 sum_sim = 0.0
#                 for i in in_na:
#                     for j in in_nb:
#                         sum_sim += sim_matrix.get((i, j), 0.0)
                
#                 new_sim = (C / (len(in_na) * len(in_nb))) * sum_sim
#                 new_sim_matrix[(na, nb)] = new_sim
#                 max_diff = max(max_diff, abs(new_sim - sim_matrix.get((na, nb), 0.0)))
        
#         sim_matrix = new_sim_matrix
#         if max_diff < tolerance:
#             break
    
#     # Calculate final similarity
#     sum_sim = 0.0
#     for na in in_neighbors_a:
#         for nb in in_neighbors_b:
#             sum_sim += sim_matrix.get((na, nb), 0.0)
    
#     return (C / (len(in_neighbors_a) * len(in_neighbors_b))) * sum_sim

# def compute_simrank(df, query_nodes, C=0.8, max_iterations=10, tolerance=1e-4):
#     """Compute SimRank similarities for given query nodes."""
#     print("Caching in-neighbors...")
#     in_neighbors_dict = cache_in_neighbors(df)
    
#     # Get all unique nodes
#     all_nodes = set([row['node'] for row in df.select("source").union(
#         df.select("target")).distinct().withColumnRenamed("source", "node").collect()])
#     print(f"Total unique nodes: {len(all_nodes)}")
    
#     results = []
#     for query_node in tqdm(query_nodes, desc="Processing query nodes"):
#         node_results = []
#         for target_node in tqdm(all_nodes, desc=f"Computing similarities for node {query_node}", leave=False):
#             sim = compute_simrank_similarity(
#                 query_node, 
#                 target_node, 
#                 in_neighbors_dict, 
#                 C,
#                 max_iterations,
#                 tolerance
#             )
#             node_results.append((query_node, target_node, sim))
#         results.extend(node_results)
    
#     return pd.DataFrame(results, columns=['query_node', 'target_node', 'similarity'])

# def run_simrank_analysis(edges_df, query_nodes, decay_factors, output_dir="simrank_results"):
#     """Run SimRank analysis for multiple decay factors and save results."""
#     os.makedirs(output_dir, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     all_results = []
    
#     for C in tqdm(decay_factors, desc="Processing decay factors"):
#         print(f"\nComputing SimRank with decay factor C = {C}")
        
#         results_df = compute_simrank(edges_df, query_nodes, C=C)
#         results_df['decay_factor'] = C
#         all_results.append(results_df)
        
#         # Save intermediate results
#         results_df.to_csv(
#             f"{output_dir}/simrank_results_C{C}_{timestamp}.csv",
#             index=False
#         )
    
#     final_results = pd.concat(all_results, ignore_index=True)
    
#     # Save complete results
#     final_results.to_csv(
#         f"{output_dir}/simrank_all_results_{timestamp}.csv",
#         index=False
#     )
    
#     # Generate and save top 10 results
#     top_results = []
#     for C in decay_factors:
#         for query in query_nodes:
#             top_10 = final_results[
#                 (final_results['decay_factor'] == C) & 
#                 (final_results['query_node'] == query)
#             ].nlargest(10, 'similarity')
#             top_10['rank'] = range(1, 11)
#             top_results.append(top_10)
    
#     top_results_df = pd.concat(top_results, ignore_index=True)
#     top_results_df.to_csv(
#         f"{output_dir}/simrank_top_results_{timestamp}.csv",
#         index=False
#     )
    
#     # Print summary
#     print("\nTop 5 most similar nodes for each query node and decay factor:")
#     for C in decay_factors:
#         print(f"\nDecay factor C = {C}")
#         for query in query_nodes:
#             print(f"\nQuery node: {query}")
#             top_5 = top_results_df[
#                 (top_results_df['decay_factor'] == C) & 
#                 (top_results_df['query_node'] == query)
#             ].head()
#             print(top_5[['target_node', 'similarity', 'rank']].to_string())
    
#     return final_results, top_results_df

# # Run the analysis
# query_nodes = [2982615777, 1556418098]
# decay_factors = [0.7, 0.8, 0.9]

# # Correct column names in the edges DataFrame
# edges_df = spark.read.csv("graph_edges.csv", header=True)
# edges_df = edges_df.withColumnRenamed("source", "src").withColumnRenamed("target", "dst")

# final_results, top_results = run_simrank_analysis(
#     edges_df,
#     query_nodes=query_nodes,
#     decay_factors=decay_factors,
#     output_dir="simrank_results"
# )


"""Approach-4"""
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# from neo4j import GraphDatabase
# import pandas as pd
# import os
# from datetime import datetime
# from tqdm.auto import tqdm

# class CitationGraphAnalyzer:
#     def __init__(self, neo4j_uri="bolt://localhost:7687", 
#                  neo4j_user="neo4j", neo4j_password="paras2003"):
#         """Initialize with Neo4j and Spark connections"""
#         self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
#         self.spark = SparkSession.builder \
#             .appName("Citation Graph Analysis") \
#             .config("spark.driver.memory", "4g") \
#             .config("spark.executor.memory", "4g") \
#             .config("spark.task.maxFailures", "4") \
#             .getOrCreate()

#     def get_graph_data(self):
#         """Extract graph data from Neo4j for Spark processing"""
#         with self.driver.session() as session:
#             result = session.run("""
#                 MATCH (p1:Paper)-[:CITES]->(p2:Paper)
#                 RETURN p1.id as source, p2.id as target
#             """)
#             edges = [(record["source"], record["target"]) for record in result]
#             return edges

#     def compute_simrank(self, edges_df, query_nodes, decay_factors, output_dir="simrank_results", max_iterations=10, tolerance=1e-4):
#         """Run SimRank analysis on citation graph"""
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Ensure column names are standardized
#         edges_df = edges_df.withColumnRenamed("src", "source").withColumnRenamed("dst", "target")
        
#         # Cache in-neighbors
#         in_neighbors_dict = self._cache_in_neighbors(edges_df)
        
#         all_results = []
        
#         for C in decay_factors:
#             print(f"\nComputing SimRank with decay factor C = {C}")
            
#             results = []
#             all_nodes = set(row['source'] for row in edges_df.select("source").distinct().collect())
#             all_nodes.update(row['target'] for row in edges_df.select("target").distinct().collect())
            
#             for query_node in tqdm(query_nodes, desc="Processing query nodes"):
#                 node_results = []
#                 for target_node in tqdm(all_nodes, desc=f"Computing similarities for node {query_node}", leave=False):
#                     sim = self._compute_simrank_similarity(
#                         query_node,
#                         target_node,
#                         in_neighbors_dict,
#                         C,
#                         max_iterations,
#                         tolerance
#                     )
#                     node_results.append((query_node, target_node, sim))
#                 results.extend(node_results)
            
#             results_df = pd.DataFrame(results, columns=['query_node', 'target_node', 'similarity'])
#             results_df['decay_factor'] = C
#             all_results.append(results_df)
            
#             # Save intermediate results
#             output_path = f"{output_dir}/simrank_results_C{C}_{timestamp}.csv"
#             results_df.to_csv(output_path, index=False)
        
#         return self._save_and_summarize_results(all_results, query_nodes, decay_factors, timestamp, output_dir)

#     def _cache_in_neighbors(self, edges_df):
#         """Cache in-neighbors for all nodes"""
#         in_neighbors = edges_df.groupBy('target').agg(F.collect_list('source').alias('in_neighbors'))
#         return {row['target']: row['in_neighbors'] for row in in_neighbors.collect()}

#     def _compute_simrank_similarity(self, a, b, in_neighbors_dict, C, max_iterations, tolerance):
#         """Compute SimRank similarity between two nodes."""
#         if a == b:
#             return 1.0
        
#         in_neighbors_a = in_neighbors_dict.get(a, [])
#         in_neighbors_b = in_neighbors_dict.get(b, [])
        
#         if not in_neighbors_a or not in_neighbors_b:
#             return 0.0
        
#         # Initialize similarity matrix for in-neighbors
#         sim_matrix = {}
#         for na in in_neighbors_a:
#             for nb in in_neighbors_b:
#                 sim_matrix[(na, nb)] = 0.0
        
#         # Iterate until convergence
#         for _ in range(max_iterations):
#             new_sim_matrix = {}
#             max_diff = 0.0
            
#             for na in in_neighbors_a:
#                 for nb in in_neighbors_b:
#                     in_na = in_neighbors_dict.get(na, [])
#                     in_nb = in_neighbors_dict.get(nb, [])
                    
#                     if not in_na or not in_nb:
#                         continue
                    
#                     sum_sim = sum(sim_matrix.get((i, j), 0.0) for i in in_na for j in in_nb)
#                     new_sim = (C / (len(in_na) * len(in_nb))) * sum_sim
#                     new_sim_matrix[(na, nb)] = new_sim
#                     max_diff = max(max_diff, abs(new_sim - sim_matrix.get((na, nb), 0.0)))
            
#             sim_matrix = new_sim_matrix
#             if max_diff < tolerance:
#                 break
        
#         sum_sim = sum(sim_matrix.get((na, nb), 0.0) for na in in_neighbors_a for nb in in_neighbors_b)
#         normalization_factor = (len(in_neighbors_a) * len(in_neighbors_b)) or 1  # Prevent division by 0
#         return (C / normalization_factor) * sum_sim

#     def _save_and_summarize_results(self, all_results, query_nodes, decay_factors, timestamp, output_dir):
#         """Save and summarize final results"""
#         final_results = pd.concat(all_results, ignore_index=True)
#         final_results_path = f"{output_dir}/final_simrank_results_{timestamp}.csv"
#         final_results.to_csv(final_results_path, index=False)
        
#         top_results = final_results.groupby(['query_node', 'decay_factor']).apply(
#             lambda group: group.nlargest(10, 'similarity')).reset_index(drop=True)
#         top_results_path = f"{output_dir}/top_simrank_results_{timestamp}.csv"
#         top_results.to_csv(top_results_path, index=False)
        
#         return final_results, top_results

#     def close(self):
#         """Close Neo4j and Spark connections"""
#         self.driver.close()
#         self.spark.stop()

#     def bfs_traversal(self, start_node, depth=2):
#         """Perform BFS to get all nodes within a given depth"""
#         with self.driver.session() as session:
#             result = session.run("""
#                 MATCH (start:Paper)-[:CITES*1..{depth}]->(p:Paper)
#                 WHERE start.id = $start_node
#                 RETURN p.id as node
#             """, start_node=start_node, depth=depth)
#             return [record["node"] for record in result]

# # Example usage
# if __name__ == "__main__":
#     edges_csv_path = "graph_edges.csv"
#     edges_df = pd.read_csv(edges_csv_path)
    
#     spark = SparkSession.builder.appName("SimRank Analysis").getOrCreate()
#     edges_sdf = spark.createDataFrame(edges_df)

#     analyzer = CitationGraphAnalyzer()

#     try:
#         query_nodes = [2982615777, 1556418098]
#         decay_factors = [0.7, 0.8, 0.9]
#         print("Running SimRank analysis...")
#         final_results, top_results = analyzer.compute_simrank(
#             edges_sdf,
#             query_nodes=query_nodes,
#             decay_factors=decay_factors
#         )
#     finally:
#         analyzer.close()
