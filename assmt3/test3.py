from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, collect_list
from pyspark.sql.types import *
import numpy as np
from collections import defaultdict
import time
from datetime import datetime

def create_spark_session():
    print(f"[{datetime.now()}] Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("CitationGraphSimRank") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    print(f"[{datetime.now()}] Spark Session initialized successfully")
    return spark

def load_and_process_data(spark, file_path):
    print(f"\n[{datetime.now()}] Loading data from {file_path}")
    
    # Define schema
    schema = StructType([
        StructField("reference", ArrayType(StringType()), True),
        StructField("paper", StringType(), True)
    ])

    # Read only needed columns
    start_time = time.time()
    df = spark.read.schema(schema).json(file_path)
    print(f"[{datetime.now()}] Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Create vertices DataFrame
    print(f"[{datetime.now()}] Creating vertices DataFrame...")
    vertices = df.select("paper").distinct()
    vertex_count = vertices.count()
    print(f"[{datetime.now()}] Found {vertex_count} unique papers")
    
    # Create edges DataFrame
    print(f"[{datetime.now()}] Creating edges DataFrame...")
    edges = df.select(
        col("paper").alias("src"),
        explode(col("reference")).alias("dst")
    ).distinct()
    edge_count = edges.count()
    print(f"[{datetime.now()}] Found {edge_count} unique citation relationships")
    
    return vertices, edges

def create_adjacency_lists(edges_df):
    print(f"\n[{datetime.now()}] Creating adjacency lists...")
    start_time = time.time()
    
    # Collect edges to driver for more efficient processing
    edge_pairs = edges_df.collect()
    
    # Create incoming neighbors dictionary
    in_neighbors = defaultdict(list)
    for i, edge in enumerate(edge_pairs):
        in_neighbors[edge.dst].append(edge.src)
        if i % 10000 == 0:  # Progress update every 10000 edges
            print(f"[{datetime.now()}] Processed {i}/{len(edge_pairs)} edges...")
    
    print(f"[{datetime.now()}] Adjacency lists created in {time.time() - start_time:.2f} seconds")
    print(f"[{datetime.now()}] Number of nodes with incoming edges: {len(in_neighbors)}")
    
    return dict(in_neighbors)

def print_similarity_stats(sim_matrix):
    """Print statistics about similarity values"""
    non_diagonal = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]
    print(f"\nSimilarity Statistics:")
    print(f"Average similarity: {np.mean(non_diagonal):.4f}")
    print(f"Max similarity: {np.max(non_diagonal):.4f}")
    print(f"Min similarity: {np.min(non_diagonal):.4f}")
    print(f"Median similarity: {np.median(non_diagonal):.4f}")

def simrank_optimized(vertices_df, in_neighbors, c, max_iterations=10, tolerance=0.001, query_nodes=None):
    print(f"\n[{datetime.now()}] Starting SimRank computation...")
    start_time = time.time()
    
    # Get list of all nodes
    nodes = [row.paper for row in vertices_df.collect()]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    print(f"[{datetime.now()}] Processing {len(nodes)} nodes")
    
    # Initialize similarity matrix
    n = len(nodes)
    sim_matrix = np.zeros((n, n))
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Create new matrix for iteration
    new_sim_matrix = np.zeros((n, n))
    
    # Iterate until convergence
    for iteration in range(max_iterations):
        iter_start_time = time.time()
        new_sim_matrix.fill(0)
        np.fill_diagonal(new_sim_matrix, 1.0)
        
        # Only compute similarities for query nodes if specified
        target_nodes = [query_nodes] if query_nodes else nodes
        processed_pairs = 0
        total_pairs = len(target_nodes) * len(nodes)
        
        for u in target_nodes:
            u_idx = node_to_idx[u]
            u_in = in_neighbors.get(u, [])
            if not u_in:
                continue
                
            for v in nodes:
                processed_pairs += 1
                if processed_pairs % 1000 == 0:  # Progress update every 1000 pairs
                    progress = (processed_pairs / total_pairs) * 100
                    print(f"[{datetime.now()}] Iteration {iteration + 1}/{max_iterations}: "
                          f"Processed {processed_pairs}/{total_pairs} pairs ({progress:.1f}%)")
                
                if u == v:
                    continue
                    
                v_idx = node_to_idx[v]
                v_in = in_neighbors.get(v, [])
                if not v_in:
                    continue
                
                sum_score = 0.0
                for nu in u_in:
                    nu_idx = node_to_idx[nu]
                    for nv in v_in:
                        nv_idx = node_to_idx[nv]
                        sum_score += sim_matrix[nu_idx][nv_idx]
                
                new_sim_matrix[u_idx][v_idx] = (c / (len(u_in) * len(v_in))) * sum_score
                new_sim_matrix[v_idx][u_idx] = new_sim_matrix[u_idx][v_idx]
        
        # Check convergence
        diff = np.abs(new_sim_matrix - sim_matrix).max()
        sim_matrix = new_sim_matrix.copy()
        
        iter_time = time.time() - iter_start_time
        print(f"\n[{datetime.now()}] Iteration {iteration + 1} completed in {iter_time:.2f} seconds")
        print(f"Maximum change in similarity: {diff:.6f}")
        print_similarity_stats(sim_matrix)
        
        if diff < tolerance:
            print(f"\n[{datetime.now()}] Convergence reached after {iteration + 1} iterations!")
            break
    
    total_time = time.time() - start_time
    print(f"\n[{datetime.now()}] SimRank computation completed in {total_time:.2f} seconds")
    
    return sim_matrix, nodes, node_to_idx

def main():
    # Initialize Spark
    spark = create_spark_session()
    
    # Load and process data
    vertices, edges = load_and_process_data(spark, "train.json")
    
    # Create adjacency lists
    in_neighbors = create_adjacency_lists(edges)
    
    # Query nodes
    query_nodes = ["2982615777", "1556418098"]
    print(f"\n[{datetime.now()}] Query nodes: {query_nodes}")
    
    # Run SimRank for different C values
    c_values = [0.7, 0.8, 0.9]
    
    for c in c_values:
        print(f"\n{'='*80}")
        print(f"[{datetime.now()}] Starting SimRank computation for C = {c}")
        print('='*80)
        
        # Run SimRank
        sim_matrix, nodes, node_to_idx = simrank_optimized(
            vertices,
            in_neighbors,
            c,
            query_nodes=query_nodes
        )
        
        # Print results for query nodes
        print(f"\n[{datetime.now()}] Final Results for C = {c}:")
        print('-'*40)
        for i, node1 in enumerate(query_nodes):
            for node2 in query_nodes[i:]:
                idx1 = node_to_idx[node1]
                idx2 = node_to_idx[node2]
                similarity = sim_matrix[idx1][idx2]
                print(f"Similarity between {node1} and {node2}: {similarity:.4f}")
    
    print(f"\n[{datetime.now()}] Program completed successfully")
    spark.stop()

if __name__ == "__main__":
    main()