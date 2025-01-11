from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, collect_list
from pyspark.sql.types import *
import numpy as np
from collections import defaultdict

def create_spark_session():
    return SparkSession.builder \
        .appName("CitationGraphSimRank") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_and_process_data(spark, file_path):
    # Define schema
    schema = StructType([
        StructField("reference", ArrayType(StringType()), True),
        StructField("paper", StringType(), True)
    ])

    # Read only needed columns
    df = spark.read.schema(schema).json(file_path)
    
    # Create vertices DataFrame
    vertices = df.select("paper").distinct()
    
    # Create edges DataFrame
    edges = df.select(
        col("paper").alias("src"),
        explode(col("reference")).alias("dst")
    ).distinct()
    
    return vertices, edges

def create_adjacency_lists(edges_df):
    # Collect edges to driver for more efficient processing
    edge_pairs = edges_df.collect()
    
    # Create incoming neighbors dictionary
    in_neighbors = defaultdict(list)
    for edge in edge_pairs:
        in_neighbors[edge.dst].append(edge.src)
    
    return dict(in_neighbors)

def simrank_optimized(vertices_df, in_neighbors, c, max_iterations=10, tolerance=0.001, query_nodes=None):
    # Get list of all nodes
    nodes = [row.paper for row in vertices_df.collect()]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Initialize similarity matrix
    n = len(nodes)
    sim_matrix = np.zeros((n, n))
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Create new matrix for iteration
    new_sim_matrix = np.zeros((n, n))
    
    # Iterate until convergence
    for _ in range(max_iterations):
        new_sim_matrix.fill(0)
        np.fill_diagonal(new_sim_matrix, 1.0)
        
        # Only compute similarities for query nodes if specified
        target_nodes = [query_nodes] if query_nodes else nodes
        
        for u in target_nodes:
            u_idx = node_to_idx[u]
            u_in = in_neighbors.get(u, [])
            if not u_in:
                continue
                
            for v in nodes:
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
        
        if diff < tolerance:
            break
    
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
    
    # Run SimRank for different C values
    c_values = [0.7, 0.8, 0.9]
    
    for c in c_values:
        print(f"\nSimRank results for C = {c}")
        
        # Run SimRank
        sim_matrix, nodes, node_to_idx = simrank_optimized(
            vertices,
            in_neighbors,
            c,
            query_nodes=query_nodes
        )
        
        # Print results for query nodes
        for i, node1 in enumerate(query_nodes):
            for node2 in query_nodes[i:]:
                idx1 = node_to_idx[node1]
                idx2 = node_to_idx[node2]
                similarity = sim_matrix[idx1][idx2]
                print(f"Similarity between {node1} and {node2}: {similarity:.4f}")
    
    spark.stop()

if __name__ == "__main__":
    main()