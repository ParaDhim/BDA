from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from pyspark.sql.types import *
import numpy as np
from collections import defaultdict
import gc
import psutil
import os
from datetime import datetime

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"[{datetime.now()}] Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def create_spark_session():
    print(f"[{datetime.now()}] Initializing Spark Session...")
    spark = SparkSession.builder \
        .appName("CitationGraphSimRank") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "10") \
        .config("spark.default.parallelism", "4") \
        .config("spark.memory.fraction", "0.6") \
        .config("spark.memory.storageFraction", "0.5") \
        .getOrCreate()
    print(f"[{datetime.now()}] Spark Session initialized successfully")
    return spark

def validate_query_nodes(query_nodes, vertices):
    """Validate query nodes and return only those present in the dataset"""
    valid_nodes = []
    for node in query_nodes:
        if node in vertices:
            valid_nodes.append(node)
        else:
            print(f"Warning: Query node {node} not found in dataset")
    return valid_nodes

def process_data_in_chunks(spark, file_path, chunk_size=1000):
    print(f"[{datetime.now()}] Processing data in chunks of {chunk_size}")
    print_memory_usage()
    
    schema = StructType([
        StructField("reference", ArrayType(StringType()), True),
        StructField("paper", StringType(), True)
    ])

    # Read data in chunks
    df = spark.read.schema(schema).json(file_path)
    
    # First, get all unique papers to ensure we don't miss any query nodes
    print("Collecting all unique papers...")
    papers_df = df.select("paper").distinct()
    all_papers = set(row.paper for row in papers_df.collect())
    
    # Also collect all references
    print("Collecting all references...")
    refs_df = df.select(explode(col("reference")).alias("ref")).distinct()
    all_refs = set(row.ref for row in refs_df.collect())
    
    # Combine all unique vertices
    vertices = all_papers.union(all_refs)
    print(f"Total unique vertices: {len(vertices)}")
    
    # Process edges in chunks
    edges = defaultdict(set)
    total_rows = df.count()
    
    for offset in range(0, total_rows, chunk_size):
        chunk = df.limit(chunk_size).offset(offset)
        
        # Process chunk
        chunk_data = chunk.collect()
        for row in chunk_data:
            if row.reference:
                for ref in row.reference:
                    edges[row.paper].add(ref)
        
        print(f"[{datetime.now()}] Processed {min(offset + chunk_size, total_rows)}/{total_rows} rows")
        print_memory_usage()
        
        # Force garbage collection
        gc.collect()
    
    return list(vertices), dict(edges)

def create_sparse_simrank_matrix(vertices, edges, c, max_iterations=10, tolerance=0.001, query_nodes=None):
    print(f"[{datetime.now()}] Starting sparse SimRank computation")
    print_memory_usage()
    
    # Validate query nodes first
    if query_nodes:
        valid_query_nodes = validate_query_nodes(query_nodes, set(vertices))
        if not valid_query_nodes:
            raise ValueError("None of the query nodes were found in the dataset")
        query_nodes = valid_query_nodes
    
    # Create node mapping
    node_to_idx = {node: idx for idx, node in enumerate(vertices)}
    n = len(vertices)
    
    # Create sparse similarity matrix using dictionary
    sim_scores = {}
    
    # Initialize diagonal elements
    for i in range(n):
        sim_scores[(i, i)] = 1.0
    
    # Initialize query node pairs
    if query_nodes:
        query_indices = [node_to_idx[node] for node in query_nodes]
        for i in query_indices:
            for j in query_indices:
                if i != j:
                    sim_scores[(i, j)] = 0.0
    
    # Create reverse edges (incoming neighbors)
    incoming = defaultdict(list)
    for src, dsts in edges.items():
        src_idx = node_to_idx[src]
        for dst in dsts:
            if dst in node_to_idx:  # Check if destination node exists
                dst_idx = node_to_idx[dst]
                incoming[dst_idx].append(src_idx)
    
    # SimRank iterations
    for iteration in range(max_iterations):
        print(f"\n[{datetime.now()}] Iteration {iteration + 1}")
        print_memory_usage()
        
        new_scores = {}
        max_diff = 0.0
        
        # Only process query nodes if specified
        target_nodes = query_indices if query_nodes else range(n)
        
        for i in target_nodes:
            i_in = incoming[i]
            if not i_in:
                continue
                
            for j in range(n):
                if i == j:
                    new_scores[(i, j)] = 1.0
                    continue
                
                j_in = incoming[j]
                if not j_in:
                    new_scores[(i, j)] = 0.0
                    continue
                
                # Compute similarity
                sum_score = 0.0
                for ni in i_in:
                    for nj in j_in:
                        if (ni, nj) in sim_scores:
                            sum_score += sim_scores[(ni, nj)]
                
                new_score = (c / (len(i_in) * len(j_in))) * sum_score
                new_scores[(i, j)] = new_score
                
                # Update maximum difference
                old_score = sim_scores.get((i, j), 0.0)
                max_diff = max(max_diff, abs(new_score - old_score))
            
            # Progress update
            if len(target_nodes) > 0:
                progress = ((target_nodes.index(i) + 1) / len(target_nodes)) * 100
                print(f"Progress: {progress:.1f}%")
                print_memory_usage()
        
        # Update similarity scores
        sim_scores = new_scores
        
        print(f"Maximum change in similarity: {max_diff:.6f}")
        
        # Check convergence
        if max_diff < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Force garbage collection
        gc.collect()
    
    return sim_scores, node_to_idx, query_nodes

def main():
    # Initialize Spark
    spark = create_spark_session()
    print_memory_usage()
    
    # Process data in chunks
    vertices, edges = process_data_in_chunks(spark, "train.json", chunk_size=500)
    print(f"Total vertices: {len(vertices)}")
    print(f"Total edges: {sum(len(dsts) for dsts in edges.values())}")
    print_memory_usage()
    
    # Query nodes
    original_query_nodes = ["2982615777", "1556418098"]
    
    try:
        # Run SimRank for different C values
        c_values = [0.7, 0.8, 0.9]
        
        for c in c_values:
            print(f"\n{'='*80}")
            print(f"SimRank computation for C = {c}")
            print('='*80)
            
            # Run sparse SimRank
            sim_scores, node_to_idx, valid_query_nodes = create_sparse_simrank_matrix(
                vertices,
                edges,
                c,
                query_nodes=original_query_nodes
            )
            
            # Print results for valid query nodes
            print("\nResults:")
            for i, node1 in enumerate(valid_query_nodes):
                idx1 = node_to_idx[node1]
                for node2 in valid_query_nodes[i:]:
                    idx2 = node_to_idx[node2]
                    similarity = sim_scores.get((idx1, idx2), 0.0)
                    print(f"Similarity between {node1} and {node2}: {similarity:.4f}")
            
            print_memory_usage()
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check the query nodes and ensure they exist in the dataset")
    
    spark.stop()

if __name__ == "__main__":
    main()