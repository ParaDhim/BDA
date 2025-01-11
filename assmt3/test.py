from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col
from pyspark.sql.types import *
import itertools

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CitationGraphSimRank") \
    .getOrCreate()

# Define schema for the JSON data
schema = StructType([
    StructField("reference", ArrayType(StringType()), True),
    StructField("author", ArrayType(StringType()), True),
    StructField("text", StringType(), True),
    StructField("venue", StringType(), True),
    StructField("paper", StringType(), True),
    StructField("label", ArrayType(StringType()), True)
])

# Read JSON file
df = spark.read.schema(schema).json("train.json")

# Create vertices DataFrame
vertices = df.select("paper").distinct()

# Create edges DataFrame by exploding the reference array
edges = df.select(
    col("paper").alias("src"),
    explode(col("reference")).alias("dst")
).distinct()

# Custom SimRank implementation using Spark
def simrank_spark(vertices_df, edges_df, c, max_iterations=10, tolerance=0.0001):
    # Convert DataFrames to RDDs for easier processing
    vertices_rdd = vertices_df.rdd.map(lambda x: x[0])
    edges_rdd = edges_df.rdd.map(lambda x: (x[0], x[1]))
    
    # Create adjacency lists (incoming edges)
    in_neighbors = edges_rdd.map(lambda x: (x[1], x[0])) \
                          .groupByKey() \
                          .mapValues(list)
    
    # Initialize similarity scores
    nodes = vertices_rdd.collect()
    sim_scores = spark.sparkContext.parallelize([
        (u, v, 1.0 if u == v else 0.0)
        for u, v in itertools.product(nodes, nodes)
    ])
    
    # Iterative SimRank computation
    for iteration in range(max_iterations):
        new_scores = sim_scores.map(lambda x: (
            x[0], x[1],
            compute_simrank_score(x[0], x[1], in_neighbors, sim_scores, c)
        ))
        
        # Check convergence
        diff = new_scores.join(sim_scores) \
                        .map(lambda x: abs(x[1][0] - x[1][1])) \
                        .max()
        
        if diff < tolerance:
            break
            
        sim_scores = new_scores
    
    return sim_scores

def compute_simrank_score(u, v, in_neighbors, current_scores, c):
    if u == v:
        return 1.0
    
    u_neighbors = in_neighbors.lookup(u)
    v_neighbors = in_neighbors.lookup(v)
    
    if not u_neighbors or not v_neighbors:
        return 0.0
    
    sum_score = 0.0
    for nu in u_neighbors:
        for nv in v_neighbors:
            score = current_scores.filter(lambda x: x[0] == nu and x[1] == nv).first()[2]
            sum_score += score
            
    return (c / (len(u_neighbors) * len(v_neighbors))) * sum_score

# Run SimRank for different C values
query_nodes = ["2982615777", "1556418098"]
c_values = [0.7, 0.8, 0.9]

for c in c_values:
    print(f"\nSimRank results for C = {c}")
    sim_scores = simrank_spark(vertices, edges, c)
    
    # Filter results for query nodes
    results = sim_scores.filter(
        lambda x: x[0] in query_nodes and x[1] in query_nodes
    ).collect()
    
    for result in results:
        print(f"Similarity between {result[0]} and {result[1]}: {result[2]:.4f}")

# Stop Spark session
spark.stop()