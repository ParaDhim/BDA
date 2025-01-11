"""111111"""
import json
from neo4j import GraphDatabase
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, array_contains
import networkx as nx
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from math import sqrt

# Step 1: Load JSON Data
with open('train.json', 'r') as file:
    data = [json.loads(line) for line in file]

# Connect to Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "paras2003"))

# Create citation graph in Neo4j
with driver.session() as session:
    for row in data:
        paper_id = row['paper']
        references = row['reference']
        
        # Create paper node
        session.run("CREATE (p:Paper {id: $paper_id})", paper_id=paper_id)
        
        # Create citation relationships
        for ref in references:
            session.run("MATCH (p1:Paper {id: $paper_id}) "
                       "MATCH (p2:Paper {id: $ref}) "
                       "CREATE (p1)-[:CITES]->(p2)", paper_id=paper_id, ref=ref)

# Load citation graph into NetworkX
G = nx.DiGraph()
with driver.session() as session:
    result = session.run("MATCH (p:Paper) RETURN p.id AS id, size([(p)-[:CITES]->() | 1]) AS in_degree, size([(()-[:CITES]->(p) | 1]) AS out_degree")
    for record in result:
        G.add_node(record["id"], in_degree=record["in_degree"], out_degree=record["out_degree"])
    result = session.run("MATCH (p1:Paper)-[r:CITES]->(p2:Paper) RETURN p1.id, p2.id")
    for record in result:
        G.add_edge(record["p1.id"], record["p2.id"])

# Run SimRank algorithm on the citation graph
spark = SparkSession.builder.appName("SimRank").getOrCreate()
sc = spark.sparkContext

@udf(DoubleType())
def simrank_score(src_id, dst_id):
    """
    Calculate the SimRank score between two nodes.
    
    Parameters:
    src_id (str): Source node ID
    dst_id (str): Destination node ID
    
    Returns:
    float: SimRank score between the source and destination nodes
    """
    if src_id == dst_id:
        return 1.0
    
    src_neighbors = [n for n in G.neighbors(src_id)]
    dst_neighbors = [n for n in G.neighbors(dst_id)]
    
    if not src_neighbors or not dst_neighbors:
        return 0.0
    
    score = 0.0
    for src_neighbor in src_neighbors:
        for dst_neighbor in dst_neighbors:
            score += simrank_score(src_neighbor, dst_neighbor)
    
    return importance_factor * score / (len(src_neighbors) * len(dst_neighbors))

def simrank(G, source, importance_factor, max_iterations, tolerance):
    """
    Run the SimRank algorithm on the citation graph.
    
    Parameters:
    G (networkx.DiGraph): Citation graph
    source (list): List of source node IDs to compute similarity for
    importance_factor (float): SimRank importance factor
    max_iterations (int): Maximum number of iterations
    tolerance (float): Convergence tolerance
    
    Returns:
    list: List of SimRank scores for the source nodes
    """
    # Convert NetworkX graph to Spark DataFrame
    nodes = [(n, d["in_degree"], d["out_degree"]) for n, d in G.nodes(data=True)]
    edges = [(u, v) for u, v in G.edges()]
    node_schema = StructType([
        StructField("id", StringType(), True),
        StructField("in_degree", IntegerType(), True),
        StructField("out_degree", IntegerType(), True)
    ])
    edge_schema = StructType([
        StructField("src", StringType(), True),
        StructField("dst", StringType(), True)
    ])
    node_df = spark.createDataFrame(nodes, schema=node_schema)
    edge_df = spark.createDataFrame(edges, schema=edge_schema)
    
    # Compute SimRank scores
    sim_scores = node_df.crossJoin(node_df.alias("other"))
    sim_scores = sim_scores.withColumn("similarity", simrank_score(col("id"), col("other.id")))
    sim_scores = sim_scores.filter(array_contains(source, col("id")) | array_contains(source, col("other.id")))
    return sim_scores.collect()

# Run SimRank with different importance factors
source_nodes = [2982615777, 1556418098]
for importance_factor in [0.7, 0.8, 0.9]:
    print(f"SimRank with importance factor: {importance_factor}")
    sim_scores = simrank(G, source_nodes, importance_factor, 1000, 0.0001)
    for row in sim_scores:
        print(f"Similarity between {row['id']} and {row['other.id']}: {row['similarity']}")