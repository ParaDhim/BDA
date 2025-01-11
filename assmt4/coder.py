from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.graphx import Graph, VertexRDD, Edge

# Initialize Spark session
conf = SparkConf().setAppName("GitHubCommunityDetection").setMaster("local[*]")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Load the dataset
edges_path = "musae_git_edges.csv"
vertices_path = "musae_git_target.csv"

# Load edges (source, destination)
edges_df = spark.read.csv(edges_path, header=True)
edges_rdd = edges_df.rdd.map(lambda row: (int(row['source']), int(row['target'])))

# Load vertices (id, attributes)
vertices_df = spark.read.csv(vertices_path, header=True)
vertices_rdd = vertices_df.rdd.map(lambda row: (int(row['id']), {"name": row['name'], "ml_target": int(row['ml_target'])}))

# Create Graph
edges = edges_rdd.map(lambda edge: Edge(edge[0], edge[1]))
vertices = vertices_rdd.map(lambda vertex: (vertex[0], vertex[1]))
graph = Graph(vertices, edges)

# AGM Model for Community Detection
# Placeholder for Affiliation Graph Model (AGM) implementation.
# As GraphX doesn't have direct AGM support, you may need to use an alternative method.
# For simplicity, here we'll use Label Propagation for community detection.

# Detect communities using Label Propagation
def label_propagation(graph):
    return graph.labelPropagation(maxSteps=5)

communities = label_propagation(graph)

# Evaluate Modularity
def calculate_modularity(graph, communities):
    # Modularity calculation logic here
    # Placeholder for modularity implementation
    pass

modularity = calculate_modularity(graph, communities)

# Output Results
print("Detected Communities:")
for community in communities.collect():
    print(community)

print(f"Modularity Score: {modularity}")

# Stop Spark session
sc.stop()
