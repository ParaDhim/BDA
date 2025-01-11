import json

# Open and read each line as a separate JSON object
with open('train.json', 'r') as file:
    data = [json.loads(line) for line in file]




from neo4j import GraphDatabase

# Neo4j connection details
uri = "bolt://localhost:7687"  # update if different
username = "neo4j"  # replace with your username
password = "paras2003"  # replace with your password

# Connect to Neo4j
driver = GraphDatabase.driver(uri, auth=(username, password))

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





from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("SimRank").getOrCreate()

# Load citation graph CSV into a Spark DataFrame
df = spark.read.csv("citation_graph.csv", header=True, inferSchema=True)
df.show()





from itertools import product

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

# Run the SimRank algorithm for different values of C
results = {}
for C_value in [0.7, 0.8, 0.9]:
    results[C_value] = simrank(df, query_nodes=[2982615777, 1556418098], C=C_value)




for C_value, sim_scores in results.items():
    print(f"Results for C = {C_value}:")
    for (u, v), score in sim_scores.items():
        print(f"Similarity between {u} and {v}: {score}")
