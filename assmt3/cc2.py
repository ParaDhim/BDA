from tqdm import tqdm
from neo4j import GraphDatabase
from pyspark.sql import SparkSession
import networkx as nx
import json
from itertools import product
import numpy as np
from typing import List, Dict, Set
import logging
from tqdm.auto import trange

class BatchedCitationGraph:
    def __init__(self, uri: str, user: str, password: str, batch_size: int = 1000):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.batch_size = batch_size
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Clear database in batches to avoid memory issues"""
        total_nodes = self.driver.session().run("MATCH (n) RETURN count(n) as count").single()["count"]
        deleted = 0
        with tqdm(total=total_nodes, desc="Clearing database") as pbar:
            while True:
                result = self.driver.session().run(
                    f"MATCH (n) WITH n LIMIT {self.batch_size} "
                    "DETACH DELETE n RETURN count(n)"
                ).single()
                if result[0] == 0:
                    break
                deleted += result[0]
                pbar.update(result[0])

    def create_paper_nodes_batch(self, paper_ids: Set[str]):
        """Create paper nodes in batches"""
        with self.driver.session() as session:
            for i in tqdm(range(0, len(paper_ids), self.batch_size), 
                         desc="Creating paper nodes", 
                         total=(len(paper_ids) + self.batch_size - 1) // self.batch_size):
                batch = list(paper_ids)[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS paper_id
                    MERGE (p:Paper {id: paper_id})
                """, batch=batch)

    def create_citation_edges_batch(self, citations: List[tuple]):
        """Create citation edges in batches"""
        with self.driver.session() as session:
            for i in tqdm(range(0, len(citations), self.batch_size), 
                         desc="Creating citation edges",
                         total=(len(citations) + self.batch_size - 1) // self.batch_size):
                batch = citations[i:i + self.batch_size]
                session.run("""
                    UNWIND $batch AS citation
                    MATCH (p1:Paper {id: citation[0]})
                    MATCH (p2:Paper {id: citation[1]})
                    MERGE (p1)-[:CITES]->(p2)
                """, batch=batch)

def create_neo4j_graph(data_file: str, batch_size: int = 1000):
    """Create Neo4j graph with batched operations"""
    graph = BatchedCitationGraph(
        "bolt://localhost:7687",
        "neo4j",
        "paras2003",
        batch_size=batch_size
    )

    # Clear existing data
    graph.logger.info("Starting database cleanup...")
    graph.clear_database()

    # First pass: Collect all unique paper IDs
    paper_ids = set()
    citations = []
    
    # Count total lines in file first
    total_lines = sum(1 for _ in open(data_file, 'r'))
    
    graph.logger.info("Processing data file...")
    with open(data_file, 'r') as f:
        with tqdm(f, total=total_lines, desc="Reading papers") as pbar:
            for line in pbar:
                record = json.loads(line)
                citing_paper = record["paper"]
                references = record["reference"]
                
                paper_ids.add(citing_paper)
                paper_ids.update(references)
                citations.extend((citing_paper, cited_paper) 
                               for cited_paper in references)
                pbar.set_postfix({'papers': len(paper_ids), 'citations': len(citations)})

    # Create nodes in batches
    graph.logger.info(f"Creating {len(paper_ids)} paper nodes...")
    graph.create_paper_nodes_batch(paper_ids)

    # Create edges in batches
    graph.logger.info(f"Creating {len(citations)} citation edges...")
    graph.create_citation_edges_batch(citations)

    graph.logger.info("Graph creation completed")
    graph.close()

def convert_to_networkx(neo4j_driver, batch_size: int = 1000):
    """Convert Neo4j graph to NetworkX graph in batches"""
    G = nx.DiGraph()
    
    with neo4j_driver.session() as session:
        # Get total counts for progress bars
        total_nodes = session.run("MATCH (p:Paper) RETURN count(p) as count").single()["count"]
        total_edges = session.run("MATCH ()-[:CITES]->() RETURN count(*) as count").single()["count"]
        
        # Process nodes in batches
        with tqdm(total=total_nodes, desc="Adding nodes to NetworkX") as pbar:
            for offset in range(0, total_nodes, batch_size):
                nodes = session.run(
                    f"MATCH (p:Paper) RETURN p.id AS id "
                    f"SKIP {offset} LIMIT {batch_size}"
                )
                batch_nodes = [node["id"] for node in nodes]
                G.add_nodes_from(batch_nodes)
                pbar.update(len(batch_nodes))
        
        # Process edges in batches
        with tqdm(total=total_edges, desc="Adding edges to NetworkX") as pbar:
            processed = 0
            while processed < total_edges:
                edges = session.run(f"""
                    MATCH (p1:Paper)-[:CITES]->(p2:Paper) 
                    RETURN p1.id AS source, p2.id AS target 
                    SKIP {processed} LIMIT {batch_size}
                """).data()
                
                if not edges:
                    break
                
                G.add_edges_from((edge["source"], edge["target"]) for edge in edges)
                processed += len(edges)
                pbar.update(len(edges))
            
    return G

def simrank_spark(G, query_nodes, C, max_iterations=100, tolerance=0.0001, partition_size=1000):
    """Memory-optimized SimRank implementation with progress tracking"""
    spark = SparkSession.builder \
        .appName("SimRank") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "4g") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()


    # Process predecessors in batches
    predecessors = {}
    nodes = list(G.nodes())
    with tqdm(total=len(nodes), desc="Building predecessor map") as pbar:
        for i in range(0, len(nodes), partition_size):
            batch_nodes = nodes[i:i + partition_size]
            for node in batch_nodes:
                predecessors[node] = list(G.predecessors(node))
            pbar.update(len(batch_nodes))

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Initialize similarity matrix using sparse representation
    from scipy.sparse import lil_matrix
    sim_matrix = lil_matrix((n, n))
    for i in tqdm(range(n), desc="Initializing similarity matrix"):
        sim_matrix[i, i] = 1.0

    sim_rdd = spark.sparkContext.broadcast(sim_matrix.tocsr())

    def compute_pair_similarity(u, v):
        if u == v:
            return 1.0
        
        in_u = predecessors[u]
        in_v = predecessors[v]
        
        if not in_u or not in_v:
            return 0.0
        
        scale = C / (len(in_u) * len(in_v))
        curr_sim = sim_rdd.value
        
        sum_sim = 0
        for w, x in product(in_u, in_v):
            sum_sim += curr_sim[node_to_idx[w], node_to_idx[x]]
        
        return scale * sum_sim

    # Process iterations
    pbar = tqdm(total=max_iterations, desc="SimRank iterations")
    for iteration in range(max_iterations):
        new_sim = lil_matrix((n, n))
        
        # Process node pairs in batches with nested progress bars
        total_batches = ((n + partition_size - 1) // partition_size) ** 2
        batch_pbar = tqdm(total=total_batches, desc=f"Processing batch pairs", leave=False)
        
        for i in range(0, n, partition_size):
            for j in range(0, n, partition_size):
                batch_pairs = [(nodes[ii], nodes[jj]) 
                             for ii in range(i, min(i + partition_size, n))
                             for jj in range(j, min(j + partition_size, n))]
                
                node_pairs_rdd = spark.sparkContext.parallelize(batch_pairs)
                similarities = node_pairs_rdd.map(lambda pair: 
                    (pair, compute_pair_similarity(pair[0], pair[1]))).collect()
                
                for (u, v), sim in similarities:
                    new_sim[node_to_idx[u], node_to_idx[v]] = sim
                
                batch_pbar.update(1)
        
        batch_pbar.close()
        
        # Check convergence
        diff = abs(new_sim - sim_rdd.value).max()
        sim_rdd = spark.sparkContext.broadcast(new_sim.tocsr())
        
        pbar.set_postfix({'diff': f'{diff:.6f}'})
        pbar.update(1)
        
        if diff < tolerance:
            logging.info(f"Converged after {iteration + 1} iterations")
            break
    
    pbar.close()
    
    # Get results for query nodes
    results = {}
    for query_node in tqdm(query_nodes, desc="Processing query nodes"):
        if query_node not in node_to_idx:
            logging.warning(f"Query node {query_node} not found in graph")
            continue
            
        query_idx = node_to_idx[query_node]
        similarities = []
        
        # Process similarities in batches
        for i in range(0, n, partition_size):
            batch_nodes = nodes[i:i + partition_size]
            batch_similarities = [(node, sim_rdd.value[query_idx, node_to_idx[node]])
                                for node in batch_nodes if node != query_node]
            similarities.extend(batch_similarities)
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        results[query_node] = similarities
    
    spark.stop()
    return results
def main():
    # Create Neo4j graph with smaller batch size
    create_neo4j_graph("train.json", batch_size=500)
    
    neo4j_connection = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "paras2003")
    )
    
    # Convert to NetworkX with batching
    G = convert_to_networkx(neo4j_connection, batch_size=500)
    neo4j_connection.close()
    
    query_nodes = ["2982615777", "1556418098"]
    results = simrank_spark(G, query_nodes, C=0.8)
main()