from pyspark.sql import SparkSession
from neo4j import GraphDatabase
import pandas as pd
import os
from datetime import datetime
from tqdm.auto import tqdm
from collections import defaultdict
from pyspark.sql import functions as F
import json

class CitationGraphAnalyzer:
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", neo4j_password="paras2003"):
        """Initialize with Neo4j and Spark connections"""
        # Initialize Neo4j connection
        # self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("Citation Graph Analysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
    
    # def create_neo4j_graph(self, papers_data):
    #     """Create graph in Neo4j from citation data"""
    #     with self.driver.session() as session:
    #         # Clear existing data
    #         session.run("MATCH (n) DETACH DELETE n")
            
    #         # Create paper nodes and relationships
    #         for citing_paper, cited_papers in tqdm(papers_data.items(), desc="Creating graph in Neo4j", ncols=100):
    #             session.run("MERGE (p:Paper {id: $paper_id})", paper_id=citing_paper)
    #             for cited_paper in cited_papers:
    #                 session.run("""
    #                     MERGE (cited:Paper {id: $cited_id})
    #                     MERGE (citing:Paper {id: $citing_id})
    #                     MERGE (citing)-[:CITES]->(cited)
    #                 """, cited_id=cited_paper, citing_id=citing_paper)
                        
    # def get_graph_data(self):
    #     """Extract graph data from Neo4j for Spark processing"""
    #     with self.driver.session() as session:
    #         result = session.run("""
    #             MATCH (p1:Paper)-[:CITES]->(p2:Paper)
    #             RETURN p1.id as source, p2.id as target
    #         """)
    #         edges = [(record["source"], record["target"]) for record in result]
    #         return edges
    
    def compute_simrank_similarity(self, a, b, in_neighbors_dict, C, max_iterations=10, tolerance=1e-4):
        """Compute SimRank similarity between two nodes"""
        if a == b:
            return 1.0
            
        in_neighbors_a = in_neighbors_dict.get(a, [])
        in_neighbors_b = in_neighbors_dict.get(b, [])
        
        if not in_neighbors_a or not in_neighbors_b:
            return 0.0
        
        # Initialize similarity scores
        sim_scores = defaultdict(lambda: defaultdict(float))
        for node in set(in_neighbors_a + in_neighbors_b):
            sim_scores[node][node] = 1.0
        
        # SimRank iterations
        for _ in range(max_iterations):
            new_scores = defaultdict(lambda: defaultdict(float))
            max_diff = 0.0
            
            for na in in_neighbors_a:
                for nb in in_neighbors_b:
                    in_na = in_neighbors_dict.get(na, [])
                    in_nb = in_neighbors_dict.get(nb, [])
                    
                    if not in_na or not in_nb:
                        continue
                    
                    similarity_sum = sum(sim_scores[i][j] for i in in_na for j in in_nb)
                    new_sim = (C / (len(in_na) * len(in_nb))) * similarity_sum
                    new_scores[na][nb] = new_sim
                    max_diff = max(max_diff, abs(new_sim - sim_scores[na][nb]))
            
            sim_scores = new_scores
            
            if max_diff < tolerance:
                break
        
        similarity_sum = sum(sim_scores[i][j] for i in in_neighbors_a for j in in_neighbors_b)
        return (C / (len(in_neighbors_a) * len(in_neighbors_b))) * similarity_sum
    
    def analyze_citation_graph(self, query_nodes, decay_factors, output_dir="simrank_results"):
        """Run SimRank analysis on citation graph"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get graph data from Neo4j
        edges = self.get_graph_data()
        edges_df = self.spark.createDataFrame(edges, ["source", "target"])
        in_neighbors_dict = self._cache_in_neighbors(edges_df)
        
        all_results = []
        
        for C in decay_factors:
            results = []
            all_nodes = set(edges_df.select("source").distinct().toPandas()["source"])
            all_nodes.update(edges_df.select("target").distinct().toPandas()["target"])
            
            for query_node in query_nodes:
                for target_node in all_nodes:
                    sim = self.compute_simrank_similarity(query_node, target_node, in_neighbors_dict, C)
                    results.append((query_node, target_node, sim))
            
            results_df = pd.DataFrame(results, columns=['query_node', 'target_node', 'similarity'])
            results_df['decay_factor'] = C
            results_df.to_csv(f"{output_dir}/simrank_results_C{C}_{timestamp}.csv", index=False)
            all_results.append(results_df)
        
        return all_results
    
    def _cache_in_neighbors(self, edges_df):
        """Cache in-neighbors for all nodes"""
        in_neighbors = edges_df.groupBy('target').agg(F.collect_list('source').alias('in_neighbors'))
        return {row['target']: row['in_neighbors'] for row in in_neighbors.collect()}
    
    def close(self):
        """Close Neo4j and Spark connections"""
        # self.driver.close()
        self.spark.stop()

# Initialize analyzer
analyzer = CitationGraphAnalyzer()

try:
    # Create Neo4j graph
    analyzer.create_neo4j_graph(papers_data)
    
    # Run SimRank analysis
    query_nodes = [2982615777, 1556418098]
    decay_factors = [0.7, 0.8, 0.9]
    results = analyzer.analyze_citation_graph(query_nodes, decay_factors)
    
finally:
    analyzer.close()
