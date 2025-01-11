from pyspark.sql import SparkSession
from neo4j import GraphDatabase
import pandas as pd
import os
from datetime import datetime
from tqdm.auto import tqdm
from collections import defaultdict

class CitationGraphAnalyzer:
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", neo4j_password="paras2003"):
        """Initialize with Neo4j and Spark connections"""
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, 
                                         auth=(neo4j_user, neo4j_password))
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("Citation Graph Analysis") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
    
    def create_neo4j_graph(self, papers_data):
        """Create graph in Neo4j from citation data"""
        with self.driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create paper nodes
            for paper_id in papers_data.keys():
                session.run("""
                    CREATE (p:Paper {id: $paper_id})
                """, paper_id=paper_id)
            
            # Create citation relationships
            for citing_paper, cited_papers in papers_data.items():
                if cited_papers:  # Only create edges if there are references
                    for cited_paper in cited_papers:
                        session.run("""
                            MATCH (citing:Paper {id: $citing_id})
                            MATCH (cited:Paper {id: $cited_id})
                            CREATE (citing)-[:CITES]->(cited)
                        """, citing_id=citing_paper, cited_id=cited_paper)
                        
    def get_graph_data(self):
        """Extract graph data from Neo4j for Spark processing"""
        with self.driver.session() as session:
            # Get all citation relationships
            result = session.run("""
                MATCH (p1:Paper)-[:CITES]->(p2:Paper)
                RETURN p1.id as source, p2.id as target
            """)
            edges = [(record["source"], record["target"]) for record in result]
            
            return edges
    
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
                    if na == nb:
                        new_scores[na][nb] = 1.0
                        continue
                    
                    in_na = in_neighbors_dict.get(na, [])
                    in_nb = in_neighbors_dict.get(nb, [])
                    
                    if not in_na or not in_nb:
                        continue
                    
                    similarity_sum = sum(sim_scores[i][j] 
                                      for i in in_na 
                                      for j in in_nb)
                    
                    new_sim = (C / (len(in_na) * len(in_nb))) * similarity_sum
                    new_scores[na][nb] = new_sim
                    new_scores[nb][na] = new_sim
                    
                    max_diff = max(max_diff, abs(new_sim - sim_scores[na][nb]))
            
            sim_scores = new_scores
            
            if max_diff < tolerance:
                break
        
        similarity_sum = sum(sim_scores[i][j] 
                           for i in in_neighbors_a 
                           for j in in_neighbors_b)
        
        if not in_neighbors_a or not in_neighbors_b:
            return 0.0
            
        return (C / (len(in_neighbors_a) * len(in_neighbors_b))) * similarity_sum
    
    def analyze_citation_graph(self, query_nodes, decay_factors, output_dir="simrank_results"):
        """Run SimRank analysis on citation graph"""
        os.makedirs(output_dir, exist_ok=True)
        assert os.path.isdir(output_dir), f"Could not create or access directory: {output_dir}"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get graph data from Neo4j
        edges = self.get_graph_data()
        edges_df = self.spark.createDataFrame(edges, ["source", "target"])
        
        # Cache in-neighbors
        in_neighbors_dict = self._cache_in_neighbors(edges_df)
        
        all_results = []
        
        for C in decay_factors:
            print(f"\nComputing SimRank with decay factor C = {C}")
            
            results = []
            # Get all unique nodes
            all_nodes = set(row['source'] for row in edges_df.select("source").distinct().collect())
            all_nodes.update(row['target'] for row in edges_df.select("target").distinct().collect())


            
            for query_node in tqdm(query_nodes, desc="Processing query nodes"):
                node_results = []
                for target_node in tqdm(all_nodes, desc=f"Computing similarities for node {query_node}", leave=False):
                    sim = self.compute_simrank_similarity(
                        query_node,
                        target_node,
                        in_neighbors_dict,
                        C
                    )
                    node_results.append((query_node, target_node, sim))
                results.extend(node_results)
            
            results_df = pd.DataFrame(results, columns=['query_node', 'target_node', 'similarity'])
            results_df['decay_factor'] = C
            all_results.append(results_df)
            
            # Save intermediate results
            output_path = f"{output_dir}/simrank_results_C{C}_{timestamp}.csv"
            results_df.to_csv(output_path, index=False)
            
        return self._save_and_summarize_results(all_results, query_nodes, decay_factors, timestamp, output_dir)
    
    def _cache_in_neighbors(self, edges_df):
        """Cache in-neighbors for all nodes"""
        in_neighbors = edges_df.groupBy('target').agg(F.collect_list('source').alias('in_neighbors'))
        return {row['target']: row['in_neighbors'] for row in in_neighbors.collect()}

    
    def _save_and_summarize_results(self, all_results, query_nodes, decay_factors, timestamp, output_dir):
        """Save and summarize final results"""
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Save complete results
        final_path = f"{output_dir}/simrank_all_results_{timestamp}.csv"
        final_results.to_csv(final_path, index=False)
        
        # Generate top results
        top_results = []
        for C in decay_factors:
            for query in query_nodes:
                mask = (final_results['decay_factor'] == C) & (final_results['query_node'] == query)
                subset = final_results[mask].nlargest(10, 'similarity')
                subset = subset.copy()
                subset['rank'] = range(1, len(subset) + 1)
                top_results.append(subset)
        
        top_results_df = pd.concat(top_results, ignore_index=True)
        top_path = f"{output_dir}/simrank_top_results_{timestamp}.csv"
        top_results_df.to_csv(top_path, index=False)
        
        self._print_summary(top_results_df, decay_factors, query_nodes)
        
        return final_results, top_results_df
    
    def _print_summary(self, top_results_df, decay_factors, query_nodes):
        """Print summary of results"""
        print("\nTop 5 most similar nodes for each query node and decay factor:")
        for C in decay_factors:
            print(f"\nDecay factor C = {C}")
            for query in query_nodes:
                print(f"\nQuery node: {query}")
                mask = (top_results_df['decay_factor'] == C) & (top_results_df['query_node'] == query)
                top_5 = top_results_df[mask].head()
                if not top_5.empty:
                    print(top_5[['target_node', 'similarity', 'rank']].to_string())
                else:
                    print("No results found")
    
    def close(self):
        """Close Neo4j and Spark connections"""
        self.driver.close()
        self.spark.stop()

# Example usage
# if __name__ == "__main__":
    # Sample citation data
papers_data = {
    2982615777: [2087551257, 2044328306],
    2044328306: [2087551257],
    2087551257: [2044328306],
    1556418098: []  # Paper with no references
}

# Initialize analyzer
analyzer = CitationGraphAnalyzer()

try:
    # Create Neo4j graph
    print("Creating Neo4j graph...")
    analyzer.create_neo4j_graph(papers_data)
    
    # Run analysis
    query_nodes = [2982615777, 1556418098]
    decay_factors = [0.7, 0.8, 0.9]
    
    print("Running SimRank analysis...")
    final_results, top_results = analyzer.analyze_citation_graph(
        query_nodes=query_nodes,
        decay_factors=decay_factors
    )
    
finally:
    # Clean up connections
    analyzer.close()