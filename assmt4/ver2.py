import os
import json
import numpy as np
import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode, struct
from pyspark.sql.types import (
    StructType, StructField, 
    StringType, IntegerType, 
    ArrayType, DoubleType, BooleanType
)
from typing import List, Dict, Any

import logging

class AffiliationGraphModelDetector:
    def __init__(
        self, 
        edges_path: str, 
        nodes_path: str = None, 
        features_path: str = None, 
        num_communities: int = 5, 
        max_iterations: int = 100
    ):
        """
        Initialize Affiliation Graph Model Community Detector
        
        :param edges_path: Path to edges CSV file
        :param nodes_path: Optional path to nodes CSV file
        :param features_path: Optional path to features JSON file
        :param num_communities: Number of communities to detect
        :param max_iterations: Maximum iterations for community detection
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Validate input paths
        self._validate_file_paths(edges_path, nodes_path, features_path)
        
        # Store paths and parameters
        self.edges_path = edges_path
        self.nodes_path = nodes_path
        self.features_path = features_path
        self.num_communities = num_communities
        self.max_iterations = max_iterations
        
        # Initialize Spark Session
        self.spark = self._create_spark_session()
        
        # Load data
        self.edges_df = self._load_edges()
        self.nodes_df = self._load_nodes()
        self.features = self._load_features()
        
        # Create NetworkX graph for advanced community detection
        self.graph = self._create_networkx_graph()
    
    def _validate_file_paths(self, *paths):
        """
        Validate that input file paths exist
        
        :param paths: Paths to validate
        """
        for path in paths:
            if path and not os.path.exists(path):
                self.logger.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")
    
    def _create_spark_session(self):
        """
        Create Spark Session with necessary configurations
        
        :return: Configured SparkSession
        """
        return SparkSession.builder \
            .appName("Affiliation Graph Model Community Detection") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
    
    def _load_edges(self):
        """
        Load edges from CSV file
        
        :return: Spark DataFrame of edges
        """
        try:
            edges_df = self.spark.read.csv(self.edges_path, header=True)
            
            # Rename columns if needed
            if 'id_1' in edges_df.columns and 'id_2' in edges_df.columns:
                edges_df = edges_df.withColumnRenamed("id_1", "src") \
                                   .withColumnRenamed("id_2", "dst")
            
            self.logger.info(f"Loaded {edges_df.count()} edges")
            return edges_df
        except Exception as e:
            self.logger.error(f"Error loading edges: {e}")
            raise
    
    def _load_nodes(self):
        """
        Load nodes from CSV file or generate from edges
        
        :return: Spark DataFrame of nodes
        """
        if self.nodes_path:
            try:
                nodes_df = self.spark.read.csv(self.nodes_path, header=True)
                self.logger.info(f"Loaded {nodes_df.count()} nodes")
                return nodes_df
            except Exception as e:
                self.logger.warning(f"Error loading nodes file: {e}. Generating nodes from edges.")
        
        # Generate nodes from unique node IDs in edges
        nodes_df = self.edges_df.select(col("src").alias("id")) \
            .union(self.edges_df.select(col("dst").alias("id"))) \
            .distinct()
        
        self.logger.info(f"Generated {nodes_df.count()} nodes from edges")
        return nodes_df
    
    def _load_features(self):
        """
        Load features from JSON file
        
        :return: Dictionary of node features or None
        """
        if not self.features_path:
            self.logger.info("No features path provided")
            return None
        
        try:
            with open(self.features_path, 'r') as f:
                features = json.load(f)
            self.logger.info(f"Loaded features for {len(features)} nodes")
            return features
        except Exception as e:
            self.logger.error(f"Error loading features: {e}")
            return None
    
    def _create_networkx_graph(self):
        """
        Create NetworkX graph from Spark edges DataFrame
        
        :return: NetworkX Graph
        """
        try:
            # Convert Spark DataFrame to list of edges
            edges = self.edges_df.select("src", "dst").collect()
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_edges_from([(row.src, row.dst) for row in edges])
            
            self.logger.info(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
        except Exception as e:
            self.logger.error(f"Error creating NetworkX graph: {e}")
            raise
    
    def detect_communities(self):
        """
        Detect communities using Louvain method
        
        :return: Communities and modularity
        """
        try:
            # Use Louvain method for community detection
            communities = list(nx.community.louvain_communities(self.graph))
            
            # Calculate modularity
            modularity = nx.community.modularity(self.graph, communities)
            
            # Log results
            self.logger.info(f"Detected {len(communities)} communities")
            self.logger.info(f"Modularity Score: {modularity}")
            
            return {
                "communities": communities,
                "modularity": modularity,
                "num_communities": len(communities)
            }
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            raise
    
    def export_results(self, results, output_path='community_results.json'):
        """
        Export community detection results
        
        :param results: Community detection results
        :param output_path: Path to save results
        """
        try:
            # Convert communities to a list of node lists
            serializable_communities = [list(community) for community in results['communities']]
            
            export_data = {
                "num_communities": results['num_communities'],
                "modularity": results['modularity'],
                "communities": serializable_communities
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Results exported to {output_path}")
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
    
    def analyze_communities(self, results):
        """
        Perform additional analysis on detected communities
        
        :param results: Community detection results
        """
        try:
            communities = results['communities']
            
            # Community size distribution
            community_sizes = [len(community) for community in communities]
            
            print("\nCommunity Analysis:")
            print(f"Total Communities: {len(communities)}")
            print(f"Average Community Size: {np.mean(community_sizes):.2f}")
            print(f"Smallest Community Size: {min(community_sizes)}")
            print(f"Largest Community Size: {max(community_sizes)}")
            
            # Optional: Feature-based analysis if features are available
            if self.features:
                self._analyze_community_features(communities)
        
        except Exception as e:
            self.logger.error(f"Community analysis failed: {e}")
    
    def _analyze_community_features(self, communities):
        """
        Analyze community features if available
        
        :param communities: List of communities
        """
        if not self.features:
            return
        
        # Placeholder for feature-based analysis
        # You can extend this to analyze feature distributions in communities
        pass

def main(
    edges_path, 
    nodes_path=None, 
    features_path=None, 
    num_communities=5, 
    max_iterations=50
):
    """
    Main function to run Affiliation Graph Model Community Detection
    
    :param edges_path: Path to edges CSV file
    :param nodes_path: Optional path to nodes CSV file
    :param features_path: Optional path to features JSON file
    :param num_communities: Number of communities to detect
    :param max_iterations: Maximum iterations for community detection
    """
    # Create detector
    detector = AffiliationGraphModelDetector(
        edges_path=edges_path, 
        nodes_path=nodes_path,
        features_path=features_path,
        num_communities=num_communities,
        max_iterations=max_iterations
    )
    
    # Detect communities
    results = detector.detect_communities()
    
    # Analyze communities
    detector.analyze_communities(results)
    
    # Export results
    detector.export_results(results)
    
    print("\nCommunity Detection Completed Successfully!")

if __name__ == "__main__":
    # REPLACE THESE PATHS WITH YOUR ACTUAL FILE PATHS
    # EDGES_PATH = "/path/to/your/edges.csv"
    # NODES_PATH = "/path/to/your/nodes.csv"  # Optional
    # FEATURES_PATH = "/path/to/your/features.json"  # Optional
    
    EDGES_PATH = "/Users/parasdhiman/Desktop/assmt/BDA/assmt4/git_web_ml/musae_git_edges.csv"
    NODES_PATH = "/Users/parasdhiman/Desktop/assmt/BDA/assmt4/git_web_ml/musae_git_target.csv"
    FEATURES_PATH = "/Users/parasdhiman/Desktop/assmt/BDA/assmt4/git_web_ml/musae_git_features.json"
    
    # Run the main function
    main(
        edges_path=EDGES_PATH, 
        nodes_path=NODES_PATH,
        features_path=FEATURES_PATH
    )