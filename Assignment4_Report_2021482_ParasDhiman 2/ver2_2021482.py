from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from graphframes import GraphFrame
import numpy as np
from typing import Iterator, Tuple

class GitHubCommunityDetection:
    def __init__(self, edges_path: str, vertices_path: str):
        self.spark = SparkSession.builder \
            .appName("GitHub Community Detection") \
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "8g") \
            .config("spark.executor.cores", "4") \
            .getOrCreate()
            
        # Define schemas for the actual data format
        self.edges_schema = StructType([
            StructField("id_1", IntegerType(), False),
            StructField("id_2", IntegerType(), False)
        ])
        
        self.vertices_schema = StructType([
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), False),
            StructField("ml_target", IntegerType(), False)
        ])
        
        # Read data with the correct schema
        self.edges_df = self.spark.read.csv(edges_path, header=True, schema=self.edges_schema)
        self.vertices_df = self.spark.read.csv(vertices_path, header=True, schema=self.vertices_schema)
        
        # Prepare DataFrames for GraphFrames and rename columns to avoid ambiguity
        self.edges_df = self.edges_df \
            .withColumnRenamed("id_1", "src") \
            .withColumnRenamed("id_2", "dst")
            
        # Keep original column names for vertices DataFrame
        self.vertices_df = self.vertices_df.alias("vertices")

    def detect_communities(self, max_iterations: int = 200) -> None:
        """
        Detect communities using Label Propagation Algorithm and calculate modularity.
        
        Args:
            max_iterations: Maximum number of iterations for LPA
        """
        # Create GraphFrame
        graph = GraphFrame(self.vertices_df, self.edges_df)
        
        # Run LPA for community detection
        communities = graph.labelPropagation(maxIter=max_iterations)
        
        # Calculate modularity
        modularity = self._calculate_modularity(graph, communities)
        
        # Get community statistics
        community_stats = communities.groupBy("label").count().orderBy("count", ascending=False)
        
        # Print results
        print(f"\nNetwork Modularity: {modularity:.4f}")
        print("\nTop 10 Communities by Size:")
        community_stats.show(10)
        
        # Save results
        self._save_results(communities)
        self.spark.stop()

    def _calculate_modularity(self, graph: GraphFrame, communities) -> float:
        """
        Calculate modularity score for the network.
        
        Args:
            graph: GraphFrame object containing the network
            communities: DataFrame with community assignments
            
        Returns:
            float: Modularity score
        """
        # Get total number of edges
        m = graph.edges.count()
        
        # Calculate node degrees
        degrees = graph.degrees
        
        # Prepare community DataFrames with unique column names
        src_communities = communities.select(
            col("id").alias("src"),
            col("label").alias("src_community")
        )
        
        dst_communities = communities.select(
            col("id").alias("dst"),
            col("label").alias("dst_community")
        )
        
        # Join community assignments with edges using explicit column references
        edges_with_communities = graph.edges \
            .join(src_communities, "src") \
            .join(dst_communities, "dst")
            
        # Prepare degree DataFrames with unique column names
        src_degrees = degrees.select(
            col("id").alias("src"),
            col("degree").alias("src_degree")
        )
        
        dst_degrees = degrees.select(
            col("id").alias("dst"),
            col("degree").alias("dst_degree")
        )
        
        # Join with degrees using explicit column references
        edges_with_info = edges_with_communities \
            .join(src_degrees, "src") \
            .join(dst_degrees, "dst")
        
        # Calculate modularity contributions
        def modularity_contribution(src_comm, dst_comm, src_deg, dst_deg) -> float:
            if src_comm == dst_comm:
                return 1.0 - (src_deg * dst_deg) / (2.0 * m)
            return 0.0
        
        # Register UDF
        from pyspark.sql.functions import udf
        mod_udf = udf(modularity_contribution)
        
        # Calculate total modularity
        modularity = edges_with_info \
            .select(mod_udf("src_community", "dst_community", "src_degree", "dst_degree").alias("contribution")) \
            .agg({"contribution": "sum"}) \
            .collect()[0][0]
            
        return modularity / (2.0 * m)

    def _save_results(self, communities) -> None:
        """
        Save community detection results.
        
        Args:
            communities: DataFrame with community assignments
        """
        # First, create a view of the vertices DataFrame
        self.vertices_df.createOrReplaceTempView("vertices_view")
        
        # Create a view of the communities DataFrame with selected columns
        communities_selected = communities.select(
            col("id"),
            col("label").alias("community_label")
        )
        communities_selected.createOrReplaceTempView("communities_view")
        
        # Use SQL to join the data and select columns
        results = self.spark.sql("""
            SELECT 
                c.id,
                v.name,
                v.ml_target,
                c.community_label as community
            FROM communities_view c
            JOIN vertices_view v ON c.id = v.id
        """)
        
        # Save to CSV
        results.write \
            .mode("overwrite") \
            .option("header", True) \
            .csv("community_results")

def main():
    # Initialize and run community detection
    detector = GitHubCommunityDetection(
        edges_path="musae_git_edges.csv",
        vertices_path="musae_git_target.csv"
    )
    detector.detect_communities()

if __name__ == "__main__":
    main()