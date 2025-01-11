import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.*;
import org.apache.spark.graphx.Graph;
import org.apache.spark.graphframes.GraphFrame;
import org.apache.spark.sql.functions;

import java.io.Serializable;

public class GitHubCommunityDetection implements Serializable {
    private transient SparkSession spark;
    private Dataset<Row> edgesDf;
    private Dataset<Row> verticesDf;
    
    public GitHubCommunityDetection(String edgesPath, String verticesPath) {
        // Initialize Spark Session
        spark = SparkSession.builder()
            .appName("GitHub Community Detection")
            .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "8g")
            .config("spark.executor.cores", "4")
            .getOrCreate();
        
        // Define schemas for the actual data format
        StructType edgesSchema = new StructType(new StructField[]{
            DataTypes.createStructField("id_1", DataTypes.IntegerType, false),
            DataTypes.createStructField("id_2", DataTypes.IntegerType, false)
        });
        
        StructType verticesSchema = new StructType(new StructField[]{
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, false),
            DataTypes.createStructField("ml_target", DataTypes.IntegerType, false)
        });
        
        // Read data with the correct schema
        edgesDf = spark.read().schema(edgesSchema).option("header", true).csv(edgesPath)
            .withColumnRenamed("id_1", "src")
            .withColumnRenamed("id_2", "dst");
        
        verticesDf = spark.read().schema(verticesSchema).option("header", true).csv(verticesPath);
    }
    
    public void detectCommunities() {
        return detectCommunities(200);
    }
    
    public void detectCommunities(int maxIterations) {
        // Create GraphFrame
        GraphFrame graph = GraphFrame.apply(verticesDf, edgesDf);
        
        // Run Label Propagation Algorithm for community detection
        Dataset<Row> communities = graph.labelPropagation(maxIterations);
        
        // Calculate modularity
        double modularity = calculateModularity(graph, communities);
        
        // Get community statistics
        Dataset<Row> communityStats = communities.groupBy("label")
            .count()
            .orderBy(functions.col("count").desc());
        
        // Print results
        System.out.printf("\nNetwork Modularity: %.4f\n", modularity);
        System.out.println("\nTop 10 Communities by Size:");
        communityStats.show(10);
        
        // Save results
        saveResults(communities);
    }
    
    private double calculateModularity(GraphFrame graph, Dataset<Row> communities) {
        // Get total number of edges
        long m = graph.edges().count();
        
        // Calculate node degrees
        Dataset<Row> degrees = graph.degrees();
        
        // Prepare community DataFrames with unique column names
        Dataset<Row> srcCommunities = communities.select(
            functions.col("id").alias("src"),
            functions.col("label").alias("src_community")
        );
        
        Dataset<Row> dstCommunities = communities.select(
            functions.col("id").alias("dst"),
            functions.col("label").alias("dst_community")
        );
        
        // Join community assignments with edges
        Dataset<Row> edgesWithCommunities = graph.edges()
            .join(srcCommunities, "src")
            .join(dstCommunities, "dst");
        
        // Prepare degree DataFrames with unique column names
        Dataset<Row> srcDegrees = degrees.select(
            functions.col("id").alias("src"),
            functions.col("degree").alias("src_degree")
        );
        
        Dataset<Row> dstDegrees = degrees.select(
            functions.col("id").alias("dst"),
            functions.col("degree").alias("dst_degree")
        );
        
        // Join with degrees 
        Dataset<Row> edgesWithInfo = edgesWithCommunities
            .join(srcDegrees, "src")
            .join(dstDegrees, "dst");
        
        // Register UDF for modularity contribution
        spark.udf().register("modularity_contribution", 
            (Integer srcComm, Integer dstComm, Integer srcDeg, Integer dstDeg) -> {
                if (srcComm.equals(dstComm)) {
                    return 1.0 - (srcDeg * dstDeg) / (2.0 * m);
                }
                return 0.0;
            }, DataTypes.DoubleType);
        
        // Calculate total modularity
        double modularity = edgesWithInfo
            .selectExpr("modularity_contribution(src_community, dst_community, src_degree, dst_degree) as contribution")
            .agg(functions.sum("contribution"))
            .collectAsList()
            .get(0)
            .getDouble(0);
        
        return modularity / (2.0 * m);
    }
    
    private void saveResults(Dataset<Row> communities) {
        // Create temporary views
        verticesDf.createOrReplaceTempView("vertices_view");
        
        Dataset<Row> communitiesSelected = communities.select(
            functions.col("id"),
            functions.col("label").alias("community_label")
        );
        communitiesSelected.createOrReplaceTempView("communities_view");
        
        // Use SQL to join the data and select columns
        Dataset<Row> results = spark.sql(
            "SELECT " +
            "c.id, " +
            "v.name, " +
            "v.ml_target, " +
            "c.community_label as community " +
            "FROM communities_view c " +
            "JOIN vertices_view v ON c.id = v.id"
        );
        
        // Save to CSV
        results.write()
            .mode("overwrite")
            .option("header", true)
            .csv("community_results");
    }
    
    // Close Spark session
    public void close() {
        if (spark != null) {
            spark.close();
        }
    }
    
    public static void main(String[] args) {
        GitHubCommunityDetection detector = new GitHubCommunityDetection(
            "musae_git_edges.csv",
            "musae_git_target.csv"
        );
        
        try {
            detector.detectCommunities();
        } finally {
            detector.close();
        }
    }
}