import org.apache.spark.sql.SparkSession
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.util.Random

object GitHubCommunityDetection {
  def main(args: Array[String]): Unit = {
    // Initialize Spark Session
    val spark = SparkSession.builder()
      .appName("GitHub Community Detection")
      .master("local[*]")
      .getOrCreate()
    
    val sc = spark.sparkContext

    // File paths (replace with your actual paths)
    val edgesPath = "/Users/parasdhiman/Desktop/assmt/BDA/assmt4/musae_git_edges.csv"
    val nodesPath = "/Users/parasdhiman/Desktop/assmt/BDA/assmt4/musae_git_target.csv"

    // Load and preprocess edges
    val edgesRaw = sc.textFile(edgesPath)
      .filter(line => !line.startsWith("id_1"))
    
    val edges: RDD[Edge[Int]] = edgesRaw.map { line =>
      val parts = line.split(",").map(_.trim)
      Edge(parts(0).toLong, parts(1).toLong, 1)
    }

    // Load and preprocess nodes
    val nodesRaw = sc.textFile(nodesPath)
      .filter(line => !line.startsWith("id"))
    
    val vertices: RDD[(VertexId, Int)] = nodesRaw.map { line =>
      val parts = line.split(",").map(_.trim)
      (parts(0).toLong, parts(2).toInt)
    }

    // Create graph
    val graph = Graph(vertices, edges)

    // Initialize random community assignments
    val initialGraph = graph.mapVertices { case (id, _) => Random.nextLong() }

    // AGM-based community detection
    val maxIterations = 10
    val agmGraph = (0 until maxIterations).foldLeft(initialGraph) { (currentGraph, _) =>
      val updatedVertices = currentGraph.aggregateMessages[Map[Long, Int]](
        triplet => {
          triplet.sendToSrc(Map(triplet.dstAttr -> 1))
          triplet.sendToDst(Map(triplet.srcAttr -> 1))
        },
        (a, b) => (a.keySet ++ b.keySet).map(k => k -> (a.getOrElse(k, 0) + b.getOrElse(k, 0))).toMap
      )

      currentGraph.outerJoinVertices(updatedVertices) { 
        case (_, oldAttr, newAttrOpt) =>
          newAttrOpt match {
            case Some(newAttr) => newAttr.maxBy(_._2)._1
            case None => oldAttr
          }
      }
    }

    // Evaluate modularity
    val communities = agmGraph.vertices.map(_._2).distinct().collect()
    val modularity = communities.map { community =>
      val subgraph = agmGraph.subgraph(vpred = (_, attr) => attr == community)
      val internalEdges = subgraph.edges.count()
      val totalEdges = agmGraph.edges.count()
      internalEdges.toDouble / totalEdges
    }.sum

    // Print results
    println(s"Detected Communities: ${communities.mkString(", ")}")
    println(s"Modularity: $modularity")

    // Stop Spark Session
    spark.stop()
  }
}