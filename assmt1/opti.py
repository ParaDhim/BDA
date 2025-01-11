from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import avg, col, count, collect_list, explode, array_contains, size, broadcast
import time

def create_spark_session():
    return SparkSession.builder \
        .appName("University Information System") \
        .config("spark.mongodb.input.uri", "mongodb://localhost:27017/university_information_system") \
        .config("spark.mongodb.output.uri", "mongodb://localhost:27017/university_information_system") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()

def load_data(spark):
    students_df = spark.read.format("mongo").option("collection", "students").load()
    courses_df = spark.read.format("mongo").option("collection", "courses").load()
    instructors_df = spark.read.format("mongo").option("collection", "instructors").load()
    departments_df = spark.read.format("mongo").option("collection", "departments").load()
    return students_df, courses_df, instructors_df, departments_df

def measure_performance(func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Initialize Spark session and load data
spark = create_spark_session()
students_df, courses_df, instructors_df, departments_df = load_data(spark)

# Query 1: Fetching all students enrolled in a specific course
def query1_original():
    course_id = 1
    return students_df \
        .filter(F.array_contains(students_df.enrollments.course_id, course_id)) \
        .select("first_name", "last_name", "email")

def query1_optimized():
    course_id = 1
    # Optimization: Use broadcast join
    enrollments = broadcast(F.array(F.struct(F.lit(course_id).alias("course_id"))))
    return students_df \
        .filter(F.array_contains(students_df.enrollments, enrollments[0])) \
        .select("first_name", "last_name", "email")

# Query 2: Calculating the average number of students enrolled in courses offered by a particular instructor
def query2_original():
    instructor_id = 1
    return courses_df \
        .filter(F.array_contains(courses_df.instructors, instructor_id)) \
        .agg(F.avg("enrollment_count").alias("average_enrollment"))

def query2_optimized():
    instructor_id = 1
    # Optimization: Use caching
    cached_courses_df = courses_df.cache()
    return cached_courses_df \
        .filter(F.array_contains(cached_courses_df.instructors, instructor_id)) \
        .agg(F.avg("enrollment_count").alias("average_enrollment"))

# Query 3: Listing all courses offered by a specific department
def query3_original():
    department_id = 1
    return courses_df \
        .filter(courses_df.department_id == department_id) \
        .select("course_name")

def query3_optimized():
    department_id = 1
    # Optimization: Use broadcast join
    department = broadcast(F.lit(department_id))
    return courses_df \
        .filter(courses_df.department_id == department) \
        .select("course_name")

# Query 4: Finding the total number of students per department
def query4_original():
    return students_df \
        .groupBy("department_id") \
        .agg(F.count("*").alias("total_students"))

def query4_optimized():
    # Optimization: Use caching and repartitioning
    cached_students_df = students_df.cache()
    return cached_students_df \
        .repartition("department_id") \
        .groupBy("department_id") \
        .agg(F.count("*").alias("total_students"))

# Query 5: Finding instructors who have taught all the BTech CSE core courses
def query5_original():
    cs_department = departments_df.filter(col("department_name") == "Computer Science").first()
    cs_department_id = cs_department["_id"]
    core_courses = courses_df.filter(col("department_id") == cs_department_id) \
        .orderBy("_id") \
        .limit(5) \
        .select("_id")
    core_course_ids = [row["_id"] for row in core_courses.collect()]
    return instructors_df \
        .filter(size(col("courses_taught")) >= len(core_course_ids)) \
        .filter(array_contains(col("courses_taught"), core_course_ids[0]) &
                array_contains(col("courses_taught"), core_course_ids[1]) &
                array_contains(col("courses_taught"), core_course_ids[2]) &
                array_contains(col("courses_taught"), core_course_ids[3]) &
                array_contains(col("courses_taught"), core_course_ids[4]))

def query5_optimized():
    # Optimization: Use broadcast variables and caching
    cs_department = broadcast(departments_df.filter(col("department_name") == "Computer Science").first())
    cs_department_id = cs_department["_id"]
    core_courses = broadcast(courses_df.filter(col("department_id") == cs_department_id) \
        .orderBy("_id") \
        .limit(5) \
        .select("_id"))
    core_course_ids = [row["_id"] for row in core_courses.collect()]
    cached_instructors_df = instructors_df.cache()
    return cached_instructors_df \
        .filter(size(col("courses_taught")) >= len(core_course_ids)) \
        .filter(array_contains(col("courses_taught"), core_course_ids[0]) &
                array_contains(col("courses_taught"), core_course_ids[1]) &
                array_contains(col("courses_taught"), core_course_ids[2]) &
                array_contains(col("courses_taught"), core_course_ids[3]) &
                array_contains(col("courses_taught"), core_course_ids[4]))

# Query 6: Finding top-10 courses with the highest enrollments
def query6_original():
    return courses_df \
        .orderBy(courses_df.enrollment_count.desc()) \
        .limit(10) \
        .select("course_name", "enrollment_count")

def query6_optimized():
    # Optimization: Use caching and repartitioning
    cached_courses_df = courses_df.cache()
    return cached_courses_df \
        .repartition(1) \
        .orderBy(cached_courses_df.enrollment_count.desc()) \
        .limit(10) \
        .select("course_name", "enrollment_count")

# Run performance tests
queries = [
    ("Query 1", query1_original, query1_optimized),
    ("Query 2", query2_original, query2_optimized),
    ("Query 3", query3_original, query3_optimized),
    ("Query 4", query4_original, query4_optimized),
    ("Query 5", query5_original, query5_optimized),
    ("Query 6", query6_original, query6_optimized)
]

for query_name, original_func, optimized_func in queries:
    original_result, original_time = measure_performance(original_func)
    optimized_result, optimized_time = measure_performance(optimized_func)
    
    print(f"\n{query_name} Performance:")
    print(f"Original execution time: {original_time:.4f} seconds")
    print(f"Optimized execution time: {optimized_time:.4f} seconds")
    print(f"Improvement: {(original_time - optimized_time) / original_time * 100:.2f}%")

spark.stop()