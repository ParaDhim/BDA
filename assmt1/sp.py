from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import avg, col, count, collect_list, explode, array_contains, size

# spark = SparkSession.builder \
#     .appName("University Information System") \
#     .config("spark.mongodb.input.uri", "mongodb://localhost:27017/university_information_system") \
#     .config("spark.mongodb.output.uri", "mongodb://localhost:27017/university_information_system") \
#     .getOrCreate()
# from pyspark.sql import SparkSession

# spark = SparkSession.builder \
#     .appName("University Information System") \
#     .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
#     .config("spark.mongodb.input.uri", "mongodb://localhost:27017/university_information_system") \
#     .config("spark.mongodb.output.uri", "mongodb://localhost:27017/university_information_system") \
#     .getOrCreate()


from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("University Information System") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/university_information_system") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/university_information_system") \
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
    .getOrCreate()

students_df = spark.read.format("mongo").option("collection", "students").load()
courses_df = spark.read.format("mongo").option("collection", "courses").load()
instructors_df = spark.read.format("mongo").option("collection", "instructors").load()
departments_df = spark.read.format("mongo").option("collection", "departments").load()


students_df.show()
courses_df.show()
instructors_df.show()
departments_df.show()

if spark.sparkContext is None:
    print("SparkContext is not active!")
else:
    print("SparkContext is active.")

# 1. Fetching all students enrolled in a specific course
course_id = 1  # Replace with the specific course ID
enrolled_students = students_df \
    .filter(F.array_contains(students_df.enrollments.course_id, course_id)) \
    .select("first_name", "last_name", "email")


print("Enrolled Students in Course:", course_id)
enrolled_students.show()

# Print the schema for both DataFrames
students_df.printSchema()
instructors_df.printSchema()
courses_df.printSchema()
departments_df.printSchema()

# 2. Calculating the average number of students enrolled in courses offered by a particular instructor
instructor_id = 1  # Replace with the specific instructor ID
avg_enrollment = courses_df \
    .filter(F.array_contains(courses_df.instructors, instructor_id)) \
    .agg(F.avg("enrollment_count").alias("average_enrollment"))
    
print("Average Enrollment for Instructor:", instructor_id)
avg_enrollment.show()

# 3. Listing all courses offered by a specific department
department_id = 1  # Replace with the specific department ID
department_courses = courses_df \
    .filter(courses_df.department_id == department_id) \
    .select("course_name")
    
print("Courses in Department:", department_id)
department_courses.show()

# 4. Finding the total number of students per department
total_students_per_dept = students_df \
    .groupBy("department_id") \
    .agg(F.count("*").alias("total_students"))

print("Total Students per Department:")
total_students_per_dept.show()


# 5 Finding instructors who have taught all the BTech CSE core courses sometime during their tenure at the university.
# Define BTech CSE core courses (assuming these are the first 5 courses in the Computer Science department)
cs_department = departments_df.filter(col("department_name") == "Computer Science").first()
cs_department_id = cs_department["_id"]

core_courses = courses_df.filter(col("department_id") == cs_department_id) \
    .orderBy("_id") \
    .limit(5) \
    .select("_id")

core_course_ids = [row["_id"] for row in core_courses.collect()]

# Find instructors who have taught all core courses
qualified_instructors = instructors_df \
    .filter(size(col("courses_taught")) >= len(core_course_ids)) \
    .filter(array_contains(col("courses_taught"), core_course_ids[0]) &
            array_contains(col("courses_taught"), core_course_ids[1]) &
            array_contains(col("courses_taught"), core_course_ids[2]) &
            array_contains(col("courses_taught"), core_course_ids[3]) &
            array_contains(col("courses_taught"), core_course_ids[4]))

# Display results
print("Instructors who have taught all BTech CSE core courses:")
qualified_instructors.select("first_name", "last_name", "email").show()

# 6. Finding top-10 courses with the highest enrollments
top_courses = courses_df \
    .orderBy(courses_df.enrollment_count.desc()) \
    .limit(10) \
    .select("course_name", "enrollment_count")

print("Top 10 Courses by Enrollment:")
top_courses.show()

spark.stop()