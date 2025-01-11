from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Create Spark session with MongoDB connection
spark = SparkSession.builder \
    .appName("MongoDB Spark Integration") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/university_information_system") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/university_information_system") \
    .getOrCreate()

# Load collections into DataFrames
students_df = spark.read.format("mongo").load("students")
courses_df = spark.read.format("mongo").load("courses")
instructors_df = spark.read.format("mongo").load("instructors")
departments_df = spark.read.format("mongo").load("departments")

# 1. Fetching all students enrolled in a specific course
course_id = 1  # Replace with the specific course ID
enrolled_students = students_df \
    .filter(F.array_contains(students_df.enrollments.course_id, course_id)) \
    .select("first_name", "last_name", "email")

# 2. Calculating the average number of students enrolled in courses offered by a particular instructor
# instructor_id = 1  # Replace with the specific instructor ID
# avg_enrollment = courses_df \
#     .filter(F.array_contains(courses_df.instructors, instructor_id)) \
#     .agg(F.avg("enrollment_count").alias("average_enrollment"))

# # 3. Listing all courses offered by a specific department
# department_id = 1  # Replace with the specific department ID
# department_courses = courses_df \
#     .filter(courses_df.department_id == department_id) \
#     .select("course_name")

# # 4. Finding the total number of students per department
# total_students_per_dept = students_df \
#     .groupBy("department_id") \
#     .agg(F.count("*").alias("total_students"))

# # 5. Finding instructors who have taught all BTech CSE core courses
# core_courses = [1, 2, 3]  # Replace with actual core course IDs
# instructors_with_all_courses = instructors_df \
#     .filter(F.size(F.array_intersect(instructors_df.courses_taught, core_courses)) == len(core_courses)) \
#     .select("first_name", "last_name", "email")

# # 6. Finding top-10 courses with the highest enrollments
# top_courses = courses_df \
#     .orderBy(courses_df.enrollment_count.desc()) \
#     .limit(10) \
#     .select("course_name", "enrollment_count")

# Show results for each query
print("Enrolled Students in Course:", course_id)
enrolled_students.show()

# print("Average Enrollment for Instructor:", instructor_id)
# avg_enrollment.show()

# print("Courses in Department:", department_id)
# department_courses.show()

# print("Total Students per Department:")
# total_students_per_dept.show()

# print("Instructors Who Taught All Core Courses:")
# instructors_with_all_courses.show()

# print("Top 10 Courses by Enrollment:")
# top_courses.show()

# Stop the Spark session
spark.stop()
