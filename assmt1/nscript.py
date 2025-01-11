# import psycopg2
# import json
# from pymongo import MongoClient

# ########################################################################    
# """ Extracting Data from PostgreSQL """
# ########################################################################


# # Database connection parameters
# pg_conn_params = {
#     'dbname': 'university_db',
#     'user': 'postgres',
#     'password': 'root',
#     'host': 'localhost',
#     'port': 5432
# }

# def extract_data():
#     print("at extract data")
#     try:
#         # Connect to PostgreSQL
#         conn = psycopg2.connect(**pg_conn_params)
#         cursor = conn.cursor()

#         # Query to extract data from each table
#         tables = {
#             'students': 'SELECT * FROM students',
#             'courses': 'SELECT * FROM courses',
#             'enrollments': 'SELECT * FROM enrollments',
#             'instructors': 'SELECT * FROM instructors',
#             'departments': 'SELECT * FROM departments'
#         }

#         data = {}
#         for table, query in tables.items():
#             cursor.execute(query)
#             data[table] = cursor.fetchall()

#         # Close the connection
#         cursor.close()
#         conn.close()

#         return data

#     except Exception as e:
#         print(f"Error extracting data: {e}")
#         return None
    
    
    
    
# ########################################################################    
# """Transforming Data"""
# ########################################################################

# def transform_data(data):
#     print("at transform data")
#     # Transform data into MongoDB format
#     students = []
#     courses = []
#     instructors = []
#     departments = []

#     # Transform Students
#     for student in data['students']:
#         students.append({
#             '_id': student[0],  # student_id
#             'first_name': student[1],
#             'last_name': student[2],
#             'email': student[3],
#             'department_id': student[4],
#             'enrollments': []  # To be filled later
#         })

#     # Transform Courses
#     for course in data['courses']:
#         courses.append({
#             '_id': course[0],  # course_id
#             'course_name': course[1],
#             'department_id': course[2],
#             'credits': course[3],
#             'enrollment_count': 0,  # Will be updated during load
#             'instructors': []  # To be filled later
#         })

#     # Transform Instructors
#     for instructor in data['instructors']:
#         instructors.append({
#             '_id': instructor[0],  # instructor_id
#             'first_name': instructor[1],
#             'last_name': instructor[2],
#             'email': instructor[3],
#             'department_id': instructor[4],
#             'courses_taught': []  # To be filled later
#         })

#     # Transform Departments
#     for department in data['departments']:
#         departments.append({
#             '_id': department[0],  # department_id
#             'department_name': department[1],
#             'students': [],  # Will be updated during load
#             'courses': []    # Will be updated during load
#         })

#     return {
#         'students': students,
#         'courses': courses,
#         'instructors': instructors,
#         'departments': departments
#     }







# ########################################################################    
# """  Loading Data into MongoDB """
# ########################################################################


# # MongoDB connection parameters
# mongo_conn_params = {
#     'host': 'localhost',
#     'port': 27017,
#     'db_name': 'university_information_system'
# }

# def load_data(transformed_data, extracted_data):
#     print("at load data")
#     try:
#         # Connect to MongoDB
#         client = MongoClient(mongo_conn_params['host'], mongo_conn_params['port'])
#         db = client[mongo_conn_params['db_name']]

#         # Insert Departments
#         db.departments.insert_many(transformed_data['departments'])

#         # Insert Instructors
#         db.instructors.insert_many(transformed_data['instructors'])

#         # Insert Courses
#         courses_result = db.courses.insert_many(transformed_data['courses'])
        
#         # Update Courses' enrollment count and instructors
#         for course_id in courses_result.inserted_ids:
#             # Get enrollments from PostgreSQL
#             for enrollment in extracted_data['enrollments']:
#                 if enrollment[1] == course_id:  # Match course_id
#                     db.courses.update_one({'_id': course_id}, {'$inc': {'enrollment_count': 1}})
#                     # Update students and instructors accordingly
#                     student_id = enrollment[0]
#                     db.students.update_one({'_id': student_id}, {'$push': {'enrollments': {'course_id': course_id, 'course_name': enrollment[2], 'grade': enrollment[3]}}})

#                     # Update instructors
#                     instructor_id = enrollment[4]  # Assuming enrollment includes instructor_id
#                     db.instructors.update_one({'_id': instructor_id}, {'$addToSet': {'courses_taught': course_id}})

#         # Insert Students
#         db.students.insert_many(transformed_data['students'])

#         print("Data migration completed successfully.")
        
#     except Exception as e:
#         print(f"Error loading data: {e}")




# ########################################################################    
# """  main function """
# ########################################################################

# if __name__ == '__main__':
#     extracted_data = extract_data()
#     if extracted_data:
#         transformed_data = transform_data(extracted_data)
#         load_data(transformed_data, extracted_data)



import psycopg2
from pymongo import MongoClient

########################################################################    
""" Extracting Data from PostgreSQL """
########################################################################

# Database connection parameters
pg_conn_params = {
    'dbname': 'university_db',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
    'port': 5432
}

def extract_data():
    print("at extract data")
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**pg_conn_params)
        cursor = conn.cursor()

        # Query to extract data from each table
        tables = {
            'students': 'SELECT * FROM students',
            'courses': 'SELECT * FROM courses',
            'instructors': 'SELECT * FROM instructors',
            'departments': 'SELECT * FROM departments'
        }

        data = {}
        for table, query in tables.items():
            cursor.execute(query)
            data[table] = cursor.fetchall()

        # Extract enrollments with additional information
        cursor.execute('''
            SELECT 
                e.student_id,
                e.course_id,
                e.grade,
                c.department_id  -- To link courses with departments
            FROM enrollments e
            JOIN courses c ON e.course_id = c.course_id
        ''')
        data['enrollments'] = cursor.fetchall()

        # Extract instructor-course relationships
        cursor.execute('SELECT instructor_id, course_id FROM course_instructors')
        data['course_instructors'] = cursor.fetchall()

        # Close the connection
        cursor.close()
        conn.close()

        return data

    except Exception as e:
        print(f"Error extracting data: {e}")
        return None

########################################################################    
"""Transforming Data"""
########################################################################

def transform_data(data):
    print("at transform data")
    # Transform data into MongoDB format
    students = []
    courses = []
    instructors = []
    departments = []

    # Transform Students
    for student in data['students']:
        students.append({
            '_id': student[0],  # student_id
            'first_name': student[1],
            'last_name': student[2],
            'email': student[3],
            'department_id': student[4],
            'enrollments': [  # Directly filled with enrollments
                {'course_id': enrollment[1], 'grade': enrollment[2]} 
                for enrollment in data['enrollments'] if enrollment[0] == student[0]
            ]
        })

    # Transform Courses
    for course in data['courses']:
        courses.append({
            '_id': course[0],  # course_id
            'course_name': course[1],
            'department_id': course[2],
            'credits': course[3],
            'enrollment_count': sum(1 for enrollment in data['enrollments'] if enrollment[1] == course[0]),  # Count of enrollments
            'instructors': [  # Directly filled with instructors
                instructor_id for instructor_id, course_id in data['course_instructors'] if course_id == course[0]
            ]
        })

    # Transform Instructors
    for instructor in data['instructors']:
        instructors.append({
            '_id': instructor[0],  # instructor_id
            'first_name': instructor[1],
            'last_name': instructor[2],
            'email': instructor[3],
            'department_id': instructor[4],
            'courses_taught': [  # Directly filled with courses
                course_id for instructor_id, course_id in data['course_instructors'] if instructor_id == instructor[0]
            ]
        })

    # Transform Departments
    for department in data['departments']:
        departments.append({
            '_id': department[0],  # department_id
            'department_name': department[1],
            'students': [  # Directly filled with student IDs
                student['_id'] for student in students if student['department_id'] == department[0]
            ],
            'courses': [  # Directly filled with course IDs
                course['_id'] for course in courses if course['department_id'] == department[0]
            ]
        })

    return {
        'students': students,
        'courses': courses,
        'instructors': instructors,
        'departments': departments
    }

########################################################################    
""" Loading Data into MongoDB """
########################################################################

# MongoDB connection parameters
mongo_conn_params = {
    'host': 'localhost',
    'port': 27017,
    'db_name': 'university_information_system'
}

def load_data(transformed_data):
    print("Loading data into MongoDB...")
    try:
        # Connect to MongoDB
        client = MongoClient(mongo_conn_params['host'], mongo_conn_params['port'])
        db = client[mongo_conn_params['db_name']]

        # Insert Departments
        db.departments.insert_many(transformed_data['departments'])

        # Insert Instructors
        db.instructors.insert_many(transformed_data['instructors'])

        # Insert Courses
        db.courses.insert_many(transformed_data['courses'])

        # Insert Students
        db.students.insert_many(transformed_data['students'])

        print("Data loading completed successfully.")

    except Exception as e:
        print(f"Error loading data: {e}")

########################################################################    
"""  main function """
########################################################################

if __name__ == '__main__':
    extracted_data = extract_data()
    if extracted_data:
        transformed_data = transform_data(extracted_data)
        load_data(transformed_data)
