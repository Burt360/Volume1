# solutions.py
"""Volume 1: SQL 2.
Nathan Schill
Section 2
Tues. Apr. 11, 2023
"""

import sqlite3 as sql

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            # Execute query
            cur.execute("SELECT SI.StudentName "
                        "FROM StudentInfo AS SI INNER JOIN StudentGrades as SG "
                        "ON SI.StudentID == SG.StudentID "
                        "WHERE SG.Grade=='B'")
            
            # Get just name strings, not tuples of name strings
            return [tup[0] for tup in cur.fetchall()]
    
    finally:
        conn.close()


# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            
            # Execute query
            cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade FROM "
                        "StudentInfo as SI INNER JOIN "
                        "StudentGrades as SG, "
                        "CourseInfo as CI ON "
                        "SI.StudentID==SG.StudentID AND "
                        "SG.CourseID==CI.CourseID LEFT OUTER JOIN "
                        "MajorInfo as MI ON "
                        "SI.MajorId==MI.MajorID WHERE "
                        "CI.CourseName=='Calculus'")

            return cur.fetchall()
    
    finally:
        conn.close()


# Problem 3
def prob3(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            
            # Execute query
            cur.execute("SELECT MI.MajorName, COUNT(*) as num_students "
                        "FROM "
                        "StudentInfo AS SI LEFT OUTER JOIN MajorInfo as MI ON "
                        "SI.MajorID==MI.MajorID "
                        "GROUP BY "
                        "SI.MajorID "
                        "ORDER BY num_students DESC, MI.MajorName ASC")

            return cur.fetchall()
    
    finally:
        conn.close()


# Problem 4
def prob4(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            # Execute query
            cur.execute("SELECT SI.StudentName, COUNT(SG.points), "
                        "AVG(SG.points) AS gpa FROM ("
                            "SELECT StudentID, CASE Grade "
                                "WHEN 'A+' THEN 4.0 "
                                "WHEN 'A' THEN 4.0 "
                                "WHEN 'A-' THEN 3.7 "
                                "WHEN 'B+' THEN 3.4 "
                                "WHEN 'B' THEN 3.0 "
                                "WHEN 'B-' THEN 2.7 "
                                "WHEN 'C+' THEN 2.4 "
                                "WHEN 'C' THEN 2.0 "
                                "WHEN 'C-' THEN 1.7 "
                                "WHEN 'D+' THEN 1.4 "
                                "WHEN 'D' THEN 1.0 "
                                "WHEN 'D-' THEN 0.7 "
                                "ELSE 0 END AS points "
                            "FROM StudentGrades) AS SG "
                        "INNER JOIN StudentInfo AS SI "
                        "ON SG.StudentId==SI.StudentID "
                        "GROUP BY SI.StudentName "
                        "ORDER BY gpa DESC")

            return cur.fetchall()
    
    finally:
        conn.close()


# Problem 5
def prob5(db_file="mystery_database.db"):
    """Use what you've learned about SQL to identify the outlier in the mystery
    database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): outlier's name, outlier's ID number, outlier's eye color, outlier's height
    """
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            print()
            # Check column names
            cur.execute("SELECT * FROM table_1")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM table_2")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM table_3")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM table_4")
            print([d[0] for d in cur.description])

            # Clear
            cur.close()
            cur = conn.cursor()

            # Check homeworld
            cur.execute("SELECT home_world, COUNT(home_world) as hw FROM "
                        "table_4 "
                        "GROUP BY home_world "
                        "HAVING hw==1")
            # print(*[result[0] for result in cur.fetchall()], sep='\n')
            # Outlier: "Earth"

            # Check Earth's character's species
            cur.execute("SELECT species, species_2nd, species_3rd FROM "
                        "table_4 "
                        "WHERE home_world=='Earth'")
            # print(*[result for result in cur.fetchall()], sep='\n')

            # Check description
            cur.execute("SELECT ID_number, description FROM "
                        "table_2 "
                        "WHERE description LIKE '%human%Earth%'")
            # print(*[result for result in cur.fetchall()], sep='\n')
            # ID number: 830744
            # (Name: "William Thomas 'Will' Riker")

            # Check name, eye color
            cur.execute("SELECT name, eye_color FROM "
                        "table_1 "
                        "WHERE name LIKE '%Will%'")
            print(*[result for result in cur.fetchall()], sep='\n')
            # Name: William T. Riker
            # Eye color: Hazel-blue

            cur.execute("SELECT height FROM "
                        "table_3 "
                        "WHERE eye_color=='Hazel-blue'")
            # print(*[result for result in cur.fetchall()], sep='\n')
            # Height: 1.93

            return ['William T. Riker', 830744, 'Hazel-blue', 1.93]
            
    
    finally:
        conn.close()

""""
# Check gender
cur.execute("SELECT gender, COUNT(gender) as g FROM "
            "table_3 "
            "GROUP BY gender ")
# print(*[result for result in cur.fetchall()], sep='\n')

# Check species
cur.execute("SELECT species, COUNT(species) as s FROM "
            "table_4 "
            "GROUP BY species "
            "HAVING s==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')
# Outlier: "Sentient species"

# Check species_2nd
cur.execute("SELECT species_2nd, COUNT(species_2nd) as s FROM "
            "table_4 "
            "GROUP BY species_2nd "
            "HAVING s==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')

# Check species_3rd
cur.execute("SELECT species_3rd, COUNT(species_3rd) as s FROM "
            "table_4 "
            "GROUP BY species_3rd")
# print(*[result for result in cur.fetchall()], sep='\n')

# Check skin color
cur.execute("SELECT skin_color, COUNT(skin_color) as s FROM "
            "table_3 "
            "GROUP BY skin_color "
            "HAVING s==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')

# Check description
cur.execute("SELECT description, COUNT(description) as d FROM "
            "table_2 "
            "GROUP BY description "
            "HAVING d==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')

# Check height
cur.execute("SELECT height, COUNT(height) as h FROM "
            "table_3 "
            "GROUP BY height "
            "HAVING h==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')

# Check weight
cur.execute("SELECT weight, COUNT(weight) as w FROM "
            "table_3 "
            "GROUP BY weight "
            "HAVING w==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')
# Outlier: "Naturally black"

# Check hair color
cur.execute("SELECT hair_color, COUNT(hair_color) as h FROM "
            "table_3 "
            "GROUP BY hair_color "
            "HAVING h==1")
# print(*[result[0] for result in cur.fetchall()], sep='\n')
# Naturally black

# return cur.fetchall()
"""