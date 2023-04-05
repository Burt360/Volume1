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
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(db_file="mystery_database.db"):
    """Use what you've learned about SQL to identify the outlier in the mystery
    database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): outlier's name, outlier's ID number, outlier's eye color, outlier's height
    """
    raise NotImplementedError("Problem 5 Incomplete")
