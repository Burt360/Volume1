from sql1 import *

def test_student_db():
    return
    student_db()

    db_file = 'students.db'
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            # Check schema
            cur.execute("SELECT * FROM MajorInfo;")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM CourseInfo;")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM StudentInfo;")
            print([d[0] for d in cur.description])

            cur.execute("SELECT * FROM StudentGrades;")
            print([d[0] for d in cur.description])
            
            # Check entries
            for row in cur.execute("SELECT * FROM MajorInfo;"):
                print(row)

            for row in cur.execute("SELECT * FROM CourseInfo;"):
                print(row)

            for row in cur.execute("SELECT * FROM StudentInfo;"):
                print(row)

            for row in cur.execute("SELECT * FROM StudentGrades;"):
                print(row)
    
    finally:
        conn.close()

def test_earthquakes_db():
    return
    earthquakes_db()

    db_file = 'earthquakes.db'
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()

            # Check schema
            cur.execute("SELECT * FROM USEarthquakes;")
            print([d[0] for d in cur.description])
            
            # Check entries
            N_ROWS = 25
            i = 0
            for row in cur.execute("SELECT * FROM USEarthquakes;"):
                print(row)
                if i >= N_ROWS:
                    break
                i += 1
            
    finally:
        conn.close()

def test_prob5():
    return
    print(prob5())

def test_prob6():
    # return
    print(prob6())