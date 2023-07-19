/* Welcome to the SQL mini project. You will carry out this project partly in
the PHPMyAdmin interface, and partly in Jupyter via a Python connection.

This is Tier 2 of the case study, which means that there'll be less guidance for you about how to setup
your local SQLite connection in PART 2 of the case study. This will make the case study more challenging for you: 
you might need to do some digging, aand revise the Working with Relational Databases in Python chapter in the previous resource.

Otherwise, the questions in the case study are exactly the same as with Tier 1. 

PART 1: PHPMyAdmin
You will complete questions 1-9 below in the PHPMyAdmin interface. 
Log in by pasting the following URL into your browser, and
using the following Username and Password:

URL: https://sql.springboard.com/
Username: student
Password: learn_sql@springboard

The data you need is in the "country_club" database. This database
contains 3 tables:
    i) the "Bookings" table,
    ii) the "Facilities" table, and
    iii) the "Members" table.

In this case study, you'll be asked a series of questions. You can
solve them using the platform, but for the final deliverable,
paste the code for each solution into this script, and upload it
to your GitHub.

Before starting with the questions, feel free to take your time,
exploring the data, and getting acquainted with the 3 tables. */


/* QUESTIONS 
/* Q1: Some of the facilities charge a fee to members, but some do not.
Write a SQL query to produce a list of the names of the facilities that do. */

/* ANSWER 1
SELECT name, 
FROM `Facilities` 
WHERE `membercost`>0

*******************************
/* Q2: How many facilities do not charge a fee to members? */

/* ANSWER 2
SELECT COUNT(name)
FROM Facilities
WHERE membercost = 0

*******************************
/* Q3: Write an SQL query to show a list of facilities that charge a fee to members,
where the fee is less than 20% of the facility's monthly maintenance cost.
Return the facid, facility name, member cost, and monthly maintenance of the
facilities in question. */

/* ANSWER 3
SELECT facid, name, membercost, monthlymaintenance
FROM Facilities
WHERE membercost > 0
AND
membercost < monthlymaintenance*0.2 

*******************************
/* Q4: Write an SQL query to retrieve the details of facilities with ID 1 and 5.
Try writing the query without using the OR operator. */

/* ANSWER 4
SELECT * 
FROM Facilities
WHERE facid IN (1, 5);

*******************************
/* Q5: Produce a list of facilities, with each labelled as
'cheap' or 'expensive', depending on if their monthly maintenance cost is
more than $100. Return the name and monthly maintenance of the facilities
in question. */

/* ANSWER 5
SELECT name,
    CASE
        WHEN monthlymaintenance > 100 THEN 'expensive'
        ELSE 'cheap'
    END AS cost_label
FROM
    Facilities;
*******************************
/* Q6: You'd like to get the first and last name of the last member(s)
who signed up. Try not to use the LIMIT clause for your solution. */

/* ANSWER 6
SELECT firstname, surname
FROM Members 
WHERE joindate = (SELECT MAX(joindate) FROM Members); 

*******************************
/* Q7: Produce a list of all members who have used a tennis court.
Include in your output the name of the court, and the name of the member
formatted as a single column. Ensure no duplicate data, and order by
the member name. */

/* ANSWER 7
SELECT DISTINCT CONCAT(m.firstname, ' ', m.surname) AS member_name, f.name AS court_name
FROM Members AS m
JOIN Bookings AS b ON m.memid = b.memid
JOIN Facilities AS f ON b.facid = f.facid
WHERE f.name LIKE 'Tennis Court%'
ORDER BY member_name; 

*******************************
/* Q8: Produce a list of bookings on the day of 2012-09-14 which
will cost the member (or guest) more than $30. Remember that guests have
different costs to members (the listed costs are per half-hour 'slot'), and
the guest user's ID is always 0. Include in your output the name of the
facility, the name of the member formatted as a single column, and the cost.
Order by descending cost, and do not use any subqueries. */

/* ANSWER 8
SELECT f.name AS facility_name, CONCAT(m.firstname, ' ', m.surname) AS member_name,
    CASE
        WHEN b.memid = 0 THEN f.guestcost * b.slots
        ELSE f.membercost * b.slots
    END AS cost
FROM Bookings AS b
JOIN Facilities AS f ON b.facid = f.facid
LEFT JOIN Members AS m ON b.memid = m.memid
WHERE b.starttime >= '2012-09-14' AND b.starttime < '2012-09-15'
    AND
    ( (b.memid = 0 AND f.guestcost * b.slots > 30)
        OR
        (b.memid != 0 AND f.membercost * b.slots > 30)
    )
ORDER BY
    cost DESC;

*******************************
/* Q9: This time, produce the same result as in Q8, but using a subquery. */

/* ANSWER 9
SELECT
    facility_name,
    member_name,
    cost
FROM
    (
        SELECT
            f.name AS facility_name,
            CONCAT(m.firstname, ' ', m.surname) AS member_name,
            CASE
                WHEN b.memid = 0 THEN f.guestcost * b.slots
                ELSE f.membercost * b.slots
            END AS cost,
            b.starttime
        FROM
            Bookings AS b
        JOIN
            Facilities AS f ON b.facid = f.facid
        LEFT JOIN
            Members AS m ON b.memid = m.memid
        WHERE
            b.starttime >= '2012-09-14' AND b.starttime < '2012-09-15'
    ) AS subquery
WHERE
    cost > 30
ORDER BY
    cost DESC;


*******************************
/* PART 2: SQLite

Export the country club data from PHPMyAdmin, and connect to a local SQLite instance from Jupyter notebook 
for the following questions.  

QUESTIONS:
/* Q10: Produce a list of facilities with a total revenue less than 1000.
The output of facility name and total revenue, sorted by revenue. Remember
that there's a different cost for guests and members! */

/* ANSWER 10

import sqlite3

# Establish connection
conn = sqlite3.connect('/Users/shsu/Documents/GitHub/dsc/Unit 8 - SQL & Databases/8.3 SQL Case Study - Country Club/SQLFiles Tier 2/sqlite_db_pythonsqlite.db')

# Create a cursor object
cursor = conn.cursor()

# Execute a query
query = """
    SELECT f.name AS facility_name,
           SUM(
               CASE
                   WHEN b.memid = 0 THEN f.guestcost * b.slots
                   ELSE f.membercost * b.slots
               END
           ) AS total_revenue
    FROM Facilities AS f
    JOIN Bookings AS b ON f.facid = b.facid
    GROUP BY f.facid, f.name
    HAVING total_revenue < 1000
    ORDER BY total_revenue
"""
cursor.execute(query)

# Fetch and print the results
results = cursor.fetchall()
for row in results:
    print("Facility: {}, Total Revenue: {}".format(row[0], row[1]))

# Close the cursor and connection
cursor.close()
conn.close()

*******************************
/* Q11: Produce a report of members and who recommended them in alphabetic surname,firstname order */

/* ANSWER 11
conn = sqlite3.connect('/Users/shsu/Documents/GitHub/dsc/Unit 8 - SQL & Databases/8.3 SQL Case Study - Country Club/SQLFiles Tier 2/sqlite_db_pythonsqlite.db')

# Create a cursor object
cursor = conn.cursor()

# Execute the query
query = """
    SELECT
        m1.surname AS member_surname,
        m1.firstname AS member_firstname,
        m2.surname AS recommender_surname,
        m2.firstname AS recommender_firstname
    FROM
        Members AS m1
    LEFT JOIN
        Members AS m2 ON m1.recommendedby = m2.memid
    ORDER BY
        m1.surname, m1.firstname
"""
cursor.execute(query)

# Fetch and print the results
results = cursor.fetchall()
for row in results:
    member_surname, member_firstname, recommender_surname, recommender_firstname = row
    print("Member: {} {}, Recommender: {} {}".format(
        member_firstname, member_surname, recommender_firstname, recommender_surname
    ))

# Close the cursor and connection
cursor.close()
conn.close()

*******************************
/* Q12: Find the facilities with their usage by member, but not guests */

/* ANSWER 12
 Establish connection
conn = sqlite3.connect('/Users/shsu/Documents/GitHub/dsc/Unit 8 - SQL & Databases/8.3 SQL Case Study - Country Club/SQLFiles Tier 2/sqlite_db_pythonsqlite.db')

# Create a cursor object
cursor = conn.cursor()
# Execute the query
query = """
    SELECT
        f.name AS facility_name,
        COUNT(*) AS usage_count
    FROM
        Facilities AS f
    JOIN
        Bookings AS b ON f.facid = b.facid
    WHERE
        b.memid != 0
    GROUP BY
        f.facid, f.name
"""
cursor.execute(query)

# Fetch and print the results
results = cursor.fetchall()
for row in results:
    facility_name, usage_count = row
    print("Facility: {}, Usage Count: {}".format(facility_name, usage_count))

# Close the cursor and connection
cursor.close()
conn.close()

*******************************
/* Q13: Find the facilities usage by month, but not guests */

# Establish connection
conn = sqlite3.connect('/Users/shsu/Documents/GitHub/dsc/Unit 8 - SQL & Databases/8.3 SQL Case Study - Country Club/SQLFiles Tier 2/sqlite_db_pythonsqlite.db')

# Create a cursor object
cursor = conn.cursor()

# Execute the query
query = """
    SELECT
        strftime('%Y-%m', b.starttime) AS usage_month,
        f.name AS facility_name,
        COUNT(*) AS usage_count
    FROM
        Facilities AS f
    JOIN
        Bookings AS b ON f.facid = b.facid
    WHERE
        b.memid != 0
    GROUP BY
        usage_month, f.facid, f.name
"""
cursor.execute(query)

# Fetch and print the results
results = cursor.fetchall()
for row in results:
    usage_month, facility_name, usage_count = row
    print("Month: {}, Facility: {}, Usage Count: {}".format(usage_month, facility_name, usage_count))

# Close the cursor and connection
cursor.close()
conn.close()
/* ANSWER 13



