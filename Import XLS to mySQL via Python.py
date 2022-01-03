import xlrd
import mysql.connector
# Establishing  a MySQL Connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",

)

book= xlrd.open_workbook("zee.xls")
sheet= book.sheet_by_name("Sheet1")

con=mydb.cursor()

con.execute("SHOW DATABASES")
for db in con:
    print(db)


import mysql.connector
# Establishing  a MySQL Connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",
    db='MysqlPython'

)

con=mydb.cursor()
con.execute("SHOW TABLES")
for tb in con:
    print(tb)


#Get the cursor which execute line by line
# Creating Databases
#cursor = mydb.cursor()
## creating a databse called 'datacamp'
#cursor.execute("CREATE DATABASE datacamp")
#cursor= database.cursor()
con.execute("CREATE TABLE B (Table_Name VARCHAR(50), State_Code INTEGER,District_Code INTEGER,Tehsil_Code INTEGER,Town_Code INTEGER)")

con.execute("SHOW TABLES")
for tb in con:
    print(tb)
# Create the INSERT INTO sql query

query= """INSERT INTO B (Table_Name, State_Code,District_Code,Tehsil_Code,Town_Code) VALUES(?,?,?,?,?)"""


## creating a for loop to iterate through each row in xls files

for r in range(7,sheet.nrows):
    Table_Name = sheet.cell(r, 0).value
    State_Code = sheet.cell(r, 1).value
    District_Code = sheet.cell(r, 2).value
    Tehsil_Code = sheet.cell(r, 3).value
    Town_Code = sheet.cell(r, 4).value

# Assigning values from each row

    values= (Table_Name, State_Code,District_Code,Tehsil_Code,Town_Code)

# Execute sql Query

    con.execute(query, values)

# commit the transaction
con.commit()


# close the cursor
con.close()


# close the database connection
database.close()



###--------MySQL connection Tutorial by DataCamp--------

import mysql.connector as mysql

## connecting to the database using 'connect()' method
## it takes 3 required parameters 'host', 'user', 'passwd'
db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "root"
)
print(db)
cursor = db.cursor()

# Creating Databases
## creating a databse called 'datacamp'
cursor.execute("CREATE DATABASE datacamp")

## TO SEE CREATED DATABASE
cursor.execute("SHOW DATABASES")


#Creating Table called 'users' in the 'datacamp' database
cursor.execute("CREATE TABLE users1 (name VARCHAR(255), user_name VARCHAR(255))")

# getting all the tables which are present in 'datacamp' database

cursor.execute("SHOW TABLES")

#Inserting Data into TABLE called user
# defining the Query
query = "INSERT INTO users (name, user_name) VALUES (%s, %s)"

## storing values in a variable
query = "INSERT INTO users (name, user_name) VALUES (%s, %s)"
## storing values in a variable
values = [
    ("Peter", "peter"),
    ("Amy", "amy"),
    ("Michael", "michael"),
    ("Hennah", "hennah")
]

## executing the query with values
cursor.execute(query, values)
## to make final output we have to run the 'commit()' method of the database object
db.commit()

####------------

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",

)
mycursor = mydb.cursor()

mycursor.execute("CREATE DATABASE mymymy")

mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255))")

sql = "INSERT INTO customers (name, address) VALUES (%s, %s)"
val = ("John", "Highway 21")
mycursor.execute(sql, val)
mydb.commit()

print(mycursor.rowcount, "record inserted.")


#------MySql with Python 1st create database -----

import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="root",

)

mycursor=mydb.cursor()
mycursor.execute("CREATE DATABASE  gudia1")

#---Now creating table under Database-----------

import mysql.connector
mydb= mysql.connector.connect(
    host='localhost', user='root',passwd='root', database='gudia1')

mycursor=mydb.cursor()
mycursor.execute('CREATE TABLE students(name VARCHAR(255), age INTEGER(10))')


### NOW INSERTING DATA into created table under dabase---

sqlFormula ="INSERT INTO students(name, age) VALUES(%s,%s)"
students=[("BOB",12),("AMAN",32),("kab",2222)]

mycursor.executemany(sqlFormula,students)

mycursor.close()
mydb.commit()
mydb.close()


###---------------------------------------

#------MySql with Python 1st create database -----
import xlrd
import mysql.connector as mysql

database = mysql.connect(
  host="localhost",
  user="root",
  passwd="root",
  db= "MysqlPython"
)

cursor=database.cursor()

book= xlrd.open_workbook("zee.xls")
sheet= book.sheet_by_name("Sheet1")


### NOW INSERTING DATA into created table under dabase---

query ="INSERT INTO T1 (Table_Name, State_Code,District_Code,Tehsil_Code,Town_Code) VALUES(%s,%s,%s,%s,%s)"

for r in range(1,sheet.nrows):

    Table_Name = sheet.cell(r, 0).value
    State_Code = sheet.cell(r, 1).value
    District_Code = sheet.cell(r, 2).value
    Tehsil_Code = sheet.cell(r, 3).value
    Town_Code = sheet.cell(r, 4).value



# Assigning values from each row

    values= (Table_Name, State_Code,District_Code,Tehsil_Code,Town_Code)

    cursor.execute(query,values)

cursor.close()
database.commit()
database.close()






