
--Rename DATABASE 
ALTER DATABASE haider MODIFY NAME = GKS;  

EXEC sp_renamedb'haider' , 'GKS';  

show DATABASES;
use edureka;
show tables;

-- # Creating an emplyee info table
CREATE TABLE emplyeeInfo(
    ID int NOT NULL AUTO_INCREMENT Primary Key,
    Emp_name varchar(255) NOT NULL,
    Address varchar(255) NOT NULL,
    City varchar(255) NOT NULL,
    Age int NOT NULL,
    DOJ date NOT NULL,
    Designation varchar(255) NOT NULL,
    Salary decimal(15,2) NOT NULL,
    Mobile int NOT NULL
);

-- # to see the table structure 

select * from emplyeeInfo;

-- # To delete particular row from the table

DELETE FROM emplyeeInfo WHERE Emp_name='Sunil kumar'; 

-- #T0 DELETE THE DABLE

DROP TABLE employeeInfo;

-- # tO ADD NEW COLUMN 

ALTER TABLE emplyeeinfo ADD Email varchar(255);

-- # To delete COLUMN

Alter TABLE emplyeeinfo DROP COLUMN Email;

-- # How to add/insert data into TABLE

INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Sunil Kumar','104, Street No.13', 'Jaipur', 29, '2020-05-03', 'Manager', 42000.0, 822200);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Manoj Sing','72, Street No.1', 'Jaipur', 30, '2020-05-08', 'Quality Analyst', 30000.0, 822201);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Anil Kumar','House No.53, Street No.7', 'Udaipur', 32, '2021-08-03','Data Analyst', 25000.0, 22202);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Kamal','P76, Block No5', 'Japur', 36, '2019-07-03', 'Sr programmer', 48500.0, 22203);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Daanish','104, Street No.13', 'Ajmer', 37, '2020-02-25', 'Sr Programmer', 49000.0, 22204);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Akhtar','Flat No.6 Jamia Nagar', 'Delhi', 38, '2021-04-03', 'Programmer', 38000.0, 22205);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Suhail','DLF City Gate No.5', 'Delhi', 25, '2022-01-05', 'Data Analyst', 42000.0, 22206);
INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES ('Zakir','Kalindi kunj, Metro Station', 'Delhi', 28, '2022-01-03', 'Forecast Analyst', 45000.0,22207);

-- Or you could directly insert all values at once as follows


INSERT INTO edureka.emplyeeinfo(Emp_name, Address, City, Age, DOJ, Designation, Salary, Mobile)
VALUES (('Sunil Kumar','104, Street No.13', 'Jaipur', 29, '2020-05-03', 'Manager', 42000.0, 822200),
 ('Manoj Sing','72, Street No.1', 'Jaipur', 30, '2020-05-08', 'Quality Analyst', 30000.0, 822201),
 ('Anil Kumar','House No.53, Street No.7', 'Udaipur', 32, '2021-08-03','Data Analyst', 25000.0, 22202),
 ('Kamal','P76, Block No5', 'Japur', 36, '2019-07-03', 'Sr programmer', 48500.0, 22203),
 ('Daanish','104, Street No.13', 'Ajmer', 37, '2020-02-25', 'Sr Programmer', 49000.0, 22204),
 ('Akhtar','Flat No.6 Jamia Nagar', 'Delhi', 38, '2021-04-03', 'Programmer', 38000.0, 22205),
 ('Suhail','DLF City Gate No.5', 'Delhi', 25, '2022-01-05', 'Data Analyst', 42000.0, 22206),
 ('Zakir','Kalindi kunj, Metro Station', 'Delhi', 28, '2022-01-03', 'Forecast Analyst', 45000.0,22207)
);

--To view only particular column from the table

SELECT Emp_name, Designation  FROM emplyeeinfo;

-- Where clause
SELECT Emp_name, City  FROM emplyeeinfo WHERE  Designation='Data Analyst';

SELECT Emp_name, Salary,City, Designation FROM emplyeeinfo 
WHERE  Salary>45000;

SELECT ID, Emp_name, Salary,City, Designation FROM emplyeeinfo 
WHERE  Designation='Data Analyst' AND Salary>40000;

SELECT ID, Emp_name, Age,City FROM emplyeeinfo 
WHERE  Designation='Data Analyst' AND Age>20;

SELECT ID, Emp_name,City FROM emplyeeinfo 
WHERE Not City='Delhi';



-- SQL update: to update table values as follows
UPDATE emplyeeinfo SET Address='DLF City Noida' WHERE ID=1;
UPDATE emplyeeinfo SET Emp_name='Zeeshan' WHERE ID=1;

UPDATE emplyeeinfo SET Mobile=995679378 WHERE ID=1;



--To delete a particular record based on condition

DELETE FROM emplyeeinfo where Id=2;

--The SQL Like Operator
--Note: It is used in a "WHERE" caluse to search for a specified pattern in a column.

SELECT*FROM emplyeeinfo 
WHERE  City Like 'J%';       -- Here % in the last then it retrives a value start with Z

SELECT*FROM emplyeeinfo 
WHERE  City Like 'D%';

SELECT*FROM emplyeeinfo 
WHERE  City not Like 'D%';


SELECT*FROM emplyeeinfo 
WHERE  Emp_name LIKE  'z%';

SELECT*FROM emplyeeinfo 
WHERE  City LIKE  '%r';    --It retrives values whose name end with 'r'
SELECT*FROM emplyeeinfo 
WHERE  City not LIKE '%r';    --It retrives values whose name not end with 'r'


SELECT*FROM emplyeeinfo 
WHERE  Mobile LIKE  '___6%';   --- It will retrieve the values start with '6' but at 3rd position from the table 

------ SQL IN operator:-----------

SELECT*FROM emplyeeinfo 
WHERE  City IN ('Delhi','Ajmer');

SELECT*FROM emplyeeinfo 
WHERE  City Not IN ('Delhi','Ajmer');

---The SQL Between operator:-----------
SELECT*FROM emplyeeinfo 
WHERE  Salary Between 20000 AND 35000;

SELECT*FROM emplyeeinfo 
WHERE  ID Between 3 AND 5;

SELECT*FROM emplyeeinfo 
WHERE  DOJ Between '2021-04-01' AND '2022-02-05';

--The SQL TOP/Limit operator:--------------------
SELECT*FROM emplyeeinfo 
LIMIT 3;

-- SQL ORDER BY caluse:--------

SELECT*FROM emplyeeinfo 
ORDER BY Emp_name, Salary;

SELECT*FROM emplyeeinfo 
ORDER BY Emp_name DESC;

--SQL Functions:------
--Count the total Employees
SELECT Count(Emp_name) FROM emplyeeinfo;
 
--Adding the salary

SELECT SUM(Salary) FROM emplyeeinfo;

--Average salary:----------------
SELECT AVG(Salary) FROM emplyeeinfo;
--Min and MAX Example salary:----------------
SELECT Emp_name, MIN(Salary) AS LowestSalary FROM emplyeeinfo;

SELECT Emp_name, MAX(Salary) AS MaxSalary FROM emplyeeinfo;

--The GroupBy statement:


Select  COUNT(Designation), SUM(Salary), Designation
From emplyeeinfo
Group By Designation;

--- SQL Insert into select statement --------------------------------
--NOTE: It is used to copy tables data into a new table 

--first creating a new table with the same structure as the
CREATE TABLE emplyeeInfoBackup(
    ID int NOT NULL AUTO_INCREMENT Primary Key,
    Emp_name varchar(255) NOT NULL,
    Address varchar(255) NOT NULL,
    City varchar(255) NOT NULL,
    Age int NOT NULL,
    DOJ date NOT NULL,
    Designation varchar(255) NOT NULL,
    Salary decimal(15,2) NOT NULL,
    Mobile int NOT NULL
);

INSERT INTO emplyeeinfoBackup 
SELECT*FROM emplyeeinfo;

--SQL TRUNCATE TABLE commad:--------
-- NOTE: If you want to delete only table data not table then truncate is used.
TRUNCATE TABLE emplyeeinfobackup;

--SQL Constraint:-------


-- SQL Joins:--------------
----Creating a new database and tables Customers and Orders
Create database companydb;
use companydb;
show tables;

CREATE TABLE Customer (
    Customer_id INT NOT NULL PRIMARY KEY,
    Customer_name VARCHAR(20) NOT NULL,
    City VARCHAR(20) NOT NULL
);

Select * FROM customer;

INSERT INTO customer VALUES(101,'Ashis','Kota');
INSERT INTO customer VALUES(102,'Ajay','Ajmer');
INSERT INTO customer VALUES(103,'Ashis','Delhi');
INSERT INTO customer VALUES(104,'Aman','Jaisalmer');
INSERT INTO customer VALUES(105,'Dinesh','Udaipur');
INSERT INTO customer VALUES(106,'Rakesh','Jaipur');
INSERT INTO customer VALUES(107,'Zakir','Jaipur');
INSERT INTO customer VALUES(108,'Husain','Kota');
INSERT INTO customer VALUES(109,'Firoz','Jodhpur');

---Or you could directly enter the values into table as follows

INSERT INTO companydb.customer (Customer_id,Customer_name, City) 
VALUES ((101,'Ashis','Kota'),
    (102,'Ajay','Ajmer'),
    (103,'Ashis','Delhi'),
    (104,'Aman','Jaisalmer'),
    (105,'Dinesh','Udaipur'),
    (106,'Rakesh','Jaipur'),
    (107,'Zakir','Jaipur'),
    (108,'Husain','Kota'),
    (109,'Firoz','Jodhpur'));

--------------------
Create table Orders(
    OrderId int,
    Customer_id int primary key,
    OrderDate date
);

Insert into orders values(1,101,'2021-01-01');
Insert into orders values(2,102,'2021-02-02');
Insert into orders values(3,103,'2021-03-03');
Insert into orders values(4,104,'2021-01-04');
Insert into orders values(5,105,'2021-01-05');
Insert into orders values(6,106,'2021-02-05');
Insert into orders values(7,107,'2021-01-06');
Insert into orders values(8,108,'2021-08-06');
Insert into orders values(9,109,'2021-07-07');
Insert into orders values(10,110,'2021-07-07');
Insert into orders values(11,111,'2021-07-07');
Insert into orders values(12,112,'2021-07-07');
Insert into orders values(13,113,'2021-07-07');


-----Now INNER Join Query
SELECT orders.OrderId, customer.customer_name, orders.orderDate FROM orders
INNER JOIN customer ON orders.customer_id = customer.customer_id;

SELECT orders.OrderDate, customer.City FROM orders
INNER JOIN customer ON orders.customer_id = customer.customer_id;

---- Some other joins: LEFT, RIGHT, FULL JOIN and SELF JOIN, you can read from GOOGLE



-----https://www.edureka.co/blog/interview-questions/sql-query-interview-questions

--Q1:Write a query to fetch the EmpFname from the EmployeeInfo table in the upper case and use the ALIAS name as EmpName.
--Q2:Write a query to fetch the number of employees working in the department ‘HR’.
--Q3:Write a query to get the current date.
--Q4:Write a query to retrieve the first four characters of  EmpLname from the EmployeeInfo table.
--Q5:Write a query to fetch only the place name(string before brackets) from the Address column of EmployeeInfo table.
--Q6:Write a query to create a new table that consists of data and structure copied from the other table.
--Q7:Write q query to find all the employees whose salary is between 50000 to 100000.
--Q8:Write a query to find the names of employees that begin with ‘S’
--Q9:Write a query to fetch top N records.
--Q10:Write a query to retrieve the EmpFname and EmpLname in a single column as “FullName”. The first name and the last name must be separated with space.
--Q11. Write a query find number of employees whose DOB is between 02/05/1970 to 31/12/1975 and are grouped according to gender
--Q12. Write a query to fetch all the records from the EmployeeInfo table ordered by EmpLname in descending order and Department in the ascending order.
--Q13. Write a query to fetch details of employees whose EmpLname ends with an alphabet ‘A’ and contains five alphabets.
--Q14. Write a query to fetch details of all employees excluding the employees with first names, “Sanjay” and “Sonia” from the EmployeeInfo table.
--Q15. Write a query to fetch details of employees with the address as “DELHI(DEL)”.
--Q16. Write a query to fetch all employees who also hold the managerial position.
--Q17. Write a query to fetch the department-wise count of employees sorted by department’s count in ascending order.
--Q18. Write a query to calculate the even and odd records from a table.
--Q19. Write a SQL query to retrieve employee details from EmployeeInfo table who have a date of joining in the EmployeePosition table.
--Q20. Write a query to retrieve two minimum and maximum salaries from the EmployeePosition table.
--Q21. Write a query to find the Nth highest salary from the table without using TOP/limit keyword.
--Q22. Write a query to retrieve duplicate records from a table.
--Q23. Write a query to retrieve the list of employees working in the same department.
--Q24. Write a query to retrieve the last 3 records from the EmployeeInfo table.
--Q25. Write a query to find the third-highest salary from the EmpPosition table.
--Q26. Write a query to display the first and the last record from the EmployeeInfo table.
--Q27. Write a query to add email validation to your database
--Q28. Write a query to retrieve Departments who have less than 2 employees working in it.
--Q29. Write a query to retrieve EmpPostion along with total salaries paid for each of them.
--Q30. Write a query to fetch 50% records from the EmployeeInfo table.


--\\
show tables;
use edureka;
--Q1:Write a query to fetch the EmpFname from the EmployeeInfo table in the upper case and use the ALIAS name as EmpName.

select UPPER(EmpFname) as EmpName from EmployeeInfo;

--Q2:Write a query to fetch the number of employees working in the department ‘HR’.
select COUNT(Department) As HR_employee from EmployeeInfo where Department='HR'
--or
select COUNT(*) As HR_employee from EmployeeInfo where Department='HR'


--Q3:Write a query to get the current date.

SELECT GETDATE();

--Q4:Write a query to retrieve the first four characters of  EmpLname from the EmployeeInfo table.
SELECT substring(EmpLname,1,4) from EmployeeInfo; 

--Q5:Write a query to fetch only the place name(string before brackets) from the Address column of EmployeeInfo table.
SELECT MID(Address, 0, LOCATE('(',Address)) FROM EmployeeInfo;
SELECT SUBSTRING(Address, 1, CHARINDEX('(',Address)) FROM EmployeeInfo;
--Q6:Write a query to create a new table that consists of data and structure copied from the other table.
CREATE TABLE NewTable AS SELECT * FROM EmployeeInfo;
--or
SELECT * INTO NewTable FROM EmployeeInfo WHERE 1 = 0;
--Q7:Write q query to find all the employees whose salary is between 50000 to 100000.

select empPosition, Salary from employeeposition where salary between 50000 and 100000;

--Q8:Write a query to find the names of employees that begin with ‘S’
select EmpFname from EmployeeInfo where EmpFname Like 'S%';   -- it returns only 
--or
select*from EmployeeInfo where EmpFname Like 'S%';  -- returns whole table with name s only

--Q9:Write a query to fetch top N records.
SELECT TOP N * FROM EmployeePosition ORDER BY Salary DESC;
--OR
SELECT * FROM EmpPosition ORDER BY Salary DESC LIMIT N;
--Q10:Write a query to retrieve the EmpFname and EmpLname in a single column as “FullName”. The first name and the last name must be separated with space.

select concat(EmpFname, ' ', EmpLname) as full_name from employeeinfo;

--Q11. Write a query find number of employees whose DOB is between 02/05/1970 to 31/12/1975 and are grouped according to gender
select count(*), Gender from EmployeeInfo where DOB between '02-05-1970' and '31-12-1977';
select count(*), empFname,Gender from EmployeeInfo where DOB between '02-05-1970' and '31-12-1977';

--Q12. Write a query to fetch all the records from the EmployeeInfo table ordered by EmpLname in descending order and Department in the ascending order.

select * from EmployeeInfo order by empLname desc, Department asc;

--Q13. Write a query to fetch details of employees whose EmpLname ends with an alphabet ‘A’ and contains five alphabets.
select * from EmployeeInfo where empLname like '____A%';

--Q14. Write a query to fetch details of all employees excluding the employees with first names, “Sanjay” and “Sonia” from the EmployeeInfo table.

select * from EmployeeInfo where empFname not in ('Sanjay', 'Sonia');

--Q15. Write a query to fetch details of employees with the address as “DELHI(DEL)”.
select * from EmployeeInfo where Address like 'DELHI(DEL)';
--Q16. Write a query to fetch all employees who also hold the managerial position.
select * from employeeposition where EmpPosition like 'Manager';
--or
SELECT E.EmpFname, E.EmpLname, P.EmpPosition 
FROM EmployeeInfo E INNER JOIN EmployeePosition P ON
E.EmpID = P.EmpID AND P.EmpPosition IN ('Manager');

--Q17. Write a query to fetch the department-wise count of employees sorted by department’s count in ascending order.

SELECT Department, count(EmpID) AS EmpDeptCount 
FROM EmployeeInfo GROUP BY Department 
ORDER BY EmpDeptCount ASC;

--Q18. Write a query to calculate the even and odd records from a table.

SELECT EmpID FROM (SELECT rowno, EmpID from EmployeeInfo) WHERE MOD(rowno,2)=0;

--or
SELECT EmpID FROM (SELECT rowno, EmpID from EmployeeInfo) WHERE MOD(rowno,2)=1;

--Q19. Write a SQL query to retrieve employee details from EmployeeInfo table who have a date of joining in the EmployeePosition table.

SELECT * FROM EmployeeInfo E 
WHERE EXISTS 
(SELECT * FROM EmployeePosition P WHERE E.EmpId = P.EmpId);

--Q20. Write a query to retrieve two minimum and maximum salaries from the EmployeePosition table.
SELECT DISTINCT Salary FROM EmployeePosition E1 
WHERE 2 >= (SELECTCOUNT(DISTINCT Salary)FROM EmployeePosition E2 
WHERE E1.Salary >= E2.Salary) ORDER BY E1.Salary DESC;

SELECT DISTINCT Salary FROM EmployeePosition E1 
WHERE 2 >= (SELECTCOUNT(DISTINCT Salary) FROM EmployeePosition E2 
WHERE E1.Salary <= E2.Salary) ORDER BY E1.Salary DESC;

--Q21. Write a query to find the Nth highest salary from the table without using TOP/limit keyword.
SELECT Salary 
FROM EmployeePosition E1 
WHERE N-1 = ( 
      SELECT COUNT( DISTINCT ( E2.Salary ) ) 
      FROM EmployeePosition E2 
      WHERE E2.Salary >  E1.Salary );

--Q22. Write a query to retrieve duplicate records from a table.

SELECT EmpID, EmpFname, Department COUNT(*) 
FROM EmployeeInfo GROUP BY EmpID, EmpFname, Department 
HAVING COUNT(*) > 1;

--Q23. Write a query to retrieve the list of employees working in the same department.
--Q24. Write a query to retrieve the last 3 records from the EmployeeInfo table.
--Q25. Write a query to find the third-highest salary from the EmpPosition table.
--Q26. Write a query to display the first and the last record from the EmployeeInfo table.
--Q27. Write a query to add email validation to your database
--Q28. Write a query to retrieve Departments who have less than 2 employees working in it.
--Q29. Write a query to retrieve EmpPostion along with total salaries paid for each of them.
--Q30. Write a query to fetch 50% records from the EmployeeInfo table.



-------------------\\--\\-----\\------\\--------\\----\\----\\----\\-----\\----\\----\\----
--Link:https://www.youtube.com/watch?v=sa6_kqlglLs

show tables;

select * from zeetestsql;

--1.SQL Query to update Date of Joining to 15-jul-2012 for empid =1.

update zeetestsql set dateofjoining='15-jul-2012' where empid=1; 

--2.SQL Query to select all student name where age is greater than 22.


--3.SQL Query to Find all employee with Salary between 40000 and 80000.
select FIRSTNAME, SALARY from zeetestsql  where SALARY between '40,000' and '80,000';

select*from zeetestsql  where SALARY between '40,000' and '80,000';


select FIRSTNAME, SALARY, count(*) from zeetestsql  where SALARY between '40,000' and '80,000' Group by FIRSTNAME;

--4.SQL Query to display full name.
Select concat(FIRSTNAME,LASTNAME) as fullname from zeetestsql;
Select FIRSTNAME,LASTNAME from zeetestsql;

--5.SQL Query to find name of employee beginning with S.
select * from zeetestsql where FIRSTNAME LIKE 'S%';

--6.Write a query to fetch details of employees whose firstname ends with an alphabet 'A' and contains exactly five alphabets.
select * from zeetestsql where FIRSTNAME LIKE '____A';
--7.	Write a query to fetch details of all employees excluding the employees with first names, "ANUSHKA' and "SOMNATH" from the Employee table.
select * from zeetestsql where FIRSTNAME != 'ANUSHKA' and 'SOMNATH';
select * from zeetestsql where FIRSTNAME NOT IN ('ANUSHKA','SOMNATH');

--8.	SQL query to display the current date?
SELECT SYSDATE, current_date, SYSTIMESTAMP, CURRENT_TIMESTAMP FROM DUAL;

--9.	SQL Query to get day of last day of the previous month?

SELECT LAST_DAY(ADD_MONTHS(SYSDATE,-1) from DUAL) from DUAL;

--10. Write an SQL query to fetch the employee FIRST names and replace the A with '@'
select replace(FIRSTNAME,'A','@') from zeetestsql;
--11.	Write an SQL query to fetch the domain from an email address

--(NOTE:)select INSTR(EMAILID, '@') from zeetestsql;

select substring(EMAILID, INSTR(EMAILID, '@')+1) from zeetestsql;

--12.	Write an SQL query to update the employee names by removing leading and trailing spaces.

update zeetestsql set FIRSTNAME= LTRIM(RTRIM(FIRSTNAME));
--13.	Write an ►SQL query to fetch all the Employees details from Employee table who joined in the Year 2020
select * from zeetestsql where DATEOFJOINING between '1-JAN-2020' and '31-DEC-2020';

select * from zeetestsql where TO_CHAR(DATEOFJOINING,'YYYY')='2020';
--14.	Write an SQL query to fetch only odd rows/Even rows from the table.
select * from zeetestsql where MOD(EMPID,2)=0;
select * from zeetestsql where MOD(EMPID,2)!=0;

--15.	Write an SQL query to create a new table with data and structure copied from another table.

Create table emp AS (select * from zeetestsql);

--16.Write an SQL query to create an empty table with the same structure as some other table.

--17.	Write an SQL query to fetch top 3 HIGHEST salaries?
select*from (select distinct salary from zeetestsql ORDER BY salary DESC) where rownum<4;

--18.	Find the first employee and last employee from employee table? 

--19.	 List the ways to get the count of records in a table?
select count(*) from zeetestsql;
select count(COLUMN) from zeetestsql;

--20.Write a query to fetch the department-wise count of employees sorted by department's count in ascending order?

select dept,count(*) from zeetestsql group by dept;
select dept,count(*) from zeetestsql group by dept ORDER BY COUNT(*);

--21.Write a query to retrieve Departments who have less than 4 employees working in it. 
select dept,count(*) from zeetestsql group by dept having count(*)<4;

--22.Write a query to retrieve Department wise Maximum salary.
select dept,salary from zeetestsql group by dept;

select dept,max(salary) from zeetestsql group by dept;

--23.Write a query to Employee earning maximum salary in his department. 

SELECT * FROM zeetestsql E1 Join(
    select dept,max(salary) sal from zeetestsql group by dept) E2
    on E1.dept = E2.dept and E1.salary = E2.sal;

--24.Write an SQL query to fetch the first 50%> records from a table
select*from zeetestsql where rownum <= (select count(*) from zeetestsql)/2;

--25.Query to fetch details of employees not having computer.

select * from zeetestsql where COMPID is NULL;

--26.Query to fetch employee details along with the computer details who have been assigned with a computer.
select * from zeetestsql E join where COMPUTER C on E.COMPID=C.COMPID;


--27.Fetch all employee details along with the computer name assigned to them. 

--28.Fetch all Computer Details along with employee name using it.
--29.Delete duplicate records from a table
--30. Find Nth Highest salary

		







































