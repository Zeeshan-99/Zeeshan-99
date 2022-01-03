

\\=-------------------------------------------------------
# Top 15 Python Programming Questions & Solutions -asked in Amazon, Facebook, Microsoft, Tesla Interviews
# link:https://www.youtube.com/watch?v=BCZWQTY9xPE
1. Write a Python Program to print Prime Numbers between 2 numbers
2. Write a Sort function to sort the elements in a list
3. Write a sorting function without using the list.sort function
4. Write a Python program to print Fibonacci Series
5. Write a Python program to print a list in reverse
6: Write a matrix 10x10 code or generalized code (Addition, Multiplication, etc)
6. Write a Python program to check whether a string is a Palindrome or not 
7. Write a Python program to print set of duplicates in a list
8. Write a Python program to print number of words in a given sentence                                    ***
9. Given an array arr[] of n elements, write a Python function to search a given element x in arr[].
10. Write a Python program to implement a Binary Search
11. Write a Python program to plot a simple bar chart
12. Write a Python program to join two strings (Hint: using join())                                         **
13. Write a Python program to extract digits from given string                                              **
14. Write a Python program to split strings using newline delimiter
15. Given a string as your input, delete any reoccurring character, and return the new string.
16. Reversed number in Python
17. Revesed string in Python
18. How do you calculate the number of vowels and consonant peresent in a string
19. How do you get a matching elecemt in an integer array
20. Code the bubble sorts algorithm
21. how do you reverse an array
22. How do you reverse a list
23. Swap the two numbers without third variable.
24. How do you implement a binary search
25. find the second largest number in the array
26.

\\=----------------------------------------------------------------
    
#Q1: Write a python programm to check prime number

# 1st method : It is not so accurate as it should be, when i check for 25 it shows me it is a prime number while it is false.

number=int(input("Enter any number: "))

if number>1:
    if (number%2==0):
        print ('It is not a prime number')    
    else:
        print('It is a prime number')
else:
    print('Plese enter any number greater than 1 in  order to check prime number')    
# 2nd method: it 100% accurate result:
n=25
if n>1:
    for i in range(2,n+1):
        if n%i==0:
            print("It is not a prime number")
        else:
            print("It is a prime number")   
        

# 3rd method: It is 100% accurate result

number=int(input("Enter any number: "))
if number>1:
    for i in range(2,number+1):
        if (number%i==0):
            print('It is not a prime number') 
            break
        else:    
            print('It is a prime number')
            break 
else:
    print('Please enter any number greater than 1 in order to check the prime number')
    
# 3rd method by defining a defination.

def primeCheck(X):
        if X>1:
            for i in range(2, X+1):
                if X%i==0:
                    return 'It is not a prime number'
                    # print('It is not a prime number') 
                    break 
                else:
                    return 'It is a prime number'
                    # print('It is a prime number')
        else:
            return 'Plase enter any number greater than 1 in  order to check prime number'
            # print('Plase enter any number greater than 1 in  order to check prime number')           

# Write a programm to print the list of the 100 prime numbers or n prime numbers.

lower = 1
upper = 100

print("Prime numbers between", lower, "and", upper, "are:")

for num in range(lower, upper + 1):
   # all prime numbers are greater than 1
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           print(num)  
#--------2nd methods------------------------
lower = 1
upper = 100

prime_number=[]

print("Prime numbers between", lower, "and", upper, "are:")

for num in range(lower, upper + 1):
   # all prime numbers are greater than 1
   if num > 1:
       for i in range(2, num):
           if (num % i) == 0:
               break
       else:
           prime_number.append(num)
print(prime_number)

# Q2: write a code for odd/even number in python
#1st methods

num=int(input("Enter any number: "))

if (num % 2==0):
    print('It is an even number') 
else:
    print('Enter value greater than 1')  
    
#2nd methods
def OddEven(X):
    if (X % 2==0):
        print('It is an even number') 
    else:
        print('It is an odd number')
        

#3rd methods (under construction ):
while True:
    try:
        num= int(input("Enter any number: ")
         if (num % 2==0):
            print( "{} is an even number".format(num))
        else:
            print("{} is an odd number".format(num))
            break
    except ValueError:
        print("Oops!  That was no valid number.  Try again...")        
print('You succesfully run your Odd even number problem') 

# Q3: Write a code to swap (interchange) two variable value like x into y and y into.

#fisrt method
x = input("Enter X value : ")
y = input("Enter Y value : ")
#Swapping function 
temp=x
x=y
y=temp

print("The value of x after swapped : {}".format(x))
print("The value of y after swapped: {}".format(y))

#2nd method

def swap(x, y):
    x,y=y,x
    print("The value of x after swapped : {}".format(x))
    print('The value of y after swapped:{}'.format(y))

#Q4: Write a programm code for even list

import numpy as np
#---------for loop:----------
l=np.linspace(1,20,20, dtype='int')
l_even=[]
for item in l:
    if item%2==0:
        l_even.append(item)
print("You have list of all even numbers :", l_even) 

#-------2nd methos:----------
n=20
l_even=[]
for item in range(1,n+1):
    if item%2==0:
        l_even.append(item)
print("You have list of all even numbers :", l_even)  

#------3rd method:----using while loop:------------------=--
n=20
l_even=[]
while (n>1):
    if n%2==0:
        l_even.append(n)
    n=n-1    
print("You have list of all even numbers :", l_even)  

#--------------list comprehension:---------

l= np.linspace(1,20, 20, dtype='int')
l_even_comp=[item for item in l if item%2==0]
print("You have list of all even numbers :", l_even_comp)    
#---------------------------------------
#Q5: Write a program to filter out the number divided by 3 from the tuple
# using lambda function

#1st method

l=np.linspace(1,20,20,dtype='int')

l_filter=tuple(filter(lambda x:(x%3==0),l))    
print(l_filter)    

#2nd Method
l=np.linspace(1,20,20, dtype='int')
l_filter=list(filter(lambda x: (x%3==0),l))
print(l_filter)

# 3rd method
l=np.linspace(1,20,20, dtype='int')
list_every_3rd=[]
for item in l:
    if item%3==0:
        list_every_3rd.append(item)
print("You have list of all every 3rd element :", list_every_3rd)
 
#Q6: Write a matrix 10x10 code or generalized code
#!link:https://www.youtube.com/watch?v=66hIDupiJjo    
# 1st method
matrix=np.arange(1,101).reshape(10,10)
#2nd method
row= int(input("Enter the number of rows :"))
col= int(input("Enter the number of columns"))

Matrix=[]

for i in range(row):
    a=[]
    for j in range(col):
        a.append(int(input()))
    Matrix.append(a)  
print(Matrix)    

#-----------

# Matrix Multiplication problem Related:
# !3x3 matrix
X=[[12,7,3], 
   [4,5,6],
   [7,8,9]] 
#! 3x4 matrix
Y=[[5,8,1,2],
   [6,7,3,0],
   [4,5,9,1]]

#result
# !3x4 matrix
result=[[0,0,0,0],
        [0,0,0,0],
        [0,0,0,0]]

# !iterate through rows of X
for i in range(len(X)):
    # !iterate through columns of Y
    for j in range(len(Y[0])):
        # !iterate through rows of Y
        for k in range(len(Y)):
            result[i][j]+=X[i][k]*Y[k][j]  
print(result)

# Matrix summation problem Related:

# !3x3 matrix
X=[[12,7,3], 
   [4,5,6],
   [7,8,9]] 
#! 3x4 matrix
Y=[[5,8,1,2],
   [6,7,3,0],
   [4,5,9,1]]

#Result: 
result=[[sum(a*b for a,b in zip(X_rows,Y_rows)) for Y_col in zip(*Y) for X_rows in X]]
#------------------------------------------------------

#Q7: Write a code for finding average of n numbers

def avg_num(x):
    sum_num=0
    for i in x:
        sum_num=sum_num+i
    avg =sum_num/len(x) 
    return avg  
#note: x could be a list


#Q7:How to write a code for list count
list1 = ['red', 'green', 'blue', 'orange', 'green', 'gray', 'green']
color_count = list1.count('green')
print('The count of color: green is ', color_count)

#Q8: How can you count duplicate elements in a given list?

list1=[3,3,3,3,5,6,7,8,9,10]
num_count=list1.count(3)
print('The count of duplicate count: 3 is ', num_count)

#Q9: Write a code to get index of an element in a list using for loop

my_list = ['Guru', 'Siya', 'Tiya', 'Guru', 'Daksh', 'Riya', 'Guru'] 
all_indexes = [] 
for i in range(0, len(my_list)) : 
    if my_list[i] == 'Guru' : 
        all_indexes.append(i)
print("Original_list ", my_list)
print("Indexes for element Guru : ", all_indexes)

# Q10: Write a programme for any dataset using any classifier to find the accuracy of the model

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pydataset import data

#dataset: IRIS
df1= sns.load_dataset('IRIS')
df2=data('iris')

#---------Descptive statistic:----
df1.isnull().sum()
df1.shape
df1.describe()
df1.info()

#-------Selecting the target variable:---------

X=df1.iloc[:,:4]
X.info()
X.shape

Y=df1.iloc[:,-1]
Y.shape

#-----Importing sklearn libraries

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#---importing Decision tree classifier:----------
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#------- Now spliiting the dataset into training and testing:----

X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.20, random_state=42)

X_train.shape
Y_train.shape
X_test.shape
Y_test.shape
#-----------Now training our model using training datset:------
model= DecisionTreeClassifier()
model.fit(X_train, Y_train)

#----Now predicting : X-test data

predictions=model.predict(X_test)

#-----Checking model accuracy:----------
print(accuracy_score(Y_test, predictions))
print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))


#--------------------------------------------------------------
#Q11: Write a programme for Linear regression model and check their accuracy report as well

df=pd.read_csv('E:/Z-Jupyter/USA_Housing.csv')
df.head()
df.info()
df.shape

sns.pairplot(df)
sns.displot(df['Price'])

df.isnull().sum()

df.corr()
sns.heatmap(df.corr(), annot=True)

#----Now selcting the target variable:-----
df.info()

X=df.iloc[:,:5]
X.info()

Y=df.loc[:,'Price']
Y.info()

X.shape
Y.shape

#---Now splitting the datset into training and testing :-----
X_train,X_test,Y_train,Y_test =train_test_split(X,Y, test_size=0.33, random_state=42)

X_train.shape
Y_train.shape

#----Now importing linear regression models
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

model= LinearRegression()
model.fit(X_train,Y_train)


predictions= model.predict(X_test)

predictions
Y_test

#-----
print(model.intercept_)
print(model.coef_)

#-------scatter plot:-----
sns.scatterplot(Y_test, predictions)
plt.legend(True)
#-----Distplot:----
sns.distplot(Y_test-predictions)
#----------------------------------------------------------------
#----model accuracy checking:----

mse=mean_absolute_error(Y_test, predictions)

rmse= np.sqrt(mse)
rmse




#---------------------------------------------------
import pandas as pd
import numpy as np

df=pd.read_excel('Selectiondata.xlsx')
df.isna().sum()

df1=df.fillna(0)
df1.isna().sum()

df1=df1.astype(int)

df1['D']=np.where(df1['ST']==df1['AD'], '1','0')
df1

# duplicate_value=[]

# for i in range(df.shape[0]):
#     if df1['ST']=df1['AD']:
#         print(df['ST'])

# print(duplicate_value)

df1.shape[0]
#-------------------------------

# Q: Write the code for finding the maximum number of the given two
def maxfind(a,b):
    if a>b:
        return a
    else:
        return b

a=-10
b=-20
print(maxfind(a,b))            

# Q: Write the programming for the finding circle area

def circleArea(r):
    area=(22/7)*r**2
    # area=(22/7)*(r*r)
    return area
r=10   
print("Area of the circle is: %.2f" %circleArea(r)) 

# Q: Write the programming code for finding the sum of the square of first n natural number 

def squar_n_number(n):
            sm=0
            for i in range(1,n+1):
                sm=sm+(i*i)
            return sm   
    
print( squar_n_number(64)) 

# Q: Write a programme code to the cube sum of the first n natural numbers;

def cube_n_number(n):
    sm=0
    for i in range(1,n+1):
        sm=sm+(i**3)
    return sm 

cube_n_number(4)


# Q: Sum of the all n natural number cube with alternative sign.
    
def cub_alt_sign(n):
    sm=0
    for i in range(1,n+1):
        if (i%2)==1:                 # to get the all odd number
            sm=sm+(i*i*i)              #sum the all odd numbers
        else:
            sm=sm-(i*i*i)            # to get the all even number
    return sm                        # sum the all even numbers
            
cub_alt_sign(5)

# Q: Sum of cube of n-even number 

#https://www.geeksforgeeks.org/sum-of-cubes-of-first-n-even-numbers/?ref=lbp
def cube_sum_even(n):
    sm=0
    for i in range(1,n+1):
        sm=sm+(2*i)*(2*i)*(2*i)
    return sm
    
print(cube_sum_even(8))  

# Q: Sum of cube of n-odd number 

# https://www.geeksforgeeks.org/sum-of-cubes-of-first-n-odd-natural-numbers/?ref=lbp
def cube_sum_odd(n):
    sm=0
    for i in range(0,n):
        sm=sm+(2*i+1)*(2*i+1)*(2*i+1)
    return sm
print(cube_sum_odd(2))  

# Q: write a factorial function code for n natural number

def factorial_n(n):
    if n<0:
        return 0
    elif n==0 or n==1:
        return 1
    else:
        fact=1
        while(n>1):
            fact=fact*n   #fact*=n
            n=n-1          #n-=1
        return fact
print(factorial_n(2))

#----2nd methos: for factorial-----------------------------------
n=-4
fact=1
while (n>0):
         fact=fact*n
         n=n-1 
           
print("The factorial is:", fact)            


#Q: Write a program to generate fibonacci series

def febonacci_series(n):
            #First two term
            a,b=0,1
            count=0
            #Check if the number of terms is valid
            if n<=0:
                print("The enter a positive integer")
            #If there is only one term, return n1
            elif n==1:
                print("Fibonacci sequence upto",n,':')
                print(a)
                #generate fibonacci sequence
            else:
                print("Fibonacci sequence:")
                while(count<=n):
                    print(a)
                    sum=a+b
                    a=b
                    b=sum
                    count=count+1
                            
print(febonacci_series(5))

# Q: Write a program to create a pattern of numbers using for loop
# 1.Square patter
n=4
for i in range(n):
    for j in range(n):
        print(" * ", end='')
    print()    

# 2.Left Triangle patter 
n=4
for i in range(n):
    for j in range(i+1):
        print(" * ", end='')
    print()    
    
    
# 2.Revers Left Triangle patter 
# (1st method)
n=4
for i in range(n):
    for j in range(4-i):
        print(" * ", end='')
    print()    
# (2nd method)
n=4
for i in range(n):
    for j in range(i,n):
        print(" * ", end='')
    print()    

# 2.lower Right Triangle patter 
n=5
for i in range(n):
    for j in range(n-i-1):
        print(" ", end=" ")
    for j in range(i+1):
        print("*", end=" ")    
    print()    

# Reverse left triangle
n=5
for i in range(n):
    for j in range(n-i):
        print('*', end=' ')
    for j in range(i):
        print(' ', end=' ') 
    print()    

# Reverse right triangle
n=5
for i in range(n):
    for j in range(5-i-1):
        print('*', end=' ')
    for j in range(i):
        print(' ', end=' ') 
    print()    
    
# Q: Printing pyramid stars using while loop

# link:https://www.youtube.com/watch?v=PTHSTjBfXmY

# n=int(input("Enter the number of rows :"))
n=10
k=1
i=1
while i<=n:
    b=1
    while b<=n-i:
        print(" ", end="")
        b=b+1
    j=1
    while j<=k:  
        print ("*", end="") 
        j=j+1
    print() 
    i=i+1
    k=k+2    
    
    

# Q: Printing reverse pyramid stars using while loop function

# n=input("Enter the no of rows: ")
n=10
i=1
while(n>0):
    b=1
    while(b<i):
        print(" ", end=" ")
        b=b+1
    j=1
    while(j<=n*2-1):         # we could also put 1 inplace of i
        print("*", end=" ") 
        j=j+1
    print()
    n=n-1
    i=i+1        
        
# Q: Write the code to printing the stars in the shape of hallow heart

for row in range(6):   
    for col in range(7):
        if (row==0 and col%3!=0) or (row==1 and col%3==0) or (row-col==2)or(row+col==8):
            print("*", end="")
        else:
            print(" ", end="")    
    print()    
             

# Q: Write the code to printing the stars in the shape of solid heart

for i in range(4):
    for j in range(4-i-1):
        print(" ", end="")  
    for j in range(i+1):
        print("* ", end="") 
    for j in range(2*(4-i-1)):
         print(" ", end="")  
    for j in range(i+1):
        print("* ", end="")        
    print()                 
for i in range(8,0,-1):
    for j in range(8-i):
        print(" ", end="") 
    for j in range(i,0,-1):
        print("* ", end="") 
    print()        
    
# Q: Write the code to printing the stars in the shape of solid heart with writing text
# link:https://www.youtube.com/watch?v=6lJqSEvE1Rw

num=int(input("Enter the number :"))
n=num//2

for i in range(n):
    for j in range(n-i-1):
        print(" ", end="")  
    for j in range(i+1):
        print("* ", end="") 
    for j in range(2*(n-i-1)):
         print(" ", end="")  
    for j in range(i+1):
        print("* ", end="")        
    print()
                 
for i in range(2*n,0,-1):
    for j in range(2*n-i):
        print(" ", end="") 
    for j in range(i,0,-1):
        print("* ", end="") 
    print()        
    
# second method

num=int(input("Enter the number :"))
msg=input("Enter the message")
l=len(msg)
n=num//2

for i in range(n):
    print(" "*(n-i-1)+"* "*(i+1), end="") 
    if num%2 == 0:
        for j in range(n-i-1):
            print(" ", end="")  
        else:
            for j in range(2*(n-i-1)):
                print(" ", end="") 
        else:
            for j in range(2*(n-i-1)):
                print(" ", end="")        
        for j in range(i+1):
            print("* ", end="") 
        print()     
for i in range(num,0,-1):
    print(" "*(num-i)+"* "*(i))    
#------
# printing starts
n=4
for i in range(n):
    for j in range(n):
      print("*", end="")
    print()
#----------------------------------------------
n=8
for i in range(n):            #ith row
    for j in range(i):    #jth column
        print(" * ", end="")
    print()  
    
      
#----------------------
n=8
for i in range(n,1,-1):            #ith row
    for j in range(i-1):    #jth column
        print(" * ", end="")
    print()  
#----------------------
n=8
for i in range(n,0,-1):
    for k in range(0,n-i):
        print(" ", end="")
    for j in range(1,i+1):
        print("*", end="")   
    print("\n")     
         
# Printing hollow square

n=18
for i in range(n):
    for j in range(n):
        if (i==0 or i==n-1 or j==0 or j==n-1):
            print("*", end="")
        else:
             print(" ", end="")
    print('\n')


# Printing a letter 'A' 

n=5
for i in range(n+2):
    for j in range(n):
        if (((j==0 or j==n-1) and i!=0) or (i==0 or i==3) and (j>0 and j<4)):
            print("*", end="")
        else:
             print(" ", end="")
    print('\n')
    


    
    
#  -------------Rough work  ----------------------------------------
count=0
sum=0
print('before :',"Count=",count,"Sum =",sum)
for value in [9,41,12,3,74,15]:
    count=count+1
    sum=sum+value
    print(count, sum, value)

print("After :", "Count=", count, "Sum =", sum, "Average =", sum/count)    
    
# ----------ending rough work---------------------------------- 

# Q16: Write a code to print the reverse order of any digits (at least two digits)

Number=int(input("Please Enter any Number: "))
Number1= Number
Reverse=0
while(Number>0):
    Reminder = Number%10
    Reverse=(Reverse*10)+Reminder
    Number=Number//10    
# print("Reverse of the entered number is =%d" %Reverse)  
print("Reverse of the entered number ={} is ={}".format(Number1,Reverse))  

#-------------exp---------------------------------------
# Number=int(input("Please Enter any Number: "))
Number = 1234 
Number1= Number
Reverse=0
while(Number>0):
    Reminder = Number%10
    print("Reminder =%d" %Reminder) 
    Reverse=(Reverse*10)+Reminder
    print("Revsersed =%d" %Reverse)
    Number=Number//10  
    print("Numbrrs for which reminder is taken =%d" %Number)  
# print("Reverse of the entered number is =%d" %Reverse)  
print("Reverse of the entered number ={} is ={}".format(Number1,Reverse))  

#-------2nd methods:-----
num = 123456
print(str(num)[::-1])

#--------------------------



