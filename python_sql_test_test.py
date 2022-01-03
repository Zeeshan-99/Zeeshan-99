# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#-------------------------------100% working mysql-reading and retrieving data-------------------------------------------
import pandas as pd
import  numpy as np
import mysql.connector

#----------------------------Creating new database-----------------------------------------------------------------------
database=mysql.connector.connect(host='localhost',user='root',passwd='root')
cursor=database.cursor()
cursor.execute('CREATE DATABASE cbir')

#--------------------Now establishing connection and creating new table------------------------------
database=mysql.connector.connect(host='localhost',user='root',passwd='root',database='cbir')
cursor=database.cursor()

cursor.execute("create table img(images blob not null)")
database.commit()
cursor.close()
database.close()
#--------------Now Retrieving data from mysql-table and showing----------------------------------------------------------------


import  mysql.connector
import base64
import io

import PIL.Image
with open('F:\ZePycharm\helicopter3.jpg', 'rb') as f:
    photo = f.read()
encodestring = base64.b64encode(photo)
db= mysql.connector.connect(user="root",password="root",host="localhost",database="cbir")
mycursor=db.cursor()
sql = "insert into img values(%s)"
mycursor.execute(sql,(encodestring,))
db.commit()
sql1="select * from img"
mycursor.execute(sql1)
data = mycursor.fetchall()
data1=base64.b64decode(data[0][0])
file_like=io.BytesIO(data1)
img=PIL.Image.open(file_like)
img.show()
db.close()

#----------------------------------------------------
import pandas as pd
import seaborn as sns
import numpy as np

path1="F:\\ZePycharm\\zee_pycharm_Analysis_codes\\salesfunnel.xlsx"

df=pd.read_excel(path1,sheet_name='Sheet1')





