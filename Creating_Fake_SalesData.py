# Creating a fake sales data using python to make a lucrative dashbord in Excel using advanced feature


import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import datetime



sales= pd.DataFrame({'Segment':np.random.choice(['Government','Midmarket','Channel Partner','Enterprise','Small Business'],size=337), 
                     'Employee':np.random.choice(['Peter Jones','Shane Bond','Leo Paul','Ashley Thomas','John Terry'],size=337),
                     'Product':np.random.choice(['Computer','Printer','Projector','Phone'],size=337),
                     'Discount':np.random.choice(['High','None'],size=337),'Unit_Sold':np.random.randint(200,5000,337),
                     'Manf_cost':np.nan,'Sales_Price':np.nan,'Gross_Sales': np.nan, 'Sales':np.nan,
                     'Profit_per_Unit':np.nan,'Profit':np.nan,'Date':random_date(start='2014/01/01',end='2014/12/30',periods=337)})




#--------------------------------------------------------------------------------------------
salesdata1=salesdata.copy()
salesdata1['Manf_cost']= salesdata1['Product'].map({'Phone':20,'Computer':100,'Printer':50,'Projector':75})

salesdata1['Sales_Price']= salesdata1['Product'].map({'Phone':45,'Computer':170,'Printer':80,'Projector':150})
salesdata1['Profit_per_Unit']=salesdata1['Sales# _Price']-salesdata1['Manf_cost']
salesdata1['Profit']=salesdata1['Unit_Sold']*salesdata1['Profit_per_Unit']
salesdata1['Gross_Sales']= salesdata1['Unit_Sold']*salesdata1['Sales_Price']

salesdata1['Sales']= salesdata1['Gross_Sales']

salesdata1['Month_Number']=salesdata1['Date'].apply(lambda x:x.month)
salesdata1['Year']=salesdata1['Date'].apply(lambda x:x.year)

salesdata1.to_excel('salesData.xlsx',index=False)

#---------------------------------------------------------------------------------------------------------------------

# Creating a fake data for * Interective Excel Dashboard Ep.1
import pandas as pd
import numpy as np
import random


salesdata1= pd.DataFrame({'Date':pd.date_range(start='01/01/2017',end='31/12/2019',periods=6000),'Customer Acquisition Type':np.random.choice(['Ad','Returning','Organic'],size=6000),'State':np.random.choice(['Florida','North Carolina','Mississippi','Georgia','Alabama','South Carolina','Tennessee'],size=6000),'Product':np.random.choice(['Product 1','Product 2','Product 3','Product 4','Product 5'],size=6000),'Price':np.nan,'Units':np.random.randint(low=4,high=49,size=6000),'Revenue':np.nan,'Delivery Performance':np.random.choice(['on-time','delayed'],size=6000),'Return':np.random.choice(['no','yes'],size=6000),'Customer Satisfaction':np.random.choice(['(2) low','(1) vey low','(3) ok','(4) high','(5) very high'],size=6000)})


salesData=salesdata1.copy()

salesData['Price']=salesData['Product'].map({'Product 1':499,'Product 2':199,'Product 3':299,'Product 4':99,'Product 5':399})

salesData['Revenue']=salesData['Price']*salesData['Units']

salesData.to_excel('SalesData1.xlsx',index=False,sheet_name='Data')

#--------------------------------------------------------------------------------------------------------------------------