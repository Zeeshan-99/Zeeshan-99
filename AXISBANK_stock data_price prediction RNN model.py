import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

#Interval required 5 minutes
data = yf.download(tickers='AXISBANK.NS', period='50d', interval='5m')

#Print data

data.to_csv('AXISBANK_STOCKdata_50days.csv')


##------ Candle sticks plot using  python:------------
# link: https://stackoverflow.com/questions/53697655/how-to-plot-candlesticks-in-python

import plotly.graph_objects as go

import pandas as pd
from datetime import datetime

df = pd.read_csv('E:\Z-Jupyter\AXISBANK_STOCKdata_50days.csv')
df=df.head(100)
fig = go.Figure(data=[go.Candlestick(x=df['Datetime'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

fig.show()

#-------2nd method:-----------

link:https://www.youtube.com/watch?v=jRZEBat_n7c

import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# df = pd.read_csv('E:\Z-Jupyter\AXISBANK_STOCKdata_50days.csv')
df = pd.read_csv('AAPL.csv')
# df=df.head(100)
df=df.iloc[::-1]
df['Date']=pd.to_datetime(df['Date'])
df['20wma']=df['Close'].rolling(window=140).mean()
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])]
                )
fig.add_trace(
    go.Scatter(
        x=df['Date'],
        y=df['20wma'],
        line=dict(color= '#e0e0e0'),
        name= '20-week'
    )
)
fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
fig.update_layout(yaxis_title ="Stock prices", xaxis_title="Date", title='Apple stocks')
fig.update_yaxes(type="log")
fig.show()

#-----------3rd method using finlap-------

import pandas as pd
import finplot as fplt
import yfinance

df = yfinance.download('AAPL')
# df.to_csv('AAPL_STOCKdata.csv')
# df = pd.read_csv('E:\Z-Jupyter\AAPL_STOCKdata.csv')
# df.dtypes
# df['Date']=pd.to_datetime(df['Date'])
# df=df.set_index('Date')

fplt.candlestick_ochl(df[['Open', 'Close', 'High', 'Low']])
fplt.show()

#---------------RNN-LSTM Model on AXIS BANK data to predict the prices at microlevel (15 minutes intervals) -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------importing required libraries----------
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

#----------importing  data:----------------
df=pd.read_csv('AXISBANK_STOCKdata_50days.csv')
df.info()
df.isnull().sum()
df['Datetime']=pd.to_datetime(df['Datetime'])
df.dtypes

df=df.dropna()
#------Creating dataframe-------------
data= df.sort_index(ascending=True, axis=0)
Date=[]
Close=[]

for i, j in data.iterrows():

    Date.append(j['Datetime'])
    Close.append(j['Close'])
    
new_data = pd.DataFrame(data= zip(Date, Close), index=range(0,len(df)), columns=['Date','Close'])   

#-----------------
new_data.head()

# Setting index
new_data.index= new_data.Date
new_data.drop('Date', axis=1, inplace=True)
new_data

# Creating train and test sets

dataset= new_data.values

train= dataset[0:2500,:]
valid= dataset[2500:,:]

# Scaling the dataset

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data= scaler.fit_transform(dataset)

# Converting dataset into x_train and y_train

x_train, y_train= [],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train= np.reshape( x_train, (x_train.shape[0], x_train.shape[1],1))

#------------Fitting RNN Model--------------------------------------

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# # create and fit the LSTM model
model= Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#------------------
# predicting  values, using past 60 from the train data

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1,1)
inputs= scaler.transform(inputs)

inputs

# X_test creating in 3d format
x_test=[]

for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60:i,0])
    
x_test= np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
x_test.shape

#Prediction 
closing_price= model.predict(x_test)
closing_price= scaler.inverse_transform(closing_price)
closing_price
# Result

rms = np.sqrt(np.mean(np.power((valid-closing_price),2)))
rms


# For plotting

train = new_data[:2500]
valid= new_data[2500:]

valid['Predictions']= closing_price

plt.figure(figsize=(15,8))
plt.plot(train['Close'])
plt.plot( valid[['Close','Predictions']])
plt.legend()
plt.show()


#---Now plotting in plotly format

import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

#----preprocessing for train dataset----
train1=train.reset_index()
train1['Date']=pd.to_datetime(train1['Date'])
train1.head(3)
#--now preprocessing for valid dataset
valid1=valid.reset_index()
valid1['Date']=pd.to_datetime(valid1['Date'])

#---Plotly-----------------
fig = go.Figure(data=[go.Scatter(x=train1['Date'],
                                 y=train1['Close'],
                                 line_color ='#4033FF')]
                )

fig.add_trace(
    go.Scatter(
        x=valid1['Date'],
        y=valid1['Predictions'],
        line =dict(color= '#FF5633'),
        name= 'test'
    )
)

fig.add_trace(
    go.Scatter(
        x=valid1['Date'],
        y=valid1['Close'],
        line =dict(color= '#33FF50'),
        name= 'Predictions'
    )
)
fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark')
fig.update_layout(yaxis_title ="Stock prices", xaxis_title="Date", title='Axis Bank stocks')
fig.update_yaxes(type="log")
fig.show()

#------------------------------------------------------------------------------------------------
#--------   Rough:--------------







