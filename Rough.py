# Project on Classification problem of IRIS-dataset using Multiple classification ML algorithm.

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier       # AdaBoost
from sklearn.ensemble import RandomForestClassifier    # Random Forest
import pandas as pd
import numpy as np
from numpy.core.fromnumeric import argmax, argmin, reshape
from numpy.core.numeric import Inf
from numpy.core.shape_base import stack
from numpy.lib.utils import info
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from scipy.sparse.construct import random
import seaborn as sns
from seaborn import palettes
from seaborn.axisgrid import FacetGrid, JointGrid, jointplot
from seaborn.categorical import countplot, pointplot, swarmplot, violinplot
from seaborn.distributions import distplot, histplot
from seaborn.regression import lmplot
from seaborn.relational import scatterplot
from seaborn.utils import get_dataset_names
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS


from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


df = sns.load_dataset('IRIS')

df.groupby('species').size()

df.plot(kind='box', subplots=True, layout=(2, 2))

df.hist()

sns.pairplot(df)

array = df.values

X = array[:, 0:4]
Y = array[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

X_train.shape
X_test.shape


# Model applying methods.

models = []

models.append(('LR', LogisticRegression(
    solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluating each model in tern

results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(
        model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s:%f(%f)' % (name, cv_results.mean(), cv_results.std()))


plt.boxplot(results, labels=names)

# -----
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ---Evaluation prediction------------
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# ------Red wine dataset to analysze using classification algorithm----


# import xgboost as xgb                                 # extreem boosting
# from xgboost import XGBClassifier


df = pd.read_csv('winequality-red.csv')
# df.isna().sum()

# df.columns

# plt.hist(df['quality'])

# sns.distplot(df['quality'], rug=True,norm_hist=True)

# plt.figure(figsize=(15,10))
# sns.heatmap(df.corr(), annot=True, cmap=sns.diverging_palette(220,20,as_cmap=True))


# df['quality'].unique()

# df['goodquality']=[1 if x>=7 else 0 for x in df['quality']]

# #Seperating feature variable and target variable

# X=df.drop(['quality','goodquality'],axis=1)
# Y=df['goodquality']

# # Value count of good quality and bad quality wine
# df['goodquality'].value_counts()

# # Standard feature of variable
# X
# X=StandardScaler().fit_transform(X)


# X_train,X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.25, random_state=0)


# models=[]

# models.append(('CART',DecisionTreeClassifier()))
# models.append(('R-Forest', RandomForestClassifier()))
# models.append(('AdaBoost',AdaBoostClassifier()))
# #models.append(('GBoost',GradientBoostingClassifier()))
# # models.append(('XGB',XGBClassifier()))
# # Crosss validation technique through kflod methos

# results=[]
# names=[]

# for name, model in models:
#     kFold=StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
#     cv_results=cross_val_score(model,X_train,Y_train, cv=kFold,scoring='accuracy')
#     results.append(cv_results)
#     names.append(name)

#     print('%s:%f(%f)'%(name, cv_results.mean(), cv_results.std()))

# #--------
# plt.boxplot(results, labels=names)


# Now modeling and prediction on testing dataset
model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Now making dataframe
df_pred = pd.DataFrame({'Y_trur': y_test, 'y_pred': predictions})
# Now accuracy ckecking

conf_accuracy = confusion_matrix(y_test, predictions)

acc_scr = accuracy_score(y_test, predictions)
class_rpt = classification_report(y_test, predictions)

print("Confusion matrix :", conf_accuracy)
print("accuracy score:", acc_scr)
print("Classification report :", class_rpt)


# --------------------------------------------------

#  California house price prediction using Multiple ML algorithm


data = pd.read_csv("kc_house_data - Copy1.csv")

data.head(3)

# Custom Transformer that extracts columns passed as argument to its constructots


class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class constructror
    def __init__(self, feature_names):
        self._feature_names = feature_names
     # Return self nothing else to do here

    def fit(self, X, y=None):
        return self
    # Method that describes what we need this transformer to do.

    def transformer(self, X, y=None):
        return X[self._feature_names]

# Categorical Pipeline


class CategoricalTranformer (BaseEstimator, TransformerMixin):
    def __init__(self, use_dates=['year', 'month', 'day']):
        self._usedates = use_dates

    def fit(self, X, y=None):
        return self

    def get_year(self, obj):
        return str(obj)[:4]

    def get_month(self, obj):
        return str(obj)[4:6]

    def get_day(self, obj):
        return str(obj)[6:8]

    def create_binary(self, obj):
        if obj == 0:
            return 'No'
        else:
            return 'Yes'

# Tranformer method we erote for this transformer

    def transformer(self, X, y=None):
        for spec in self._use_dates:
            exec("X.loc[:,'{}']=X['date'].apply(self.get_{})".format(spec, spec))
        # Drop unusable column
        X = X.drop('date', axis=1)

        # Now handling binary for one hot-encoder
        X.loc[:, 'waterfront'] = X['waterfront'].apply(self.create_binary)
        X.loc[:, 'waterfront'] = X['waterfront'].apply(self.create_binary)
        X.loc[:'yr_renovated'] = X['yr_renovated'].apply(self.create_binary)
        return X.values

    # Numerical Pipeline


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bath_per_bed=True, years_old=True):
        self._bathe_per_bed = bath_per_bed
        self._years_old = years_old

    def fit(self, X, y=None):
        return self

    def transformer(self, X, y=None):
        if self._bathe_per_bed:
            X.loc[:, 'bath_per_bed'] = X['bathrooms']/X['bedrooms']
            X.drop('bathrooms', axis=1)
        if self._years_old:
            X.loc[:, 'years_old'] = 2019-X['yr_built']
            X.drop('yr_built', axis=1)

        X = X.replace([np.Inf, -np.inf], np.nan)
        return X.values

# Combining the piplines together


categorical_features = ['date', 'waterfront', 'view', 'yr_renovated']
numerical_features = ['bedrooms', 'bathrooms', 'sqrft_living', 'sqft_lot', 'floors',
                    'condition', 'grade', 'sqrft_basement', 'yr_built']


# Defining steps in the categorical pipeline
categorical_pipeline = Pipeline(steps=[('cat_selector', FeatureSelector(categorical_features),
                                       ('cat_transformer', CategoricalTranformer()),
                                       ('one_hot_encoder', OneHotEncoder(sparse=False))])
# Defineing steps in the Numerical Pipeline
numerical_pipline=Pipeline(steps=[('num_selector', FeatureSelector()),
                                   ('num_transformer', NumericalTransformer()),
                                   'imputer', SimpleImputer(strategy='median'),
                                   'std_scaler', StandardScaler()])
# Combining the categorical and numerical pipeline into one big pipeline horsizontally
full_pipeline=FeatureUnion(transformer_list=[('categorical_pipeline', categorical_pipeline),
                                              ('numerical_pipeline', numerical_pipeline)])
# Now full pipoleine

full_pipeline


# ---Data visualization practice:------
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data

# %%

# data()   # to see the list of all available dataset

tips=data('tips')
list(tips.columns)
tips.info()

sns.lmplot(x='total_bill', y='tip', hue='sex',
           aspect=0.9, size=10, col='day', data=tips)

# Categorical plot
sns.countplot(x='sex', hue='smoker', data=tips)
sns.boxplot(x='sex', y='tip', hue='smoker', data=tips)
sns.boxplot(x='sex', y='tip', hue='day', data=tips)

sns.violinplot(x='sex', y='tip', hue='time', data=tips)
sns.pointplot(x='sex', y='tip', data=tips)
sns.swarmplot(x='sex', y='tip', hue='time', data=tips)

# ----Continous variable plot:--------

sns.distplot(tips['tip'], hist=True, kde=True, color='r', bins=10)
plt.grid()

plt.figure(figsize=(10, 12))
sns.distplot(tips['total_bill'], bins=50, rug=True, vertical=0)
plt.grid()

plt.figure(figsize=(10, 12))
sns.distplot(tips['total_bill'], rug=True, vertical=0, color='r', hist=True)
plt.grid()


sns.jointplot(x='total_bill', y='tip', hue='sex', data=tips)

sns.jointplot(x='total_bill', y='tip', kind='reg', data=tips)
sns.jointplot(x='total_bill', y='tip', kind='hex', data=tips)
sns.jointplot(x='total_bill', y='tip', kind='kde', data=tips)

# -----FacetGrid plot using seaborn library:--------
# titanic=data('titanic')  this data set has been taken from pydataset
titanic2=sns.load_dataset('titanic')

# --distplot for age by sex wise----
g=sns.FacetGrid(data=titanic2, col='sex')
g.map(sns.distplot, 'age')

# ----Here we are creating visualization on tips dataset:--------
tips=sns.load_dataset('tips')

g=sns.FacetGrid(data=tips, col='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip', hue='sex')

g=sns.FacetGrid(data=tips, col='sex')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip', hue='time')
g.add_legend()

g=sns.FacetGrid(data=tips, col='sex', row='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip', hue='smoker')
g.add_legend()

g=sns.FacetGrid(data=tips, col='sex', row='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip', hue='day')
g.set_axis_labels("Total bill", "Tip")
g.add_legend()

g=sns.FacetGrid(data=tips, col='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip')
g.refline(y=tips['tip'].median())
g.set_axis_labels("Total bill", "Tip")
g.add_legend()

# -----hist plot:----------------------------------------------------------------
g=sns.FacetGrid(data=tips, col='size')
g.map_dataframe(sns.histplot, 'tip')

g=sns.FacetGrid(data=tips, col='day')
g.map_dataframe(sns.histplot, 'tip', hue='sex')
g.add_legend()

g=sns.FacetGrid(data=tips, col='time', row='sex')
g.map_dataframe(sns.histplot, 'tip')
g.add_legend()

# ----jointplot:-------
g=sns.FacetGrid(data=tips, col='time', margin_titles=True)
g.map_dataframe(sns.jointplot, x='total_bill', y='tip', hue='sex')
g.tight_layout()
g.add_legend()

# -----linear plot:------

g=sns.lmplot(x='total_bill', y='tip', col='smoker', row='sex', data=tips)
# ----advanced linear plot using xtics:------
g=sns.lmplot(x='total_bill', y='tip', col='smoker', row='sex', data=tips)
g.set_axis_labels("Zee", "Farz").set(xlim=(0, 60), ylim=(
    0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])

# --- Pairplot:----
g=sns.pairplot(data=tips, hue='sex')
# ---violinplot:----

g=sns.FacetGrid(data=tips, col='sex')
g.map_dataframe(sns.violinplot, x='day', y='total_bill',
                hue='smoker', split=True, inner='quart')
g.add_legend()

#



# Create a function to d

# Now pie plt.plot
tips['tip'].min()
tips['tip'].max()
tips['tip'].unique()

val=tips[['tip']]

# ---Creating grou data using cut function available in python

df=pd.DataFrame({'number': np.random.randint(1, 100, 10)})
df['bins']=pd.cut(x=df['number'], bins=[1, 20, 40, 60, 80, 100])
df
# -----creating a dataframe:------
Designation=['Asst Professor', 'Asoc. Professor', 'Professor']

Ass_sal=np.random.randint(150000, 250000, 100)
Aso_sal=np.random.randint(250000, 350000, 100)
Prof_sal=np.random.randint(350000, 500000, 100)


df=pd.DataFrame({'Ass Professor': np.random.randint(150000, 250000, 100),
'Associate Prof': np.random.randint(250000, 350000, 100),
'Professor': np.random.randint(350000, 500000, 100)})

df.head()
df_stacked=df.stack()

df_stacked.to_excel("Prof.xlsx")

df_stacked1=df_stacked.to_frame()

df_pivot=pd.pivot_table(df, index=df.columns, values=df.values)
df_ustacked=df.unstack()

df_melted=pd.melt(df, id_vars=['Ass Professor'], value_vars=[
                  'Ass Professor', 'Associate Prof', 'Professor'])
plt.pie(val)

# -----------------------





# -------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv('GOOG.csv')


df.isna().sum()

data=df.sort_index(ascending=True, axis=0)
# creating dataframe
Date=[]
Close=[]

for i, j in data.iterrows():
      Date.append(j['Date'])
      Close.append(j['Close'])

new_data=pd.DataFrame(zip(Date, Close), index=range(
    0, len(df)), columns=['Date', 'Close'])

new_data.index=new_data.Date

new_data.drop(['Date'], axis=1, inplace=True)
# ----
dataset=new_data.values
train=dataset[0:790, :]
valid=dataset[790:, :]
# ------------
# --scaling the dataset---
scaler=MinMaxScaler(feature_range=(0, 1))
scaled_data=scaler.fit_transform(dataset)

# ----coverting dataset into training
import numpy as np
x_train, y_train=[], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i])

x_train, y_train=np.array(x_train), np.array(y_train)
x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# --------------Fitting RNN model------------------------------
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout

model=Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# -----------------Predicting values:--------
inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1, 1)
inputs=scaler.transform(inputs)
# 3----------
x_test=[]

for i in range(60, inputs.shape[0]):
    x_test.append(inputs[i-60:i, 0])
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test.shape
# -----------------------------------
# Prediction
closing_price=model.predict(x_test)
closing_price=scaler.inverse_transform(closing_price)

# -------for plotting:------
train=new_data[0:790]
valid=new_data[790:]

valid['Prediction']=closing_price

# ---plot with matplotlib:----------------

plt.plot(train['Close'])
plt.plot(valid['Close'])
plt.plot(valid['Prediction'])

# -------now plotting with plotly:----------------

train1=train.copy()
train1.reset_index(inplace=True)
valid1=valid.copy()
valid1.reset_index(inplace=True)
# -----------
fig=go.Figure(data=[go.Scatter(x=train1['Date'],
                                y=train1['Close'],
                                name='training',
                                line_color='#FF337D')])
fig.add_trace(go.Scatter(x=valid1['Date'],
                                y=valid1['Close'],
                                name='test',
                                line=dict(color='#33FF46')))
fig.add_trace(go.Scatter(x=valid1['Date'],
                                y=valid1['Prediction'],
                                name='prediction',
                                line=dict(color='#FF4833')))
fig.update_layout(template='plotly_dark')
fig.update_layout(yaxis_title='Prices', xaxis_title='Date', title='Stock')

fig.show()

# ----------------------------------------
# writing a code to print reveresed order of the entred digit

Number=123456789
n=Number
rev=0
while(Number>0):
    reminder=Number%10
    rev=rev*10+reminder
    Number=Number//10
print('The reversed number of the entred number = {} is ={}'.format(n,rev))      

#-----------------------------------------------------------------------
# write a coding programm for factorila finding
n=5
fact=1
for i in range(n+1):
    if i==0:
         return 1
    elif i==1:
         return 1
    else:
         fact=fact*(i) 
           
print("The factorial is:" %fact)            
#----------------
    
def fbn(n):
    a,b=0,1
    cnt=0
    if n<0:
        print("Enter any positive number")    
    elif n==1:
            print(a)
    else:
         print("Fibanicci series")
         while(cnt<=n):
             print(a)
             sum=a+b
             a=b
             b=sum
             cnt=cnt+1
print(fbn(n))
#-----------------------------------------
# reveresed order programm


n=12345
num=n
rev= 0
while(n>0):
    rem=n%10
    rev= rev*10+rem
    n=n//10
print("Entred number is",num)    
print("Reversed number is",rev)    



# Fibanacci number seriese:
# !Fibacci serises: 0,1,1,2,5,3,8,11,.................

def fib(n):
    a,b=0,1
    count=0
    if n<=0:
        print("Please enter any positive number")
    elif n==1:
        print(a)
    else:
        while(count<=n):
            print(a)
            sum=a+b
            a=b
            b=sum
            count=count+1
        print(sum)
# print("Fibonacci series is :", fib())           
        


        
    
    
    
    
    






    
    
    

