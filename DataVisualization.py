#---Data visualization practice:------

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data

#%%

# data()   # to see the list of all available dataset

tips=data('tips')
list(tips.columns)
tips.info()

# %%

g=sns.FacetGrid(data=tips,row='sex')
g.map_dataframe()
sns.lmplot(x='total_bill', y='tip', hue='sex',
           aspect=0.9, size=10,col='day',data=tips)

# Categorical plot
sns.countplot(x='sex',hue='smoker',data=tips)
sns.boxplot(x='sex',y='tip',hue='smoker',data=tips)
sns.boxplot(x='sex',y='tip',hue='day',data=tips)

sns.violinplot(x='sex',y='tip',hue='time',data=tips)
sns.pointplot(x='sex',y='tip',data=tips)
sns.swarmplot(x='sex',y='tip',hue='time',data=tips)

#----Continous variable plot:--------

sns.distplot(tips['tip'], hist=True, kde=True,color='r',bins=10)
plt.grid()

plt.figure(figsize=(10,12))
sns.distplot(tips['total_bill'],bins=50,rug=True, vertical=0)
plt.grid()

plt.figure(figsize=(10,12))
sns.distplot(tips['total_bill'],rug=True, vertical=0, color='r',hist=True)
plt.grid()


sns.jointplot(x='total_bill',y='tip',hue='sex',data=tips)

sns.jointplot(x='total_bill',y='tip',kind='reg',data=tips)
sns.jointplot(x='total_bill',y='tip',kind='hex',data=tips)
sns.jointplot(x='total_bill',y='tip',kind='kde',data=tips)

#-----FacetGrid plot using seaborn library:--------
# titanic=data('titanic')  this data set has been taken from pydataset
titanic2=sns.load_dataset('titanic')

#--distplot for age by sex wise----
g=sns.FacetGrid(data=titanic2, col='sex')
g.map(sns.distplot,'age')

#----Here we are creating visualization on tips dataset:--------
tips=sns.load_dataset('tips')

g=sns.FacetGrid(data=tips, col='time')
g.map_dataframe(sns.scatterplot, x='total_bill',y='tip', hue='sex')

g=sns.FacetGrid(data=tips, col='sex')
g.map_dataframe(sns.scatterplot, x='total_bill',y='tip', hue='time')
g.add_legend()

g=sns.FacetGrid(data=tips, col='sex', row='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip',hue='smoker')
g.add_legend()

g=sns.FacetGrid(data=tips, col='sex', row='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip',hue='day')
g.set_axis_labels("Total bill","Tip")
g.add_legend()

g=sns.FacetGrid(data=tips, col='time')
g.map_dataframe(sns.scatterplot, x='total_bill', y='tip')
g.refline(y=tips['tip'].median())
g.set_axis_labels("Total bill","Tip")
g.add_legend()

#-----hist plot:----------------------------------------------------------------
g=sns.FacetGrid(data=tips, col='size')
g.map_dataframe(sns.histplot,'tip')

g=sns.FacetGrid(data=tips, col='day')
g.map_dataframe(sns.histplot,'tip', hue='sex')
g.add_legend()

g=sns.FacetGrid(data=tips, col='time', row='sex')
g.map_dataframe(sns.histplot,'tip')
g.add_legend()

#----jointplot:-------
g=sns.FacetGrid(data=tips, col='time',margin_titles=True)
g.map_dataframe(sns.jointplot, x='total_bill', y='tip',hue='sex')
g.tight_layout()
g.add_legend()

#-----linear plot:------

g=sns.lmplot(x='total_bill', y='tip', col='smoker', row='sex', data=tips)
#----advanced linear plot using xtics:------
g=sns.lmplot(x='total_bill', y='tip', col='smoker', row='sex', data=tips)
g.set_axis_labels("Zee","Farz").set(xlim=(0,60),ylim=(0,12), xticks=[10,30,50],yticks=[2,6,10])

#--- Pairplot:----
g=sns.pairplot(data=tips, hue='sex')
#---violinplot:----

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

#---Creating grou data using cut function available in python

df=pd.DataFrame({'number':np.random.randint(1,100,10)})
df['bins']=pd.cut(x=df['number'], bins=[1,20,40,60,80,100])
df
#-----creating a dataframe:------
Designation=['Asst Professor','Asoc. Professor','Professor']

Ass_sal=np.random.randint(150000,250000,100)
Aso_sal=np.random.randint(250000,350000,100)
Prof_sal=np.random.randint(350000,500000,100)


df=pd.DataFrame({'Ass Professor': np.random.randint(150000,250000,100),
'Associate Prof':np.random.randint(250000,350000,100),
'Professor':np.random.randint(350000,500000,100)})

df.head()
df_stacked=df.stack()

df_stacked.to_excel("Prof.xlsx")

df_stacked1=df_stacked.to_frame()

df_pivot= pd.pivot_table(df, index=df.columns,values=df.values)
df_ustacked= df.unstack()

df_melted= pd.melt(df,id_vars=['Ass Professor'],value_vars=['Ass Professor', 'Associate Prof', 'Professor'])
plt.pie(val)

#-----------------------





# %%
