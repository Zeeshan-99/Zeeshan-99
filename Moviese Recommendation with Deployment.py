# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:11:19 2021

@author: Zeeshan Haleem
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# pwd
# os.chdir('C:\\Users\\Zeeshan Haleem\\Downloads\\Z-Jupyter')

#------------------------Preprocessing 1-----------------------------------------------
data= pd.read_csv('movies_metadata.csv')
# data.columns
data= data.loc[:,['director_name','actor_1_name', 'actor_2_name', 'actor_3_name', 'genres','title']]
data= data.rename(columns={'title':'movie_title'})

data['actor_1_name']=data['actor_1_name'].replace(np.nan,'unknown')
data['actor_2_name']=data['actor_2_name'].replace(np.nan,'unknown')
data['actor_3_name']=data['actor_3_name'].replace(np.nan,'unknown')
data['director_name']=data['director_name'].replace(np.nan,'unknown')

data['genres']=data['genres'].replace('|', ' ')
data['movie_title']=data['movie_title'].str.lower()

data.to_csv('data.csv', index=False)


#------------------------------------------------------------------------------------------
meta=pd.read_csv('movies_metadata.csv')
credits= pd.read_csv('credits.csv')


meta.columns


meta['release_date']= pd.to_datetime(meta['release_date'], errors='coerce') 
meta['year']= meta['release_date'].dt.year

meta['year'].value_counts().sort_index()

new_meta= meta.loc[meta.year==2017,['genres','id','title','year']]
new_meta.info()

new_meta['id']= new_meta['id'].astype(int)
credits.info()

data= pd.merge(new_meta, credits, on='id')

pd.set_option('display.max_colwidth',75)

data['genres'].dtypes


#-----Here i am making list of diction val -----------------------------
import ast

data['genres']= data['genres'].map(lambda x: ast.literal_eval(x))
data['cast']= data['cast'].map(lambda x: ast.literal_eval(x))
data['crew']= data['crew'].map(lambda x: ast.literal_eval(x))


# ---------------writing function---------------------------------
 
def make_genresList(x):
    gen=[]
    st=" "
    for i in x:
        if i.get('name')=='Science Fiction':
            scifi='Sci-Fi'
            gen.append(scifi)
        else:
            gen.append(i.get('name'))
    if gen==[]:
        return np.NaN
    else:
        return (st.join(gen))

data['genres_list']= data['genres'].map( lambda x: make_genresList(x))                

# print(data['genres_list'])

#  extracting actor names from 'cast' column and them making seperate actor column

def get_actor1(x):
    casts=[]
    for i in x:
        casts.append(i.get('name'))
    if casts==[]:
        return np.NaN
    else:
        return (casts[0])
    
data['actor_1_name']= data['cast'].map(lambda x: get_actor1(x))    
    


def get_actor2(x):
    casts=[]
    for i in x:
        casts.append(i.get('name'))
    if casts==[] or len(casts)<=1:
        return np.NaN
    else:
        return (casts[1])
    
            

data['actor_2_name']= data['cast'].map(lambda x: get_actor2(x))

#--------------actor 3-------------------------------------------------------------

def get_actor3(x):
    casts=[]
    for i in x:
        casts.append(i.get('name'))
    if casts==[] or len(casts)<=2:
        return np.NaN
    else:
        return (casts[2])
    
  

data['actor_3_name']= data['cast'].map(lambda x: get_actor3(x))


#-----------getting director name--------------------------------------------------------------

def get_directors(x):
    dt=[]
    st=" "
    for i in x:
        if i.get('job')=='Director':
            dt.append(i.get('name'))
        if dt==[]:
            return np.NaN
        else:
            return (st.join(dt))
        
data['director_name']= data['crew'].map(lambda x: get_directors(x))

data.columns

movie= data.loc[:,['director_name','actor_1_name', 'actor_2_name', 'actor_3_name','genres_list','title']]

movie.isna().sum()
movie= movie.dropna(how='any')

#----------------------------------------------------------------------   -------------------------------------------------     
    
movie= movie.rename(columns={'genres_list':'genres'})
movie= movie.rename(columns={'title':'movie_title'})    

movie['movie_title']= movie['movie_title'].str.lower()

movie['comb']= movie['actor_1_name']+ ' '+movie['actor_2_name']+' '+movie['actor_3_name']+' '+movie['director_name']+' '+movie['genres']

old= pd.read_csv('data.csv')

new= old.append(movie)

new.drop_duplicates(subset='movie_title', keep='last', inplace=True)
#---------------------------------
#------------Preprocessing 3 and extracting dataset for 2018 from wikipedia-----------------------------------------------------------


link="https://en.wikipedia.org/wiki/List_of_American_films_of_2018"
df1=pd.read_html(link,header=0)[2]
df2=pd.read_html(link,header=0)[3] 
df3=pd.read_html(link,header=0)[4]
df4=pd.read_html(link,header=0)[5]


df=df1.append(df2.append(df3.append(df4, ignore_index=True),ignore_index=True),ignore_index=True)

#-----------Extracting data using api keys on TMDb------------------------------------------

from tmdbv3api import TMDb
import json
import requests
tmdb=TMDb()
tmdb.api_key='3ed0a4625e09f24691268eb4515c9237'              # my api id

#------------------------------------------------------------------------------------


from tmdbv3api import Movie
tmdb_movie= Movie()

def get_genre(x):
    genres=[]
    result= tmdb_movie.search(x)
    movie_id= result[0].id
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id, tmdb.api_key)) 
    data_json = response.json()
    if data_json['genres']:
        genre_str = " "
        for i in range(0,len(data_json['genres'])):
            genres.append(data_json['genes'][i]['name'])
        return genre_str.join(genres)
    else:
        np.NAN
        
#-----------------------------------------------------------------      -----------------------------------  

df['Genre']= df['Title'].map(lambda x: get_genre(str(x)))        
        
        
    
    
    
    
    





















    
    
    
















