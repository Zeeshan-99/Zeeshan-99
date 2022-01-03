import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ## Helper function. Use them when needed ####

def get_title_from_index(index):
	return df[df.index==index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title==title]["index"].values[0]

# #----------Data Cleaning first -----------------------------------

df= pd.read_csv('movie_recomm_dataset.csv')
print(df.info())
print(df.isnull().sum())

# # df= df.fillna(0)


# #---------------select features------------
features= ['keywords','cast','genres','director']

#-----------------------------------------------
for feature in features:
	df[feature]= df[feature].fillna('')
	
# #------- Create a column in DF which combine all selected feature

def combine_feature(row):
	try:
		return row['keywords']+ " "+row['cast']+" "+row['genres']+" "+ row['director']

	except: 	
	      print ("Error :", row)


df['combine_features']= df.apply(combine_feature, axis=1)

print (df['combine_features'].head())


# #----------Creating count Matrix from this new combined column


cv= CountVectorizer()
count_matrix= cv.fit_transform(df['combine_features'])

# # Compute the cosine similarity based on the count_matrix

cosine_sim=cosine_similarity(count_matrix)
movie_user_likes="Avatar"

# #------Get index of this movie from its title
movie_index= get_index_from_title(movie_user_likes)

# #------Get a list of similar movies in descending order of similarity score
similar_movies= list(enumerate(cosine_sim[movie_index]))
# print( similar_movies)
sorted_similar_movies= sorted(similar_movies, key= lambda x:x[1], reverse=True)

# print( sorted_similar_movies)

# #----Print titles of first 50 movies----------

i=0
for movie in sorted_similar_movies:
	print (get_title_from_index(movie[0]))
	i= i+1
	if i>50:
		break


