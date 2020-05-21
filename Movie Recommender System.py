# Importing the Necessary Libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1   Read Dataset('csv')file
dataset = pd.read_csv('movie_dataset.csv');

# Step 2 Selecting the features
features = ['keywords','cast','genres','director']
# Step 3 Create a dataframe in dataset which combines all selected features
for feature in features:
    dataset[feature] = dataset[feature].fillna(' ')
    
def combine_features(row):
    try:
        return row['keywords'] + " "+ row['cast'] + " "+ row['genres'] + " "+ row['director'];
    except:
        print('Error in the row of the dataset',row)           

dataset['comb_feature'] = dataset.apply(combine_features,axis=1)
# print('Combined Features:',dataset['comb_feature'].head())
# Step 4 Create a count_matrix from this combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(dataset['comb_feature'])
# Step 5 Compute the cosine similarity based on the count_matrix
cosine_sim = cosine_similarity(count_matrix)

#################################################################################
## ⏺⏺➡➡**Enter Input which type of movie do u wanna see⬅⬅⏺⏺**
movie_user_like =  'Resident Evil: Afterlife'
#######################################################################################################
#Step 6 Get index of the movie from the given title
    #Necessary Functions
def get_title_from_index(index):
    return dataset[dataset.index == index]['title'].values[0]
def get_index_from_title(title):    
    return dataset[dataset.title == title]['index'].values[0]

movie_index = get_index_from_title(movie_user_like);

similar_movies = list(enumerate(cosine_sim[movie_index]))

# Step 7 Get a list of similar Movies in descending order of similarity
sorted_sim_mov = sorted(similar_movies,key = lambda x:x[1],reverse = True)

# Step 8 Print title of first 50 Similar Movies
i=0
for element in sorted_sim_mov:
    print(get_title_from_index(element[0]))
    i = i+1
    if(i>50):
        break
    