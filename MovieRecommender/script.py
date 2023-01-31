import os
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

os.chdir('/home/akal/Documents/DataScience/MovieLens')

rating_data = pd.read_csv("rating.csv")
movie_data = pd.read_csv("movie.csv")
data_merge = pd.merge(rating_data,movie_data,on='movieId')

grouped_data = pd.DataFrame(data_merge.groupby('title')['rating'].mean().round())
grouped_data['total ratings']=pd.DataFrame(data_merge.groupby('title')['rating'].count())

plt.figure(figsize=(10,5))
bar_plt=plt.bar(grouped_data['rating'],grouped_data['total ratings'],color="green")
plt.figure(figsize=(10,4))  

bar_plt=plt.subplot()
bar_plt.bar(grouped_data.head(10).index,grouped_data['total ratings'].head(10),color='purple')
bar_plt.set_xticklabels(grouped_data.index,rotation=90,fontsize="10",horizontalalignment='right')
bar_plt.set_title("Total Reviews for Different Movies") 
display(grouped_data.groupby("title")['rating'].mean().sort_values(ascending=False).head()) 
 
 #collaboratice filtering using the filtering algo            
data= Dataset.load_builtin('ml-100k')
trainset,testset=train_test_split(data,test_size=.15)
filtering_algo=KNNWithMeans(k=50,sim_options={
    "name":"pearson_baseline","user_based":True
})
filtering_algo.fit(trainset)

#estimating biases
uid=str(196)
lid=str(300)
prediction_result=filtering_algo.predict(uid,lid,verbose=True)
test_prediction=filtering_algo.test(testset)
test_prediction