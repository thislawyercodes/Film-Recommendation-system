import os
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#plt.show()  
display(grouped_data.groupby("title")['rating'].mean().sort_values(ascending=False).head()) 
             