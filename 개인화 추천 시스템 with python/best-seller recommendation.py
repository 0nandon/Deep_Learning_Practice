import pandas as pd
import numpy as np
from google.colab import files

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id') # 'user_id' 항목으로 인덱싱을 한다.
users.head() # .head()는 전체 데이터에서 위 5개를 추출한다.

i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 'umknown', 'Action',
          'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'war', 'western']
movies = pd.read_csv('/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies.set_index('movie_id')
movies.head()

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings.set_index('user_id')
ratings.head()

# Best-seller 추천
def recom_movie(n_items):
  movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
  recom_movies = movies.loc[movie_sort.index]
  recommendations = recom_movies['title']
  return recommendations

movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
recom_movie(5)

# RMSE 정확도 추천
def RMSE(y_true, y_pred):
  return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

rmse = []
for user in set(ratings.index):
  y_true = ratings.loc[user]['rating']
  y_pred = movie_mean[ratings.loc[user]['movie_id']]
  accuracy = RMSE(y_true, y_pred)
  rmse.append(accuracy)

print(np.mean(rmse))
