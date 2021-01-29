import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/content/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int) # timestamp 제거

TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state = 1)
cut_off = int(len(ratings) * 0.75)
ratings_train = ratings[:cut_off]
ratings_test = ratings[cut_off:]

K = 200
mu = ratings_train['rating'].mean()
M = ratings['user_id'].max() + 1
N = ratings['movie_id'].max() + 1

def RMSE(y_true, y_pred):
  return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

user = Input(shape = (1,))
item = Input(shape = (1,))
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)

R = layers.dot([P_embedding, Q_embedding], axes = 2)
R = layers.add([R, user_bias, item_bias])
R = Flatten()(R)

model = Model(inputs = [user, item], outputs = R)
model.compile(
    loss = RMSE,
    optimizer = SGD(),
    metrics = [RMSE]
)

results  = model.fit(
    x = [ratings_train['user_id'].values, ratings_train['movie_id'].values],
    y = ratings_train['rating'].values - mu,
    epochs = 60,
    batch_size = 256,
    validation_data = (
        [ratings_test['user_id'].values, ratings_test['movie_id'].values],
        ratings_test['rating'].values - mu
    )
)

# 결과값 시각화 하기
plt.plot(results.history['RMSE'], label = "Train RMSE")
plt.plot(results.history['val_RMSE'], label = "Test RMSE")
plt.legend()
plt.show()

# 테스트 하기
def RMSE2(y_true, y_pred):
  return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

user_ids = ratings['user_id'].values
movie_ids = ratings['movie_id'].values
y_pred = model.predict([user_ids, movie_ids]) + mu
y_pred = y_pred.reshape(-1,)
y_true = ratings['rating'].values

print(RMSE2(y_true, y_pred))
