import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, BatchNormalization, Concatenate, Dense, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax
from tensorflow.keras.models import model_from_json

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/content/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int) # timestamp 제거

K = 200
mu = ratings_train['rating'].mean()
M = ratings.user_id.max() + 1
N = ratings.movie_id.max() + 1

# train/test 분리 MF 알고리즘
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state = 1)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings[:cutoff]
ratings_test = ratings[cutoff:]

def RMSE(y_true, y_pred):
  return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

user = Input(shape = (1,))
item = Input(shape = (1,))
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)

P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)

R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])
R = Dense(2048)(R)
R = BatchNormalization()(R)
R = Activation('linear')(R)
R = Dense(256)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

P_embedding_MF = Embedding(M, K, embeddings_regularizer=l2())(user)
Q_embedding_MF = Embedding(N, K, embeddings_regularizer=l2())(item)
user_bias_MF = Embedding(M, 1, embeddings_regularizer = l2())(user)
item_bias_MF = Embedding(N, 1, embeddings_regularizer = l2())(item)
R_MF = layers.dot([P_embedding_MF, Q_embedding_MF], axes = 2)
R_MF = layers.add([R_MF, user_bias_MF, item_bias_MF])
R_MF = Flatten()(R_MF)

sgd = SGD(lr = 0.001)

model = Model(inputs = [user, item], outputs = 0.8 * R + 0.2 * R_MF)
model.compile(loss = RMSE, optimizer = sgd, metrics = [RMSE])
  
result = model.fit(x = [ratings_train['user_id'].values, ratings_train['movie_id'].values],
                   y = ratings_train['rating'].values - mu,
                   epochs = 60,
                   batch_size = 256,
                   validation_data = (
                       [ratings_test['user_id'].values, ratings_test['movie_id'].values],
                       ratings_test['rating'].values - mu
                   ))
