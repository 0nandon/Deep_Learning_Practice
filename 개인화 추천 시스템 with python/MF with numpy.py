
# MF(Matrix Factorization)

# MF 알고리즘은 CF 알고리즘과 달리 model-based 알고리즘이다.
# 모델을 구축하기만 하면 결과값을 빨리 도출해 내지만, 모델을 생성하는 과정에서 많은 계산값이 소모된다.
# 전체 사용자의 평가 패턴을 기반으로 모델을 구성하기 때문에 데이터가 가지고 있는 weak signal을 더 잘 잡아내는 장점이 있다.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('/content/u.data', sep='\t', names=r_cols, encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int) # timestamp 제거

class NEW_MF():
  def __init__(self, ratings, K, alpha, beta, iterations, verbose = True):
    self.R = np.array(ratings)
    
    # 사용자 아이디와 아이템 아이디가 연속값이 아닐 수 있으므로, self.R을 numpy 배열로 
    # 변환하면서 중간이 비어있는 실제 아이디와 R의 인덱스가 일치하지 않게 된다.
    
    item_id_index = [] # 현재 아이템의 아이디와 인덱스를 저장
    index_item_id = []
    for i, one_id in enumerate(ratings):
      item_id_index.append([one_id, i])
      index_item_id.append([i, one_id])
    self.item_id_index = dict(item_id_index)
    self.index_item_id = dict(index_item_id)

    user_id_index = [] # 현재 유저의 아이디의 인덱스를 저장
    index_user_id = []
    for i, one_id in enumerate(ratings.T):
      user_id_index.append([one_id, i])
      index_user_id.append([i, one_id])
    self.user_id_index = dict(user_id_index)
    self.index_user_id = dict(index_user_id)

    self.num_users, self.num_items = self.R.shape
    self.k = K # 잠재요인(latent factor)의 수
    self.alpha = alpha # 학습률
    self.beta = beta # 규제 정도
    self.iterations = iterations # epoch 수
    self.verbose = verbose # 중간 학습과정을 출력할 것인가
  
  # test case를 전처리 한다.
  def set_test(self, ratings_test):
    test_set = []
    for i in range(len(ratings_test)):
      x = self.user_id_index[ratings_test.iloc[i, 0]]
      y = self.item_id_index[ratings_test.iloc[i, 1]]
      z = ratings_test.iloc[i, 2]
      test_set.append([x, y, z])
      self.R[x, y] = 0
    self.test_set = test_set
    return test_set
  
  def test_rmse(self):
    error = 0
    for one_set in self.test_set:
      predicted = self.get_prediction(one_set[0], one_set[1])
      error += pow((one_set[2] - predicted), 2)
    return np.sqrt(error/len(self.test_set))
  
  def rmse(self):
    xs, ys = self.R.nonzero() # R에서 평점이 있는(0이 아닌) 요소의 인덱스를 가져온다.
    self.predictions = []
    self.errors = []
    for x, y in zip(xs, ys):
      prediction = self.get_prediction(x, y)
      self.predictions.append(prediction)
      self.errors.append(self.R[x, y] - prediction)
    self.predictions = np.array(self.predictions)
    self.errors = np.array(self.errors)
    return np.sqrt(np.mean(self.errors**2))
  
  # 학습 
  def test(self):
    self.P = np.random.normal(scale = 1./self.k, size = (self.num_users, self.k))
    self.Q = np.random.normal(scale = 1./self.k, size = (self.num_items, self.k))

    self.b_u = np.zeros(self.num_users)
    self.b_d = np.zeros(self.num_items)
    self.b = np.mean(self.R[self.R.nonzero()])

    rows, columns = self.R.nonzero()
    self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]

    training_process = []
    for i in range(self.iterations):
      np.random.shuffle(self.samples)
      self.sgd()
      rmse1 = self.rmse()
      rmse2 = self.test_rmse()
      training_process.append((i+1, rmse1, rmse2))
      if self.verbose:
        if (i+1) % 10 == 0:
          print("Iteration : %d ; Train RMSE = %.4f ; Test RMSE = %.4f" % (i+1, rmse1, rmse2))
    return training_process
  
  # 경사 하강법 시행
  def sgd(self):
    for i, j, r in self.samples:
      prediction = self.get_prediction(i, j)
      e = (r - prediction)

      self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
      self.b_d[i] += self.alpha * (e - self.beta * self.b_d[j])

      self.P[i,:] += self.alpha * (e * self.Q[j,:] - self.beta * self.P[i,:])
      self.Q[j,:] += self.alpha * (e * self.P[i,:] - self.beta * self.Q[j,:])
  
  def get_prediction(self, i, j):
    prediction =  self.b + self.b_u[i] + self.b_d[i] + np.dot(self.P[i,:], self.Q[j,:].T)
    return prediction

  def get_one_prediction(self, user_id, item_id):
    return self.get_prediction(self.user_id_index[user_id], self.item_id_index[item_id])
  
  def full_prediction(self):
    return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + np.dot(self.P, self.Q.T)
  
R_temp = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
mf = NEW_MF(R_temp, K=30, alpha=0.001, beta=0.02, iterations=100, verbose=True)
test_set = mf.set_test(ratings_test)
result = mf.test() # 학습 시행
