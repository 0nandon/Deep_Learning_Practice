import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify = y_train_all, test_size = 0.2, random_state = 42)

x_train = (x_train / 255).astype('float32')
x_val = (x_val / 255).astype('float32')

x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784)

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val)

class MultiClassNetwork:
  def __init__(self, units = 10, batch_size=32, learning_rate = 0.1, l1 = 0, l2 = 0):
    self.units = units
    self.batch_size = batch_size
    self.lr = learning_rate
    self.w1 = None
    self.b1 = None
    self.w2 = None
    self.b2 = None
    self.a1 = None
    self.losses = []
    self.val_losses = []
    self.lr = learning_rate
    self.l1 = l1
    self.l2 = l2

  def layer(self, x, w, b):
    z1 = tf.matmul(x, w) + b
    a = tf.sigmoid(z1)
    a_dropout = tf.nn.dropout(a, 0.3)
    return a_dropout

  def forpass(self, x):
    self.a1 = self.layer(x, self.w1, self.b1)
    z2 = tf.matmul(self.a1, self.w2) + self.b2
    return z2
  
  def init_weights(self, n_features, n_classes):
    g = tf.initializers.glorot_uniform()
    self.w1 = tf.Variable(g((n_features, self.units)), dtype = tf.float32)
    self.b1 = tf.Variable(tf.zeros(self.units), dtype = tf.float32)
    self.w2 = tf.Variable(g((self.units, n_classes)), dtype = tf.float32)
    self.b2 = tf.Variable(tf.zeros(n_classes), dtype = tf.float32)
  
  def training(self, x, y):
    m = len(x)
    with tf.GradientTape() as tape: # 자동 미분 기능을 사용하기 위해 with 블럭으로 감싸준다.
      loss = self.get_loss(x, y)
    weights_list = [self.w1, self.b1, self.w2, self.b2]
    grads = tape.gradient(loss, weights_list)
    self.optimizer.apply_gradients(zip(grads, weights_list)) # 가중치를 업데이트 한다.

  def fit(self, x, y, epochs=100, x_val = None, y_val = None):
    self.init_weights(x.shape[1], y.shape[1]) # 가중치 초기화
    self.optimizer = tf.optimizers.Adam(learning_rate = self.lr) # 경사하강법 optmizer 생성

    for i in range(epochs): 
      print('에포크', i, end = ' ') # 진행률 확인
      batch_losses = []
      for x_batch, y_batch in self.gen_batch(x, y):
         print('.', end = '') # 진행률 확인
         self.training(x_batch, y_batch)
         batch_losses.append(self.get_loss(x_batch, y_batch).numpy())
      print()
      self.losses.append(np.mean(batch_losses))
      self.val_losses.append(self.get_loss(x_val, y_val).numpy())
      print(self.losses[i], self.val_losses[i])
  
  def gen_batch(self, x, y):
    length = len(x)
    bins = length // self.batch_size
    if length % self.batch_size:
      bins += 1
    indexes = np.random.permutation(np.arange(len(x)))
    x = x[indexes]
    y = y[indexes]
    for i in range(bins):
      start = self.batch_size * i
      end = self.batch_size * (i + 1)
      yield x[start:end], y[start:end]
  
  def predict(self, x):
    z = self.forpass(x)
    return np.argmax(z.numpy(), axis = 1) # 가장 큰 값의 인덱스를 반환한다.
  
  def score(self, x, y):
    return np.mean(self.predict(x) == np.argmax(y, axis = 1))
  
  def get_loss(self, x, y):
    z = self.forpass(x)
    # z = tf.clip_by_value(z, 1e-10, 1-12-10)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
    return loss

fc = MultiClassNetwork(units = 100, learning_rate = 0.01)
fc.fit(x_train, y_train_encoded, x_val = x_val, y_val = y_val_encoded, epochs = 20) # 학습 수행

# 학습 결과를 그래프로 나타낸다.
plt.plot(fc.losses)
plt.plot(fc.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()

fc.score(x_val, y_val_encoded)
