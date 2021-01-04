import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

(x_train_all, y_train_all), (x_test, y_test) = imdb.load_data(skip_top=20, num_words=100)

for i in range(len(x_train_all)):
  x_train_all[i] = [w for w in x_train_all[i] if w > 2]

word_to_index = imdb.get_word_index()
index_to_word = { word_to_index[k]: k for k in word_to_index }

np.random.seed(42)
random_index = np.random.permutation(25000)
x_train = x_train_all[random_index[:20000]]
x_val = x_train_all[random_index[20000:]]
y_train = y_train_all[random_index[:20000]]
y_val = y_train_all[random_index[20000:]]

# 샘플의 길이를 맞추어 준다.
maxlen = 100
x_train_seq = sequence.pad_sequences(x_train, maxlen=maxlen)
x_val_seq = sequence.pad_sequences(x_val, maxlen=maxlen)

x_train_onehot = to_categorical(x_train_seq)
x_val_onehot = to_categorical(x_val_seq)

class LSTMNetwork:
  
  def __init__(self, n_cells=10, batch_size=32, learning_rate=0.1):
    self.n_cells = n_cells
    self.batch_size = batch_size
    self.w1fx = None
    self.w1fh = None
    self.b1f = None
    self.w1ix = None
    self.w1ih = None
    self.b1i = None
    self.w1jx = None
    self.w1jh = None
    self.b1j = None
    self.w1ox = None
    self.w1oh = None
    self.b1o = None
    self.w2 = None
    self.b2 = None
    self.c = None
    self.h = None
    self.losses = []
    self.val_losses = []
    self.lr = learning_rate

  def forpass(self, x):
    self.h = [np.zeros((x.shape[0], self.n_cells))]
    self.c = [np.zeros((x.shape[0], self.n_cells))]

    seq = np.swapaxes(x, 0, 1)

    for x in seq:
      z1f = np.dot(x, self.w1fx) + np.dot(self.h[-1], self.w1fh) + self.b1f
      z1i = np.dot(x, self.w1ix) + np.dot(self.h[-1], self.w1ih) + self.b1i
      z1j = np.dot(x, self.w1jx) + np.dot(self.h[-1], self.w1jh) + self.b1j
      z1o = np.dot(x, self.w1ox) + np.dot(self.h[-1], self.w1oh) + self.b1o

      z1f_sigmoid = self.sigmoid(z1f)
      z1i_sigmoid = self.sigmoid(z1i)
      z1j_tanh = np.tanh(z1j)
      z1o_sigmoid = self.sigmoid(z1o)
      I = z1i_sigmoid * z1j_tanh
      F = self.c[-1] * z1f_sigmoid
      C = F + I
      h = np.tanh(C) * z1o_sigmoid

      self.h.append(h)
      self.c.append(C)
      z2 = np.dot(h, self.w2) + self.b2
    
    return z2, z1f_sigmoid, z1f, z1i_sigmoid, z1i, z1j_tanh, z1j, z1o_sigmoid
  
  def gradient_cf(self, n, z1f, z1f_sigmoid):
    r = z1f * (1 - z1f)
    result = np.zeros((z1f.shape[0], self.n_cells))
    for c in self.c[:-n][::-1]:
      result += c * r
      r *= z1f_sigmoid
    return result

  def gradient_ci(self, n, z1f_sigmoid, z1j_tanh, z1i):
    r = z1j_tanh * z1i * (1-z1i)
    result = np.zeros((z1i.shape[0], self.n_cells))
    for c in self.c[:-n][::-1]:
      result += r
      r *= z1f_sigmoid
    return result
  
  def gradient_cj(self, n, z1f_sigmoid, z1j, z1i_sigmoid):
    r = (1 - z1j**2) * z1i_sigmoid
    result = np.zeros((z1j.shape[0], self.n_cells))
    for c in self.c[:-n][::-1]:
      result += r
      r *= z1f_sigmoid
    return result

  def backprop(self, x, err, z1f_sigmoid, z1f, z1i_sigmoid, z1i, z1j_tanh, z1j, z1o_sigmoid):
    m = len(x)

    w2_grad = np.dot(self.h[-1].T, err) / m
    b2_grad = np.sum(err) / m

    seq = np.swapaxes(x, 0, 1)

    w1fh_grad = w1fx_grad = b1f_grad = 0
    w1ih_grad = w1ix_grad = b1i_grad = 0
    w1jh_grad = w1jx_grad = b1j_grad = 0
    w1oh_grad = w1ox_grad = b1o_grad = 0 

    err_to_cell_f = np.dot(err, self.w2.T) * z1o_sigmoid * (1 - self.c[-1] ** 2)
    err_to_cell_f *= (z1f_sigmoid * self.gradient_cf(1, z1f, z1f_sigmoid) + self.c[-1] * z1f * (1-z1f))
    err_to_cell_i = np.dot(err, self.w2.T) * z1o_sigmoid * (1 - self.c[-1] ** 2)
    err_to_cell_i *= (z1f_sigmoid * self.gradient_ci(1, z1f_sigmoid, z1j_tanh, z1i) + z1j_tanh * z1i * (1-z1i))
    err_to_cell_j = np.dot(err, self.w2.T) * z1o_sigmoid * (1 - self.c[-1] ** 2)
    err_to_cell_j *= (z1f_sigmoid * self.gradient_cj(1, z1f_sigmoid, z1j, z1i_sigmoid) + (1 - z1j**2) * z1i_sigmoid)
    err_to_cell_o = np.dot(err, self.w2.T) * np.tanh(self.c[-1]) * (1 - self.h[-1] ** 2)

    i = 2
    for x, c, h in zip(seq[::-1][:30], self.c[:-30][::-1], self.h[:-30][::-1]):
      w1fh_grad += np.dot(h.T, err_to_cell_f)
      w1fx_grad += np.dot(x.T, err_to_cell_f)
      b1f_grad += np.sum(err_to_cell_f, axis = 0)

      w1ih_grad += np.dot(h.T, err_to_cell_i)
      w1ix_grad += np.dot(x.T, err_to_cell_i)
      b1i_grad += np.sum(err_to_cell_i, axis = 0)

      w1jh_grad += np.dot(h.T, err_to_cell_j)
      w1jx_grad += np.dot(x.T, err_to_cell_j)
      b1j_grad += np.sum(err_to_cell_j, axis = 0)
      
      w1oh_grad += np.dot(h.T, err_to_cell_o)
      w1ox_grad += np.dot(x.T, err_to_cell_o)
      b1o_grad += np.sum(err_to_cell_o, axis = 0)

      err_to_cell_f = np.dot(err_to_cell_f, self.w1fh) * z1o_sigmoid * (1 - c**2)
      err_to_cell_f *= (z1f_sigmoid * self.gradient_cf(i, z1f, z1f_sigmoid) + c * z1f * (1-z1f))
      err_to_cell_i = np.dot(err_to_cell_i, self.w1ih) * z1o_sigmoid * (1 - c**2)
      err_to_cell_i *= (z1f_sigmoid * self.gradient_ci(i, z1f_sigmoid, z1j_tanh, z1i) + z1j_tanh * z1i * (1-z1i))
      err_to_cell_j = np.dot(err_to_cell_j, self.w1jh) * z1o_sigmoid * (1 - c**2)
      err_to_cell_j *= (z1f_sigmoid * self.gradient_cj(i, z1f_sigmoid, z1j, z1i_sigmoid) + (1 - z1j**2) * z1i_sigmoid)
      err_to_cell_o = np.dot(err_to_cell_o, self.w1oh) * np.tanh(c) * (1 - h**2)
      i += 1

    w1fh_grad /= m
    w1fx_grad /= m
    b1f_grad /= m 
    w1ih_grad /= m
    w1ix_grad /= m
    b1i_grad /= m
    w1jh_grad /= m
    w1jx_grad /= m
    b1j_grad /= m
    w1oh_grad /= m
    w1ox_grad /= m
    b1o_grad /= m

    return (w1fh_grad, w1fx_grad, b1f_grad, 
    w1ih_grad, w1ix_grad, b1i_grad,
    w1jh_grad, w1jx_grad, b1j_grad,
    w1oh_grad, w1ox_grad, b1o_grad,
    w2_grad, b2_grad)

  def sigmoid(self, z):
    a = 1 / (1 + np.exp(-z))
    return a

  def init_weights(self, n_features, n_classes):
    orth_init = tf.initializers.Orthogonal()
    glorot_init = tf.initializers.GlorotUniform()

    self.w1fh = orth_init((self.n_cells, self.n_cells)).numpy()
    self.w1fx = glorot_init((n_features, self.n_cells)).numpy()
    self.w1ih = orth_init((self.n_cells, self.n_cells)).numpy()
    self.w1ix = glorot_init((n_features, self.n_cells)).numpy()
    self.w1jh = orth_init((self.n_cells, self.n_cells)).numpy()
    self.w1jx = glorot_init((n_features, self.n_cells)).numpy()
    self.w1oh = orth_init((self.n_cells, self.n_cells)).numpy()
    self.w1ox = glorot_init((n_features, self.n_cells)).numpy()

    self.b1f = np.zeros(self.n_cells)
    self.b1i = np.zeros(self.n_cells)
    self.b1j = np.zeros(self.n_cells)
    self.b1o = np.zeros(self.n_cells)

    self.w2 = glorot_init((self.n_cells, n_classes)).numpy()
    self.b2 = np.zeros(n_classes)

  def fit(self, x, y, epochs = 100, x_val = None, y_val = None):
    y = y.reshape(-1, 1);
    y_val = y_val.reshape(-1, 1)
    np.random.seed(42)
    self.init_weights(x.shape[2], y.shape[1])

    for i in range(epochs):
      print('에포크', i, end=' ')
      batch_losses = []
      
      for x_batch, y_batch in self.gen_batch(x, y):
        print('.', end= ' ')
        a = self.training(x_batch, y_batch)
        a = np.clip(a, 1e-10, 1-1e-10)

        loss = np.mean(-(y_batch * np.log(a) + (1-y_batch) * np.log(1-a)))
        batch_losses.append(loss)
      print()
      self.losses.append(np.mean(batch_losses))
      self.update_val_loss(x_val, y_val)

  def gen_batch(self, x, y):
    length = len(x)
    bins = length // self.batch_size
    
    if length % self.batch_size:
      bins += 1
    
    indexes = np.random.permutation(np.arange(length))
    x = x[indexes]
    y = y[indexes]

    for i in range(bins):
      start = i * self.batch_size
      end = (i+1) * self.batch_size
      yield x[start:end], y[start:end]

  
  def training(self, x, y):
    m = len(x)
    z, z1f_sigmoid, z1f, z1i_sigmoid, z1i, z1j_tanh, z1j, z1o_sigmoid = self.forpass(x)
    a = self.sigmoid(z)
    err = -(y - a)

    (w1fh_grad, w1fx_grad, b1f_grad, 
    w1ih_grad, w1ix_grad, b1i_grad,
    w1jh_grad, w1jx_grad, b1j_grad,
    w1oh_grad, w1ox_grad, b1o_grad,
    w2_grad, b2_grad) = self.backprop(x, err, z1f_sigmoid, z1f, z1i_sigmoid, z1i, z1j_tanh, z1j, z1o_sigmoid)

    self.w1fh -= self.lr * w1fh_grad
    self.w1fx -= self.lr * w1fx_grad
    self.b1f -= self.lr * b1f_grad
    self.w1ih -= self.lr * w1ih_grad
    self.w1ix -= self.lr * w1ix_grad
    self.b1i -= self.lr * b1i_grad
    self.w1jh -= self.lr * w1jh_grad
    self.w1jx -= self.lr * w1jx_grad
    self.b1j -= self.lr * b1j_grad
    self.w1oh -= self.lr * w1oh_grad
    self.w1ox -= self.lr * w1ox_grad
    self.b1o -= self.lr * b1o_grad
    self.w2 -= self.lr * w2_grad
    self.b2 -= self.lr * b2_grad

    return a

  def predict(self, x):
    (z2, z1f_sigmoid, z1f, z1i_sigmoid, z1i, z1j_tanh, z1j, z1o_sigmoid) = self.forpass(x)
    return z2 > 0

  def score(self, x, y):
    return np.mean(self.predict(x) == y.reshape(-1, 1))

  def update_val_loss(self, x_val, y_val):
    (z2, z1f_sigmoid, z1f, z1i_sigmoid, z1i, z1j_tanh, z1j, z1o_sigmoid) = self.forpass(x_val)
    a = self.sigmoid(z2)
    a = self.training(x_val, y_val)
    a = np.clip(a, 1e-10, 1-1e-10)
    val_loss = np.mean(-(y_val * np.log(a) + (1-y_val) * np.log(1-a)))
    self.val_losses.append(val_loss)


lstm = LSTMNetwork(n_cells=8, batch_size=32, learning_rate = 0.01)
lstm.fit(x_train_onehot, y_train, epochs=20, x_val=x_val_onehot, y_val=y_val)

lstm.score(x_val_onehot, y_val)
