from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class ConvolutionNetwork:
  def __init__(self, n_kernels = 10, units = 10, batch_size = 32, learning_rate = 0.1, l1 = 0, l2 = 0):
    self.n_kernels = n_kernels
    self.kernel_size = 3
    self.optimizer = None
    self.conv_w = None
    self.conv_b = None
    self.units = units
    self.batch_size = batch_size
    self.w1 =None
    self.b1 = None
    self.w2 = None
    self.b2 = None
    self.a1 = None
    self.losses = []
    self.val_losses = []
    self.lr  = learning_rate
  
  def forpass(self, x):
    c_out = tf.nn.conv2d(x, self.conv_w, strides = 1, padding = 'SAME') + self.conv_b # 3*3 합성곱 연산 수행
    r_out = tf.nn.relu(c_out) # 렐루 함수로 활성화
    p_out = tf.nn.max_pool2d(r_out, ksize = 2, strides = 2, padding = 'VALID') # 최대 풀링 수행
    f_out = tf.reshape(p_out, [x.shape[0], -1])
    z1 = tf.matmul(f_out, self.w1) + self.b1 # 첫 번째 층의 선형식을 계산
    a1 = tf.nn.relu(z1) # 렐루 함수로 활성화
    z2 = tf.matmul(a1, self.w2) + self.b2 # 두 번째 층의 선형식을 계산
    return z2
  
  def init_weights(self, input_shape, n_classes):
    g = tf.initializers.glorot_uniform() # 초기화에 글로럿 함수를 활용
    self.conv_w = tf.Variable(g((3, 3, 1, self.n_kernels)))
    self.conv_b = tf.Variable(np.zeros(self.n_kernels), dtype = float)
    n_features = 14 * 14 * self.n_kernels
    self.w1 = tf.Variable(g((n_features, self.units))) # (특성 개수, 은닉층의 크기)
    self.b1 = tf.Variable(np.zeros(self.units), dtype = float)
    self.w2 = tf.Variable(g((self.units, n_classes))) # (은닉층의 크기, 클래스 개수)
    self.b2 = tf.Variable(np.zeros(n_classes), dtype = float)
  
  def fit(self, x, y, epochs=100, x_val = None, y_val = None):
    self.init_weights(x.shape, y.shape[1]) # 가중치 초기화
    self.optimizer = tf.optimizers.SGD(learning_rate = self.lr) # 경사하강법 optmizer 생성

    for i in range(epochs): 
      print('에포크', i, end = ' ') # 진행률 확인
      batch_losses = []
      for x_batch, y_batch in self.gen_batch(x, y):
         print('.', end = '') # 진행률 확인
         self.training(x_batch, y_batch)
         batch_losses.append(self.get_loss(x_batch, y_batch))
      print()
      self.losses.append(np.mean(batch_losses))
      self.val_losses.append(self.get_loss(x_val, y_val))
  
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

  def training(self, x, y):
    m = len(x)
    with tf.GradientTape() as tape: # 자동 미분 기능을 사용하기 위해 with 블럭으로 감싸준다.
      z = self.forpass(x)
      loss = tf.nn.softmax_cross_entropy_with_logits(y, z)
      loss = tf.reduce_mean(loss)
    weights_list = [self.conv_w, self.conv_b, self.w1, self.b1, self.w2, self.b2]
    grads = tape.gradient(loss, weights_list)
    self.optimizer.apply_gradients(zip(grads, weights_list)) # 가중치를 업데이트 한다.
  
  def predict(self, x):
    z = self.forpass(x)
    return np.argmax(z.numpy(), axis = 1) # 가장 큰 값의 인덱스를 반환한다.
  
  def score(self, x, y):
    return np.mean(self.predict(x) == np.argmax(y, axis = 1))
  
  def get_loss(self, x, y):
    z = self.forpass(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, z))
    return loss.numpy()

(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() # 텐서플로의  MNIST 데이터 세트를 불러온다.
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify = y_train_all, test_size = 0.2, random_state = 42)

x_train = x_train.reshape(-1, 28, 28, 1) # x_train 데이터의 행렬 구조를 바꾼다. 
x_val = x_val.reshape(-1, 28, 28, 1) # y_train 데이터의 행렬 구조를 바꾼다.

x_train = x_train / 255 # 표준화 진행
x_val = x_val / 255 # 표준화 진행

y_train_encoded = tf.keras.utils.to_categorical(y_train) # 텐서플로우로 원-핫 인코딩
y_val_encoded = tf.keras.utils.to_categorical(y_val) # 텐서플로우로 원-핫 인코딩

cn = ConvolutionNetwork(n_kernels = 10, units = 100, batch_size = 128, learning_rate = 0.01)
cn.fit(x_train, y_train_encoded, x_val = x_val, y_val = y_val_encoded, epochs = 20) # 학습 수행

plt.plot(cn.losses)
plt.plot(cn.val_losses)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.legend(['train_loss', 'val_loss'])
plt.show()

print(cn.score(x_val, y_val_encoded))
