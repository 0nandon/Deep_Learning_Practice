import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, GRU
from tensorflow.keras.optimizers import RMSprop

data_dir = './datasets/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
  values = [float(x) for x in line.split(',')[1:]]
  float_data[i, :] = values

# float_data 데이터를 표준화한다.
mean = float_data[:200000].mean(axis = 0)
float_data -= mean
std = float_data[:200000].std(axis = 0)
float_data /= std

# float_data 배열을 받아 과거 데이터의 배치와 미래 타깃 온도를 추출하는 파이썬 제너레이터를 만든다.
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
  if max_index is None:
    max_index = len(data)-delay-1
  i = min_index + looback

  # 무한으로 데이터 배치를 만들어낸다.
  while 1:
    if shuffle: # 샘플을 섞는다.
      rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
    else:
      if i + batch_size >= max_index:
        i = min_index + lookback
      rows = np.arrange(i, min(i + batch_size, max_index))
      i += len(rows)

    samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
    targets = np.zeros((len(rows),))

    for j, row in enumerate(rows):
      indices = range(rows[j], rows[j] - lookback, step)
      samples[j] = data[indices]
      targets[j] = data[rows[j] + delay][1]
    
    yield samples, targets

# generator 함수를 활용하여 훈련용, 검증용, 테스트용으로 3개의 generator를 만든다.
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay, 
    min_index=0, max_index=200000, shuffle = True, 
    step=step, batch_size=batch_size
    )

val_gen = generator(float_data, lookback=lookback, delay=delay, 
    min_index=200001, max_index=300000, 
    step=step, batch_size=batch_size
    )

test_gen = generator(float_data, lookback=lookback, delay=delay, 
    min_index=300001, max_index=None, 
    step=step, batch_size=batch_size
    )

# 상식 수준의 기준점
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

def evaluate_naive_method():
  batch_maes = []
  for step in range(val_steps):
    samples, targets = next(val_gen)
    pred = samples[:, -1, 1]
    mae = np.mean(np.abs(pred - targets))
    batch_maes.append(mae)
  print(np.mean(batch_maes))

# 단순한 완전연결층을 사용해본다. - 데이터의 시퀀스를 고려하지 않았기 때문에 정확도가 낮다.
input = Input(shape = (lookback // step, float_data.shape[-1]))
R = Dense(32)(R)
R = Activation('relu')(R)
R = Dense(1)(R)
model = Model(inputs = [input], output = R)

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation liss')
plt.legend()

plt.show()

# GRU 신경망을 사용하여, 데이터 시퀀스를 고려한 학습을 진행해본다.
input = Input(shape = (lookback // step, float_data.shape[-1]))
R = GRU(32)(R)
R = Dense(1)(R)
model = Model(inputs = [input], output = R)

model.compile(optimizer = RMSprop(), loss = 'mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch = 500,
                              epochs = 20,
                              validation_data = val_gen,
                              validation_steps = val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation liss')
plt.legend()

plt.show()

