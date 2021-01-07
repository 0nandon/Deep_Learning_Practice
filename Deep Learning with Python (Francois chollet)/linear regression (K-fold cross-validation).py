import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []

def build_model():
  input = Input(shape = (train_data.shape[1],))
  R = Dense(64)(input)
  R = Activation('relu')(R)
  R = Dense(64)(R)
  R = Activation('relu')(R)
  R = Dense(1)(R)

  model = Model(inputs = [input], outputs = R)
  model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mean_absolute_error'])

  return model

for i in range(k):
  print('처리 중인 폴드 #', i)
  val_index = list(range(i * num_val_samples, (i+1) * num_val_samples))
  train_index = list(range(i * num_val_samples)) + list(range((i+1) * num_val_samples))

  
  partial_val_data_x = train_data[val_index]
  partial_val_data_y = train_targets[val_index]

  partial_train_data_x = train_data[train_index]
  partial_train_data_y = train_targets[train_index]

  train_mean = partial_train_data_x.mean(axis = 0)
  train_std = partial_train_data_x.std(axis = 0)

  partial_val_data_x -= train_mean
  partial_val_data_x /= train_mean

  partial_train_data_x -= train_mean
  partial_train_data_x /= train_mean

  model = build_model()
  history = model.fit(partial_train_data_x, partial_train_data_y,
            epochs = num_epochs, batch_size = 1,
            validation_data = (partial_val_data_x, partial_val_data_y))
  
  mae_history = history.history['val_mean_absolute_error']
  all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

def smooth_curve(points, factor = 0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      next = previous * factor + point * (1 - factor) # 그래프를 부드럽게 만들기 위해 지수가중이동
      smoothed_points.append(next)
    else:
      smoothed_points.append(point)

  return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[:10])

plt.plot(smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

