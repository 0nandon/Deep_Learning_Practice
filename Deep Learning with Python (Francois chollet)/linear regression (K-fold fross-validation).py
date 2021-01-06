import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

def build_model():
  input = Input(shape = (train_data.shape[1],))
  R = Dense(64)(input)
  R = Activation('relu')(R)
  R = Dense(64)(R)
  R = Activation('relu')(R)
  R = Dense(1)(R)

  model = Model(inputs = [input], outputs = R)
  model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

  return model

for i in range(k):
  print('처리 중인 폴드 #', i)
  val_index = list(range(i * num_val_samples, (i+1) * num_val_samples))
  train_index = list(range(i * num_val_samples)) + list(range((i+1) * num_val_samples))

  partial_val_data_x = train_data[val_index]
  partial_val_data_y = train_targets[val_index]

  partial_train_data_x = train_data[train_index]
  partial_train_data_y = train_targets[train_index]

  model = build_model()
  model.fit(partial_train_data_x, partial_train_data_y,
            epochs = num_epochs, batch_size = 1,
            validation_data = (partial_val_data_x, partial_val_data_y))
  
  val_mse, val_mae = model.evaluate(partial_val_data_x, partial_val_data_y)
  all_scores.append(val_mae)

np.mean(all_scores) 
