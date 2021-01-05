import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

np.random.seed(42)
random_index = np.random.permutation(25000)

x_train = train_data[random_index][:20000]
y_train = train_labels[random_index][:20000]
x_val = train_data[random_index][20000:]
y_val = train_labels[random_index][20000:]

def vectorize_sequences(sequences, dimension = 10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequences in enumerate(sequences):
    results[i, sequences] = 1.
  
  return results

x_train = vectorize_sequences(x_train)
x_val = vectorize_sequences(x_val)

y_train = np.array(y_train).astype('float32')
y_val = np.array(y_val).astype('float32')

test_data = vectorize_sequences(test_data)
test_labels = np.array(test_labels).astype('float32')

words = Input(shape = (10000,))
R = Dense(16)(words)
R = Activation('relu')(R)
R = Dropout(0.3)(R)
R = Dense(16)(R)
R = Activation('relu')(R)
R = Dropout(0.3)(R)
R = Dense(1)(R)
R = Activation('sigmoid')(R)

adam = Adamax(lr = 0.01)
model = Model(inputs = [words], outputs = R)
model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 20, batch_size = 512, validation_data = (x_val, y_val))

results = model.evaluate(np.array(test_data), np.array(test_labels))

print(results)

plt.plot(history.history['loss'], label = 'Training loss')
plt.plot(history.history['val_loss'], label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
