from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, SimpleRNN, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers

(x_train_all,  y_train_all), (x_test, y_test) = imdb.load_data(skip_top = 20, num_words = 1000)

for i in range(len(x_train_all)):
  x_train_all[i] = [w for w in x_train_all[i] if w > 2]

np.random.seed(42)
random_index = np.random.permutation(25000)

x_train = x_train_all[random_index[:20000]]
y_train = y_train_all[random_index[:20000]]
x_val = x_train_all[random_index[20000:]]
y_val = y_train_all[random_index[20000:]]

maxlen = 100
x_train_seq = sequence.pad_sequences(x_train, maxlen)
x_val_seq = sequence.pad_sequences(x_val, maxlen)

print(x_train_seq.shape)
words = Input(shape = (100,))
model_ebd = Embedding(1000, 32, embeddings_regularizer=l2())(words)
R = SimpleRNN(8)(model_ebd)
R = Dense(1)(R)
R = Activation('sigmoid')(R)

model = Model(inputs = [words], outputs = R)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(x_train_seq, y_train, epochs = 10, batch_size = 32, validation_data = (x_val_seq, y_val))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

loss, accuracy = model.evaluate(x_val_seq, y_val, verbose = 0)
print(accuracy)
