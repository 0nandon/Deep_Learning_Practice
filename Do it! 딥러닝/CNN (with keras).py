from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input, Activation

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() # 텐서플로의  MNIST 데이터 세트를 불러온다.
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify = y_train_all, test_size = 0.2, random_state = 42)

x_train = x_train.reshape(-1, 28, 28, 1) # x_train 데이터의 행렬 구조를 바꾼다. 
x_val = x_val.reshape(-1, 28, 28, 1) # y_train 데이터의 행렬 구조를 바꾼다.

x_train = x_train / 255 # 표준화 진행
x_val = x_val / 255 # 표준화 진행

y_train_encoded = tf.keras.utils.to_categorical(y_train) # 텐서플로우로 원-핫 인코딩
y_val_encoded = tf.keras.utils.to_categorical(y_val) # 텐서플로우로 원-핫 인코딩

# 과대적합을 해결하기 위한 Dropout 시행
image = Input(shape = (28, 28, 1))
R = Conv2D(10, (3, 3), activation = 'relu', padding = 'same')(image)
R = MaxPooling2D((2, 2))(R)
R = Flatten()(R)

R = Dropout(0.5)(R) # 완전연결신경망으로 들어가기 전 Dropout을 추가해준다.
R = Dense(100)(R)
R = Activation('relu')(R)
R = Dense(10)(R)
R = Activation('softmax')(R)

conv2 = Model(inputs = [image], outputs = R)
conv2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = conv2.fit(x_train, y_train_encoded, epochs = 20, validation_data = (x_val, y_val_encoded))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()
