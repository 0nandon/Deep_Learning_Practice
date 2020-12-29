from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Activation, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() # 텐서플로의  MNIST 데이터 세트를 불러온다.
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify = y_train_all, test_size = 0.2, random_state = 42)

x_train = x_train / 255 # 표준화 진행
x_val = x_val / 255 # 표준화 진행

x_train = x_train.reshape(-1, 784) # x_train 데이터의 행렬 구조를 바꾼다. 
x_val = x_val.reshape(-1, 784) # y_train 데이터의 행렬 구조를 바꾼다.

y_train_encoded = tf.keras.utils.to_categorical(y_train) # 텐서플로우로 원-핫 인코딩
y_val_encoded = tf.keras.utils.to_categorical(y_val) # 텐서플로우로 원-핫 인코딩

image = Input(shape = (784,))
R = Dense(256, kernel_regularizer = regularizers.l2(0.01))(image)
R = Activation('sigmoid')(R)
R = Dropout(0.3)(R)
R = Dense(10)(image)
R = Activation('softmax')(R)
model = Model(inputs = [image], outputs = R)
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])
history = model.fit(x_train, y_train_encoded, epochs = 40, validation_data = (x_val, y_val_encoded))

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

# evaluate 메서드를 사용하면 손실값과 metrics 매개변수에 추가한 측정 지표를 계산하여 반환한다.
loss, accuracy = model.evaluate(x_val, y_val_encoded, verbose = 0) 
print(accuracy)
