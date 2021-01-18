import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16

# imagenet은 weights 모델을 초기화할 가중치 체크포인트를 지정
# include_top은 네트워크의 최상위층 완전 분류기를 포함할지, 안할지를 지정
conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))

base_dir = '/content/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size=20

# VGG16 (사전 훈련된 합성곱 층)을 활용해서 이미지의 특성을 추출한 후, 데이터 셋을 리턴한다.
def extract_features(directory, sample_count):
  features = np.zeros(shape=(sample_count, 4, 4, 512))
  labels = np.zeros(shape=(sample_count))
  generator = datagen.flow_from_directory(
      directory,
      target_size = (150, 150),
      batch_size = batch_size,
      class_mode = 'binary'
  )

  i=0
  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    features[i * batch_size : (i+1) * batch_size] = features_batch
    labels[i * batch_size : (i+1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= sample_count:
      break
  
  return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 분리 층에 들어가기 전에 데이터를 일렬로 펼친다. (Flatten() 역할)
train_features = train_features.reshape(2000, -1)
validation_features = validation_features.reshape(1000, -1)
test_features = test_features.reshape(1000, -1)

input = Input(shape = (4 * 4 * 512,))
R = Dense(256)(input)
R = Activation('relu')(R)
R = Dropout(0.5)(R)
R = Dense(1)(R)
R = Activation('sigmoid')(R)

model = Model(inputs = [input], outputs = R)
model.compile(optimizer = RMSprop(lr=2e-5),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(train_features, train_labels,
                    epochs = 30,
                    batch_size = 20,
                    validation_data = (validation_features, validation_labels)할

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bp', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bp', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
