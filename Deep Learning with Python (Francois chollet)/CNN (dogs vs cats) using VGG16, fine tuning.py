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

# 미세 조정은 특성 추출에 사용했던 동결 모델의 상위 층 몇 개를 동결에서 해제하고 모델에 새로 추가한 층과 함께 훈련하는 것
# 합성곱 기반 층에 있는 하위 층들은 좀 더 일반적이고 재사용 가능한 특성들을 인코딩 하므로 재사용 가능

base_dir = '/content/cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, # 랜덤하게 사진을 회전시킬 각도 범위
    width_shift_range=0.2, # 사진을 수평으로 랜덤하게 평행이동 
    height_shift_range=0.2, # 사진을 수직으로 랜덤하게 평행이동
    shear_range=0.2, # 랜덤하게 전단 변환을 적용할 각도 범위
    zoom_range=0.2, # 랜덤하게 사진을 확대할 범위
    horizontal_flip=True, # 램덤하게 이미지를 수평으로 뒤집는다.
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size= 20,
    class_mode='binary',
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150, 150, 3))

# 합성곱 기반 층에 의해 사전에 학습된 표현이 훈련하는 동안 수정되지 않도록 conv_base를 동결
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
  if layer.name == 'block5_conv1':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False

input = Input(shape = (150, 150, 3,))
features = conv_base(input)
R = Flatten()(features)
R = Dense(256)(R)
R = Activation('relu')(R)
R = Dense(1)(R)
R = Activation('sigmoid')(R)

model = Model(inputs = [input], outputs = R)
model.compile(optimizer = RMSprop(lr=2e-5),
              loss = 'binary_crossentropy',
              metrics = ['acc'])

history = model.fit(train_generator,
                    steps_per_epoch = 100,
                    epochs = 100,
                    validation_data = validation_generator,
                    validation_steps=50)

def smooth_curve(points, factor = 0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1-factor))
    else:
      smoothed_points.append(point)

  return smoothed_points

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, smooth_curve(acc), 'bp', label='Training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bp', label='Training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
