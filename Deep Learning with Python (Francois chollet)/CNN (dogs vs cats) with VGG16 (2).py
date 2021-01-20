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
    batch_size= 32,
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
conv_base.trainable = False

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
                    epochs = 30,
                    batch_size = 20,
                    validation_data = validation_generator,
                    validation_steps=50,
                    verbose=2)

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
