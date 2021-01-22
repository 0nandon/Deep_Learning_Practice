import os, shutil
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
from keras import backend as K # 케라스 백엔드 함수

model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0 # block3_conv1 층의 첫 번째 채널의 필터에 대하여 경사 상승법을 시행한다.

# 결과로 나온 이미지 텐서를 출력 가능한 이미지로 변경하기 위해 후처리할 필요가 있다.
def deprocess_image(x):
  x -= x.mean()
  x /= (x.std() + 1e-5)
  x *= 0.1

  x += 0.5
  x = np.clip(x, 0, 1)

  x *= 255
  x = np.clip(x, 0, 255).astype('uint8')
  return x

print(model.get_layer('block3_conv1').output)

tf.compat.v1.disable_eager_execution()
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # 경사 상승법 과정을 부드럽게 하기 위해, 그래디언트 텐서를 L2 노름으로 나누어 정규화한다.
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128. # 잡음이 섞인 회색 이미지로 시작한다.
    
    # 필터에 대한 응답을 최대화 하기 위해 경사 상승법을 시행한다.
    step = 1.
    for _ in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * 1.
    
    img = input_img_data[0]
    return deprocess_image(img) 

plt.imshow(generate_pattern('block3_conv1', 0))
