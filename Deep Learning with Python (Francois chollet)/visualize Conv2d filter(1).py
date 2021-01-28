
import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from keras import backend as K # 케라스 백엔드 함수

img_path = '/1.jpg'
model = VGG16(weights='imagenet', include_top=False, input_shape = (150, 150, 3))

img = image.load_img(img_path, target_size=(150, 150, 3))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

layer_outputs = [layer.output for layer in model.layers[:8]]  # 레이아웃의 output을 모아 리스트로 반환
activation_model = Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(img_tensor)

first_layer_activation = activations[2]
plt.matshow(first_layer_activation[0, :, :, 5], cmap = 'viridis')
plt.colorbar()

# ================= 네트워크의 모든 활성화를 출력 ================= #

layer_names = []
for layer in model.layers[:8]:
  layer_names.append(layer.name)

images_per_row = 16
