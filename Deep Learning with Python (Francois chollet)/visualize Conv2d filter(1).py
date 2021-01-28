
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

layer_names.append(layer.name)

for layer in model.layers[:8]:
  layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names,  activations):
  
  n_features = layer_activation.shape[-1]
  if n_features < 16:
    continue

  size = layer_activation.shape[1]
  n_cols = n_features // images_per_row

  display_grid = np.zeros((size * n_cols, images_per_row * size))
  
  for col in range(n_cols):
    for row in range(images_per_row):
      channel_image = layer_activation[0, :, :, col * images_per_row + row]

      # 그래프로 나타내기 좋게 특성을 처리한다.
      channel_image -= channel_image.mean()
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')

      display_grid[col * size : (col+1) * size, row * size : (row+1) * size] = channel_image

  scale = 1. / size

  plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
  plt.title(layer_name)
  plt.grid(False)
  plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
