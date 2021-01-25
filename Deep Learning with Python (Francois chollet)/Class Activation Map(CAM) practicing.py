import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from keras import backend as K # 케라스 백엔드 함수

model = VGG16(weights='imagenet', include_top=True)

img_path = '/content/creative_commons_elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)  # (224, 224, 3) 크기의 넘파이 float32 배열
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

"""
# 사전 훈련된 네트워크로 테스트
preds = model.predict(x)
preds.shape

print('Predicted:', decode_predictions(preds, top=3)[0])
"""

african_elephant_output = model.output[:, 386]

last_conv_layer = model.get_layer('block5_conv3')
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
  conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis = -1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

