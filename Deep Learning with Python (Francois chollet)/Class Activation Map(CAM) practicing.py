import os, shutil
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from keras import backend as K # 케라스 백엔드 함수
import cv2

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

# ============= OpenCV를 사용하여 앞에서 얻은 히트맵에 원본 이미지를 겹친 이미지 만들기 ============= #

img = cv2.imread(img_path)  # cv2 모듈을 사용하여 원본 이미지를 로드한다.
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)  # heatmap을 RGB 형식으로 포맷
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img
cv2.imwrite('/content/elephant_cam.jpg', superimposed_img)

