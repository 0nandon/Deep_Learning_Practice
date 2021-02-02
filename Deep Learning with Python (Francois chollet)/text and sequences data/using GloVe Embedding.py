
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


imdb_dir = './datasets/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

# 훈련용 리뷰 하나를 문자열 하나로 만들고, neg/pos 라벨링도 한다.
for label_type in ['neg', 'pos']:
  dir_name = os.path.join(train_dir, label_type)
  for fname in os.listdir(dir_name):
    if fname[-4:] == '.txt':
      f = open(os.path.join(dir_name, fname), encoding="utf8")
      texts.append(f.read())
      f.close()
      if label_type == 'neg':
        labels.append(0)
      else:
        labels.append(1)

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

# 문자열을 숫자로 인덱싱한다. # 문자열을 숫자로 인덱싷
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=maxlen)

word_index = tokenizer.word_index

# 훈련용 데이터와 검증 데이터를 나눈다.
indices = np.arrange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = data[training_samples: training_samples + validation_samples]

# Glove 단어 임베딩 내려받기
glove_dir = './datasets/'

"""
embedding_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
"""

# Embedding층에 주입할 수 있도록 임베딩 행렬을 만든다.
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
  if i < max_words:
    embedding_vector = embedding_index[word]
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector

input = Input((maxlen,))
R = Embedding((max_words, embedding_dim))(input)
R = Flatten(R)
R = Dense(32)(R)
R = Activation('relu')(R)
R = Dense(1)(R)
R = Activation('sigmoid')(R)

model = Model(inputs = [input], output = R)

# model의 첫 번째 가중치 층을 embedding_matrix로 교체한다.
model.layers[0].set_weigths([embedding_matrix])
model.layers[0].trainable = False # 동결한다.

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', 
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

model.save_weights('pre_trained_glove_model.h5')
