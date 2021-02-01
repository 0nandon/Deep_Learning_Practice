
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# ================ one-hot encoding ================ #

samples = ['The cat sat on the mat.', 'The dog ate my homework']

token_index =  {}
for sample in samples:
  for word in sample.split():
    if word not in token_index:
      token_index[word] = len(token_index) + 1
    
max_length = 10

results = np.zeros(shape=(len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
  for j, word in list(enumerate(samples[i].split()))[:max_length]:
    index = token_index.get(word)
    results[i, j, index] = 1

# ================ one-hot encoding (using keras library) ================ #

tokenizer = Tokenizer(num_words=1000)  # 가장 빈도가 높은 1000개의 단어만 선택하도록 Tokenizer 객체를 생성
tokenizer.fit_on_texts(samples)  # 단어 인덱스를 구축
sequences = tokenizer.texts_to_sequences(samples)  # 문자열을 정수 인덱스의 리스트로 반환

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 원-핫 이진 벡터 효현으로 나타낸다.


# ================ one-hot hashing (using keras library) ================ #

dimensionality = 1000

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
  for j, word in list(enumerate(sample.split()))[:max_length]:
    index = abs(hash(word)) % dimensionality
    results[i, j, index] = 1

