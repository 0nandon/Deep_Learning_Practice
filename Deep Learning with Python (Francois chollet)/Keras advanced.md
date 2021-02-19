
## 다중 입력 모델
keras는 함수형 API(functional api)를 통해 다중 입력 모델을 구현할 수 있다. 아래와 같은 함수들이 사용된다.
* keras.layers.add
* keras.layers.concatenate

아래는 예시코드이다.
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, concatenate, Dense, Activation

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape = (None,), dtype='int32', name='text')
embedded_text = Embedding(question_vocabulary_size, 64)(text_input)
encoded_text = LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='text')
embedded_question = Embedding(question_input, 32)(question_input)
encoded_question = LSTM(16)(question_input)

concatenated = concatenate([encoded_text, encoded_question])
answer = Dense(1)(answer_vocabulary_size)(concatenate)
answer = Activation('softmax')(answer)

model = Model(inputs = [text_input, quesiton_input], outputs = answer)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics=['acc'])
```

## 다중 출력 모델
다중 출력 모델 또한 구현할 수 있다. 보통 출력 마다 다른 손실 함수를 지정합니다. 손실 함수를 지정하면, 함수에 따라 결과값의
스케일이 달라지는데, 이러한 스케일을 맞추기 위해 각 손실 함수 값에 가중치를 부여한다. 예를들어 MSE(평균 제곱 오차)는 평균적으로 3 ~ 5 사이의 값을 가지는 반면,
크로스 엔트로피 손실은 0.1로 그 값이 매우 낮다. 딥러닝 모델은 손실 값이 큰 작업에 치우쳐 학습을 하기 때문에 

아래는 예시 코드이다.
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation

vocabulary_size = 50000
num_income_groups = 0

posts_input = Input(shape = (None,), dtype='int32', name='posts')
embedded_posts = Embedding(vocabulary_size, 256)(posts_inuts)
x = Conv1D(128, 5, activation='relu')(embedded_posts)
x = MaxPooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Maxpooling1D(5)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = Conv1D(256, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)

# 다중 출력
age_prediction = Dense(1, name='age')(x)
income_prediction = Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_predictions, income_predictions, gender_predictions])
model.compile(optimizer='rmsprop',
              loss = ['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights = [0.25, 1., 10.])
```
